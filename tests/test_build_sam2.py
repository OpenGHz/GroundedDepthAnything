from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build-sam2.py"
SPEC = importlib.util.spec_from_file_location("gda_build_sam2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_sam2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_sam2)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    source = tmp_path / "grounded-sam-2"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    (source / ".gitignore").write_text("ignored.bin\n", encoding="utf-8")
    (source / "setup.py").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    return source


@pytest.fixture
def patch_file(tmp_path: Path) -> Path:
    patch = tmp_path / "sam2.patch"
    patch.write_text(
        "diff --git a/setup.py b/setup.py\n"
        "--- a/setup.py\n"
        "+++ b/setup.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+patched\n",
        encoding="utf-8",
    )
    return patch


def _marker(**overrides: str) -> dict:
    values = {
        "commit": "a" * 40,
        "patch_sha256": "b" * 64,
        "torch_version": "2.10.0+cu130",
        "torch_cuda": "13.0",
        "gpu_arch": "10.3",
        "pixi_lock_sha256": "c" * 64,
        "nvcc": "/usr/local/cuda/bin/nvcc: release 13.0",
        "build_script_sha256": "d" * 64,
    }
    values.update(overrides)
    return build_sam2.build_marker(build_sam2.build_identity(**values))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("commit", "c" * 40),
        ("patch_sha256", "d" * 64),
        ("torch_version", "2.10.1+cu130"),
        ("torch_cuda", "12.8"),
        ("gpu_arch", "9.0"),
        ("pixi_lock_sha256", "e" * 64),
        ("nvcc", "/opt/cuda/bin/nvcc: release 12.8"),
        ("build_script_sha256", "f" * 64),
    ],
)
def test_build_key_changes_for_required_identity_fields(field: str, replacement: str) -> None:
    assert _marker()["key"] != _marker(**{field: replacement})["key"]


def test_clean_source_allows_ignored_artifacts_but_rejects_changes(source_repo: Path) -> None:
    (source_repo / "ignored.bin").write_text("artifact", encoding="utf-8")
    build_sam2.ensure_clean_source(source_repo)

    (source_repo / "setup.py").write_text("modified\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-ignored changes"):
        build_sam2.ensure_clean_source(source_repo)


def test_prepare_build_tree_is_isolated_and_reusable(
    source_repo: Path,
    patch_file: Path,
    tmp_path: Path,
) -> None:
    (source_repo / "ignored.bin").write_text("artifact", encoding="utf-8")
    marker = _marker(commit=_git(source_repo, "rev-parse", "HEAD"))
    build_root = tmp_path / "cache" / "sam2"

    build_tree = build_sam2.prepare_build_tree(
        source_repo,
        build_root,
        patch_file,
        marker,
    )

    assert build_tree == build_root / marker["key"]
    assert (build_tree / "setup.py").read_text(encoding="utf-8") == "patched\n"
    stored_marker = json.loads((build_tree / build_sam2.BUILD_MARKER).read_text())
    assert stored_marker["schema_version"] == marker["schema_version"]
    assert stored_marker["key"] == marker["key"]
    assert stored_marker["identity"] == marker["identity"]
    assert stored_marker["tracked_source_paths"] == [".gitignore", "setup.py"]
    assert not (build_tree / ".git").exists()
    assert not (build_tree / "ignored.bin").exists()
    assert (source_repo / "setup.py").read_text(encoding="utf-8") == "old\n"
    assert _git(source_repo, "status", "--porcelain") == ""

    generated = build_tree / "sam2" / "_C.so"
    generated.parent.mkdir()
    generated.touch()
    assert build_sam2.prepare_build_tree(source_repo, build_root, patch_file, marker) == build_tree
    assert generated.exists()

    (build_tree / "setup.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source was modified"):
        build_sam2.prepare_build_tree(source_repo, build_root, patch_file, marker)


def test_prepare_build_tree_does_not_overwrite_unrecognized_directory(
    source_repo: Path,
    patch_file: Path,
    tmp_path: Path,
) -> None:
    marker = _marker(commit=_git(source_repo, "rev-parse", "HEAD"))
    build_tree = tmp_path / "cache" / marker["key"]
    build_tree.mkdir(parents=True)
    sentinel = build_tree / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to overwrite unrecognized"):
        build_sam2.prepare_build_tree(
            source_repo,
            build_tree.parent,
            patch_file,
            marker,
        )

    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"


def test_prepare_build_tree_rejects_mismatched_marker(
    source_repo: Path,
    patch_file: Path,
    tmp_path: Path,
) -> None:
    marker = _marker(commit=_git(source_repo, "rev-parse", "HEAD"))
    build_tree = tmp_path / "cache" / marker["key"]
    build_tree.mkdir(parents=True)
    mismatched = {**marker, "schema_version": marker["schema_version"] + 1}
    (build_tree / build_sam2.BUILD_MARKER).write_text(json.dumps(mismatched))

    with pytest.raises(RuntimeError, match="mismatched marker"):
        build_sam2.prepare_build_tree(
            source_repo,
            build_tree.parent,
            patch_file,
            marker,
        )


def test_failed_patch_removes_only_new_build_directory(
    source_repo: Path,
    tmp_path: Path,
) -> None:
    bad_patch = tmp_path / "bad.patch"
    bad_patch.write_text("not a patch\n", encoding="utf-8")
    marker = _marker(commit=_git(source_repo, "rev-parse", "HEAD"))
    build_root = tmp_path / "cache"

    with pytest.raises(RuntimeError, match="Cannot apply"):
        build_sam2.prepare_build_tree(source_repo, build_root, bad_patch, marker)

    assert not (build_root / marker["key"]).exists()


def test_install_uses_isolated_editable_tree_and_cuda_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, *, check, env):
        assert check is True
        calls.append((command, env))

    monkeypatch.setattr(build_sam2.subprocess, "run", fake_run)
    build_tree = tmp_path / "isolated"
    build_sam2.install_build_tree(build_tree, "10.3")

    command, env = calls[0]
    assert command == [
        build_sam2.sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-deps",
        "-e",
        str(build_tree),
    ]
    assert env["TORCH_CUDA_ARCH_LIST"] == "10.3"
    assert env["SAM2_BUILD_CUDA"] == "1"
    assert env["SAM2_BUILD_ALLOW_ERRORS"] == "0"
