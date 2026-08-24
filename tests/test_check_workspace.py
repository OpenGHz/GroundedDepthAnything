from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check-workspace.py"
SPEC = importlib.util.spec_from_file_location("check_workspace_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_workspace_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_workspace_script)

NO_URL_REQUIREMENTS: dict[Path, str] = {}


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=GDA Tests",
            "-c",
            "user.email=gda-tests@example.invalid",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(repository),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _create_repository(path: Path, filename: str, *, two_commits: bool = False) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "--quiet")
    (path / ".gitignore").write_text("checkpoints/\nbuild/\n", encoding="utf-8")
    (path / filename).write_text("first\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "--quiet", "-m", "initial")
    if two_commits:
        (path / filename).write_text("second\n", encoding="utf-8")
        _git(path, "commit", "--quiet", "-am", "second")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    sources = tmp_path / "sources"
    sam3 = sources / "sam3"
    depth = sources / "depth-anything-3"
    grounded_sam2 = sources / "grounded-sam-2"
    salad = sources / "salad"
    _create_repository(sam3, "sam3.py", two_commits=True)
    _create_repository(depth, "depth.py")
    _create_repository(grounded_sam2, "sam2.py")
    _create_repository(salad, "salad.py")

    _git(depth, "submodule", "add", "--quiet", str(salad), "da3_streaming/loop_utils/salad")
    _git(depth, "commit", "--quiet", "-am", "add nested submodule")

    root = tmp_path / "gda"
    root.mkdir()
    _git(root, "init", "--quiet")
    (root / "README.md").write_text("GDA test workspace\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "--quiet", "-m", "initial")
    for source, relative_path in (
        (sam3, "third_party/sam3"),
        (depth, "third_party/depth-anything-3"),
        (grounded_sam2, "third_party/grounded-sam-2"),
    ):
        _git(root, "submodule", "add", "--quiet", str(source), relative_path)
    _git(root, "commit", "--quiet", "-am", "add model submodules")
    _git(root, "submodule", "update", "--init", "--recursive")
    return root


def test_checks_index_pins_recursively_and_ignores_artifacts(workspace: Path) -> None:
    sam3 = workspace / "third_party/sam3"
    (sam3 / "checkpoints").mkdir()
    (sam3 / "checkpoints/model.pt").write_bytes(b"checkpoint")
    (sam3 / "build").mkdir()
    (sam3 / "build/extension.so").write_bytes(b"extension")

    errors, checked = check_workspace_script.check_workspace(
        workspace, expected_urls=NO_URL_REQUIREMENTS
    )

    assert errors == []
    assert checked == 4


def test_reports_checkout_that_differs_from_superproject_index(workspace: Path) -> None:
    sam3 = workspace / "third_party/sam3"
    expected = _git(workspace, "rev-parse", ":third_party/sam3")
    _git(sam3, "checkout", "--quiet", "HEAD^")
    actual = _git(sam3, "rev-parse", "HEAD")

    errors, _ = check_workspace_script.check_workspace(
        workspace, expected_urls=NO_URL_REQUIREMENTS
    )

    assert any(
        f"third_party/sam3: expected {expected} from parent index, found {actual}" in error
        for error in errors
    )


def test_reports_uninitialized_nested_submodule(workspace: Path) -> None:
    depth = workspace / "third_party/depth-anything-3"
    nested = Path("da3_streaming/loop_utils/salad")
    _git(depth, "submodule", "deinit", "--force", "--", nested.as_posix())

    errors, checked = check_workspace_script.check_workspace(
        workspace, expected_urls=NO_URL_REQUIREMENTS
    )

    assert checked == 3
    assert any(
        f"third_party/depth-anything-3/{nested}: not initialized" in error for error in errors
    )

    metadata_errors, metadata_checked = check_workspace_script.check_workspace(
        workspace,
        expected_urls=NO_URL_REQUIREMENTS,
        metadata_only=True,
    )
    assert metadata_errors == []
    assert metadata_checked == 0


def test_reports_non_ignored_dirty_submodule(workspace: Path) -> None:
    sam3 = workspace / "third_party/sam3"
    (sam3 / "scratch.txt").write_text("local work\n", encoding="utf-8")

    errors, _ = check_workspace_script.check_workspace(
        workspace, expected_urls=NO_URL_REQUIREMENTS
    )

    assert any(
        error.startswith("third_party/sam3: dirty worktree/index:") and "?? scratch.txt" in error
        for error in errors
    )


def test_source_archive_has_actionable_error(tmp_path: Path) -> None:
    archive = tmp_path / "gda-source-archive"
    archive.mkdir()

    with pytest.raises(check_workspace_script.WorkspaceError, match="source archives") as exc_info:
        check_workspace_script.check_workspace(archive, expected_urls=NO_URL_REQUIREMENTS)

    assert "git clone --recurse-submodules" in str(exc_info.value)


def test_reports_noncanonical_worktree_and_staged_urls(workspace: Path) -> None:
    sam3_path = Path("third_party/sam3")
    expected_url = "https://github.com/facebookresearch/sam3.git"
    expected_urls = {sam3_path: expected_url}

    errors, _ = check_workspace_script.check_workspace(workspace, expected_urls=expected_urls)
    assert any("worktree .gitmodules" in error for error in errors)
    assert any("staged .gitmodules" in error for error in errors)

    _git(
        workspace,
        "config",
        "-f",
        ".gitmodules",
        "submodule.third_party/sam3.url",
        expected_url,
    )
    _git(workspace, "add", ".gitmodules")

    errors, _ = check_workspace_script.check_workspace(workspace, expected_urls=expected_urls)
    assert not any(".gitmodules" in error for error in errors)
