"""Build SAM2's CUDA extension from an isolated copy of the pinned submodule."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM2_ROOT = PROJECT_ROOT / "third_party" / "grounded-sam-2"
SAM2_PATCH = PROJECT_ROOT / "scripts" / "patches" / "grounded-sam2-cuda-arch.patch"
BUILD_ROOT = PROJECT_ROOT / ".pixi" / "gda-build" / "sam2"
BUILD_MARKER = ".gda-build.json"
BUILD_SCHEMA_VERSION = 1
PIXI_LOCK = PROJECT_ROOT / "pixi.lock"


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Cannot inspect SAM2 source at {root}: {details}")
    return result.stdout


def ensure_clean_source(source_root: Path = SAM2_ROOT) -> None:
    """Reject changes that would make a commit-keyed source copy ambiguous."""

    status = _git_output(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise RuntimeError(
            "Grounded-SAM-2 submodule has non-ignored changes; refusing to create a "
            f"commit-keyed build copy from {source_root}:\n{status.rstrip()}"
        )


def source_commit(source_root: Path = SAM2_ROOT) -> str:
    return _git_output(source_root, "rev-parse", "--verify", "HEAD^{commit}").strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_arch() -> str:
    configured = os.environ.get("GDA_CUDA_ARCH")
    if configured:
        return configured
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}"
    except (ImportError, RuntimeError):
        pass
    return os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0")


def torch_identity() -> tuple[str, str | None]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch must be installed before building SAM2") from exc
    return str(torch.__version__), torch.version.cuda


def nvcc_identity() -> str | None:
    """Return the selected CUDA compiler identity when it is discoverable."""

    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        return None
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc" if CUDA_HOME else None
    if nvcc is None or not nvcc.is_file():
        nvcc_path = shutil.which("nvcc")
        nvcc = Path(nvcc_path) if nvcc_path else None
    if nvcc is None:
        return None
    if not nvcc.is_file():
        return None
    result = subprocess.run(
        [str(nvcc), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return f"{nvcc}: unavailable ({result.returncode})"
    return f"{nvcc}: {result.stdout.strip()}"


def build_identity(
    *,
    commit: str,
    patch_sha256: str,
    torch_version: str,
    torch_cuda: str | None,
    gpu_arch: str,
    pixi_lock_sha256: str,
    nvcc: str | None,
    build_script_sha256: str,
) -> dict[str, Any]:
    """Return all inputs that can affect compatibility of the cached extension."""

    return {
        "source_commit": commit,
        "patch_sha256": patch_sha256,
        "torch_version": torch_version,
        "torch_cuda": torch_cuda,
        "gpu_arch": gpu_arch,
        "pixi_lock_sha256": pixi_lock_sha256,
        "nvcc": nvcc,
        "build_script_sha256": build_script_sha256,
        "compiler": {
            "cc": os.environ.get("CC"),
            "cxx": os.environ.get("CXX"),
        },
        "python_cache_tag": sys.implementation.cache_tag,
    }


def build_key(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_marker(
    identity: dict[str, Any],
    *,
    source_manifest_sha256: str | None = None,
    tracked_source_paths: list[str] | None = None,
) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "key": build_key(identity),
        "identity": identity,
    }
    if source_manifest_sha256 is not None:
        marker["source_manifest_sha256"] = source_manifest_sha256
    if tracked_source_paths is not None:
        marker["tracked_source_paths"] = tracked_source_paths
    return marker


def _source_manifest(build_tree: Path, tracked_paths: list[str]) -> str:
    """Hash committed source files, excluding generated editable-build artifacts."""

    digest = hashlib.sha256()
    for raw_path in sorted(tracked_paths):
        path = Path(raw_path)
        source = build_tree / path
        if source.is_symlink():
            content = b"symlink:" + os.readlink(source).encode()
        elif source.is_file():
            content = source.read_bytes()
        else:
            raise RuntimeError(f"Tracked SAM2 source file is missing from build tree: {source}")
        digest.update(raw_path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _committed_paths(source_root: Path, commit: str) -> list[str]:
    output = _git_output(source_root, "ls-tree", "-r", "--name-only", commit)
    paths = [line for line in output.splitlines() if line]
    if not paths:
        raise RuntimeError(f"SAM2 source commit has no tracked files: {commit}")
    return paths


def _copy_committed_source(
    source_root: Path,
    destination: Path,
    commit: str,
) -> None:
    """Export an exact commit without .git or ignored build artifacts."""

    archive_path = destination / ".gda-source.tar"
    output = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "archive",
            "--format=tar",
            f"--output={archive_path}",
            commit,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if output.returncode != 0:
        details = output.stderr.strip() or output.stdout.strip()
        raise RuntimeError(
            f"Cannot export SAM2 source commit {commit} from {source_root}: {details}"
        )
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            archive.extractall(destination, filter="data")
    finally:
        archive_path.unlink(missing_ok=True)


def _apply_patch(build_tree: Path, patch_path: Path) -> None:
    command = ["git", "apply", "--no-index", "--unidiff-zero"]
    check = subprocess.run(
        [*command, "--check", str(patch_path)],
        cwd=build_tree,
        check=False,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            f"Cannot apply the pinned SAM2 CUDA patch to isolated copy {build_tree}:\n"
            f"{check.stderr.strip()}"
        )
    subprocess.run([*command, str(patch_path)], cwd=build_tree, check=True)


def _validate_existing_build(
    build_tree: Path,
    expected_marker: dict[str, Any],
    patch_path: Path,
) -> None:
    marker_path = build_tree / BUILD_MARKER
    if not build_tree.is_dir() or build_tree.is_symlink():
        raise RuntimeError(f"SAM2 build cache path is not a regular directory: {build_tree}")
    if not marker_path.is_file() or marker_path.is_symlink():
        raise RuntimeError(
            f"Refusing to overwrite unrecognized SAM2 build directory {build_tree}; "
            f"marker {BUILD_MARKER} is missing or invalid"
        )
    try:
        actual_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid SAM2 build marker at {marker_path}: {exc}") from exc
    if (
        actual_marker.get("schema_version") != expected_marker.get("schema_version")
        or actual_marker.get("key") != expected_marker.get("key")
        or actual_marker.get("identity") != expected_marker.get("identity")
    ):
        raise RuntimeError(
            f"Refusing to overwrite SAM2 build directory with mismatched marker: {build_tree}"
        )

    tracked_paths = actual_marker.get("tracked_source_paths")
    expected_source_manifest = actual_marker.get("source_manifest_sha256")
    if not isinstance(tracked_paths, list) or not all(
        isinstance(path, str) for path in tracked_paths
    ):
        raise RuntimeError(f"SAM2 build marker has no valid source manifest: {marker_path}")
    if not isinstance(expected_source_manifest, str):
        raise RuntimeError(f"SAM2 build marker has no source manifest hash: {marker_path}")
    actual_source_manifest = _source_manifest(build_tree, tracked_paths)
    if actual_source_manifest != expected_source_manifest:
        raise RuntimeError(f"SAM2 isolated source was modified after preparation: {build_tree}")

    reverse_check = subprocess.run(
        [
            "git",
            "apply",
            "--no-index",
            "--unidiff-zero",
            "--reverse",
            "--check",
            str(patch_path),
        ],
        cwd=build_tree,
        check=False,
        capture_output=True,
        text=True,
    )
    if reverse_check.returncode != 0:
        raise RuntimeError(
            "SAM2 build directory marker matches but its patched source is invalid: "
            f"{build_tree}\n"
            f"{reverse_check.stderr.strip()}"
        )


def prepare_build_tree(
    source_root: Path,
    build_root: Path,
    patch_path: Path,
    marker: dict[str, Any],
) -> Path:
    """Create or safely reuse an isolated, patched SAM2 source tree."""

    build_tree = build_root / marker["key"]
    if build_tree.exists() or build_tree.is_symlink():
        _validate_existing_build(build_tree, marker, patch_path)
        return build_tree

    build_root.mkdir(parents=True, exist_ok=True)
    commit = marker.get("identity", {}).get("source_commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError("SAM2 build marker is missing identity.source_commit")
    created = False
    try:
        build_tree.mkdir()
        created = True
        _copy_committed_source(source_root, build_tree, commit)
        _apply_patch(build_tree, patch_path)
        tracked_paths = _committed_paths(source_root, commit)
        complete_marker = build_marker(
            marker["identity"],
            source_manifest_sha256=_source_manifest(build_tree, tracked_paths),
            tracked_source_paths=tracked_paths,
        )
        (build_tree / BUILD_MARKER).write_text(
            json.dumps(complete_marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        if created:
            shutil.rmtree(build_tree)
        raise
    return build_tree


def install_build_tree(build_tree: Path, arch: str) -> None:
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = arch
    env["SAM2_BUILD_CUDA"] = "1"
    env["SAM2_BUILD_ALLOW_ERRORS"] = "0"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(build_tree),
        ],
        check=True,
        env=env,
    )


def main() -> None:
    if not SAM2_PATCH.is_file():
        raise FileNotFoundError(f"Missing pinned SAM2 build patch: {SAM2_PATCH}")
    ensure_clean_source()
    arch = detect_arch()
    torch_version, torch_cuda = torch_identity()
    identity = build_identity(
        commit=source_commit(),
        patch_sha256=sha256_file(SAM2_PATCH),
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        gpu_arch=arch,
        pixi_lock_sha256=sha256_file(PIXI_LOCK),
        nvcc=nvcc_identity(),
        build_script_sha256=sha256_file(Path(__file__)),
    )
    marker = build_marker(identity)
    build_tree = prepare_build_tree(SAM2_ROOT, BUILD_ROOT, SAM2_PATCH, marker)
    print(
        "building SAM2 CUDA extension "
        f"for TORCH_CUDA_ARCH_LIST={arch} from isolated copy {build_tree}"
    )
    install_build_tree(build_tree, arch)


if __name__ == "__main__":
    main()
