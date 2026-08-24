"""Validate the Git submodules required by GDA's locked environment."""

from __future__ import annotations

import configparser
import io
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SUBMODULES = (
    Path("third_party/sam3"),
    Path("third_party/depth-anything-3"),
    Path("third_party/grounded-sam-2"),
)
EXPECTED_SUBMODULE_URLS = {
    Path("third_party/sam3"): "https://github.com/facebookresearch/sam3.git",
    Path("third_party/depth-anything-3"): (
        "https://github.com/ByteDance-Seed/Depth-Anything-3.git"
    ),
    Path("third_party/grounded-sam-2"): ("https://github.com/IDEA-Research/Grounded-SAM-2.git"),
}


class WorkspaceError(RuntimeError):
    """Raised when the superproject itself cannot be inspected."""


def _git(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise WorkspaceError("Git is not installed or is not available on PATH") from exc

    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        command = " ".join(("git", *arguments))
        raise WorkspaceError(f"`{command}` failed in {repository}: {detail}")
    return result.stdout


def _repository_root(repository: Path) -> Path:
    return Path(_git(repository, "rev-parse", "--show-toplevel").strip()).resolve()


def _gitlinks(repository: Path) -> dict[Path, str]:
    """Return stage-zero gitlinks and their expected object IDs from the index."""
    output = _git(repository, "ls-files", "--stage", "-z")
    links: dict[Path, str] = {}
    for record in output.split("\0"):
        if not record:
            continue
        metadata, path = record.split("\t", maxsplit=1)
        mode, object_id, stage = metadata.split()
        if mode == "160000" and stage == "0":
            links[Path(path)] = object_id
    return links


def _dirty_summary(repository: Path) -> list[str]:
    # Porcelain status honors every repository's ignore rules. In particular,
    # ignored checkpoints and build outputs do not make a submodule look dirty.
    output = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
        "--ignore-submodules=none",
    )
    return output.splitlines()


def _parse_gitmodules(content: str, source: str) -> dict[Path, str]:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_file(io.StringIO(content), source=source)
    except configparser.Error as exc:
        raise WorkspaceError(f"cannot parse {source}: {exc}") from exc

    modules: dict[Path, str] = {}
    for section in parser.sections():
        if not section.startswith("submodule "):
            continue
        if not parser.has_option(section, "path") or not parser.has_option(section, "url"):
            raise WorkspaceError(f"{source}: {section} must define both path and url")
        path = Path(parser.get(section, "path"))
        if path in modules:
            raise WorkspaceError(f"{source}: duplicate submodule path {path}")
        modules[path] = parser.get(section, "url")
    return modules


def _check_gitmodule_urls(
    project_root: Path,
    expected_urls: dict[Path, str],
    errors: list[str],
) -> None:
    gitmodules_path = project_root / ".gitmodules"
    try:
        worktree_modules = _parse_gitmodules(
            gitmodules_path.read_text(encoding="utf-8"),
            str(gitmodules_path),
        )
    except OSError as exc:
        errors.append(f"cannot read {gitmodules_path}: {exc}")
        worktree_modules = {}

    try:
        index_modules = _parse_gitmodules(
            _git(project_root, "show", ":.gitmodules"),
            "the staged .gitmodules",
        )
    except WorkspaceError as exc:
        errors.append(str(exc))
        index_modules = {}

    for path, expected_url in expected_urls.items():
        for label, modules in (
            ("worktree .gitmodules", worktree_modules),
            ("staged .gitmodules", index_modules),
        ):
            actual_url = modules.get(path)
            if actual_url != expected_url:
                errors.append(f"{label}: {path} must use {expected_url}, found {actual_url!r}")


def _format_status(lines: list[str], limit: int = 8) -> str:
    visible = lines[:limit]
    summary = "; ".join(visible)
    if len(lines) > limit:
        summary += f"; ... ({len(lines) - limit} more entries)"
    return summary


def _check_submodule(
    repository: Path,
    relative_path: Path,
    expected: str,
    display_path: Path,
    errors: list[str],
) -> int:
    path = repository / relative_path
    git_marker = path / ".git"
    if not path.is_dir() or not git_marker.exists():
        errors.append(
            f"{display_path}: not initialized (index expects {expected}); "
            "run `git submodule update --init --recursive`"
        )
        return 0

    try:
        actual_root = _repository_root(path)
    except WorkspaceError as exc:
        errors.append(f"{display_path}: cannot inspect initialized repository: {exc}")
        return 0

    if actual_root != path.resolve():
        errors.append(
            f"{display_path}: checkout resolves to the wrong Git worktree "
            f"({actual_root}, expected {path.resolve()})"
        )
        return 0

    try:
        actual = _git(path, "rev-parse", "--verify", "HEAD^{commit}").strip()
    except WorkspaceError as exc:
        errors.append(f"{display_path}: cannot resolve HEAD: {exc}")
        return 0

    if actual != expected:
        errors.append(f"{display_path}: expected {expected} from parent index, found {actual}")

    try:
        dirty = _dirty_summary(path)
    except WorkspaceError as exc:
        errors.append(f"{display_path}: cannot inspect worktree status: {exc}")
    else:
        if dirty:
            errors.append(f"{display_path}: dirty worktree/index: {_format_status(dirty)}")

    try:
        nested_links = _gitlinks(path)
    except (ValueError, WorkspaceError) as exc:
        errors.append(f"{display_path}: cannot inspect nested gitlinks: {exc}")
        return 1

    checked = 1
    for nested_path, nested_expected in sorted(
        nested_links.items(), key=lambda item: item[0].as_posix()
    ):
        checked += _check_submodule(
            path,
            nested_path,
            nested_expected,
            display_path / nested_path,
            errors,
        )
    return checked


def check_workspace(
    project_root: Path = PROJECT_ROOT,
    expected_urls: dict[Path, str] | None = None,
    *,
    metadata_only: bool = False,
) -> tuple[list[str], int]:
    """Return validation errors and the number of initialized submodules checked."""
    project_root = project_root.resolve()
    try:
        actual_root = _repository_root(project_root)
    except WorkspaceError as exc:
        raise WorkspaceError(
            f"{project_root} is not a Git checkout. This check needs the superproject "
            "index, which is absent from source archives. Clone GDA with "
            f"`git clone --recurse-submodules <url> {project_root.name}`. ({exc})"
        ) from exc

    if actual_root != project_root:
        raise WorkspaceError(
            f"{project_root} is not the root of its own Git checkout "
            f"(Git resolved {actual_root}). Source archives and copied working trees "
            "cannot provide the recorded submodule revisions; use "
            "`git clone --recurse-submodules` instead."
        )

    try:
        root_links = _gitlinks(project_root)
    except (ValueError, WorkspaceError) as exc:
        raise WorkspaceError(f"cannot read the GDA superproject index: {exc}") from exc

    errors: list[str] = []
    _check_gitmodule_urls(
        project_root,
        EXPECTED_SUBMODULE_URLS if expected_urls is None else expected_urls,
        errors,
    )
    for relative_path in REQUIRED_SUBMODULES:
        if relative_path not in root_links:
            errors.append(
                f"{relative_path}: required path is not a stage-zero gitlink in the "
                "GDA superproject index"
            )
    if metadata_only:
        return errors, 0

    checked = 0
    for relative_path in REQUIRED_SUBMODULES:
        expected = root_links.get(relative_path)
        if expected is None:
            continue
        checked += _check_submodule(
            project_root,
            relative_path,
            expected,
            relative_path,
            errors,
        )
    return errors, checked


def main() -> None:
    arguments = sys.argv[1:]
    unknown = [argument for argument in arguments if argument != "--metadata-only"]
    if unknown:
        raise SystemExit(f"Unknown argument(s): {' '.join(unknown)}")
    metadata_only = "--metadata-only" in arguments
    try:
        errors, checked = check_workspace(metadata_only=metadata_only)
    except WorkspaceError as exc:
        raise SystemExit(f"Workspace submodule check failed:\n  - {exc}") from None

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise SystemExit(
            "Workspace submodule check failed. Initialize recorded revisions with "
            "`git submodule update --init --recursive`; review dirty entries before "
            f"changing them:\n{details}"
        )
    if metadata_only:
        print(f"verified canonical submodule metadata under {PROJECT_ROOT}")
    else:
        print(f"verified {checked} pinned submodule checkout(s) under {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
