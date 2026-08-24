"""Hash setup inputs that must invalidate anchored-install's downstream cache."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECIAL_FILES = {
    ".gitmodules",
    "MANIFEST.in",
    "pixi.lock",
    "pixi.toml",
    "pyproject.toml",
    "requirements.txt",
}
REQUIRED_GITLINKS = {
    "third_party/depth-anything-3",
    "third_party/grounded-sam-2",
    "third_party/sam3",
}


def setup_fingerprint(project_root: Path = PROJECT_ROOT) -> str:
    """Return a deterministic digest of setup files and parent gitlinks."""

    result = subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "--stage", "-z"],
        check=True,
        capture_output=True,
    )
    selected: list[tuple[str, str, bytes]] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", maxsplit=1)
        mode, object_id, stage = metadata.decode().split()
        path = raw_path.decode()
        if stage != "0":
            raise RuntimeError(f"setup input has an unresolved index stage: {path}")
        if mode == "160000":
            if path in REQUIRED_GITLINKS:
                selected.append((path, mode, object_id.encode()))
            continue
        include = (
            path.endswith(".py")
            or path in SPECIAL_FILES
            or (path.startswith("scripts/patches/") and path.endswith(".patch"))
        )
        if include and not path.startswith("third_party/"):
            selected.append((path, mode, (project_root / path).read_bytes()))

    selected_gitlinks = {path for path, mode, _ in selected if mode == "160000"}
    missing = REQUIRED_GITLINKS - selected_gitlinks
    if missing:
        raise RuntimeError(f"missing required setup gitlinks: {sorted(missing)}")

    digest = hashlib.sha256()
    for path, mode, content in sorted(selected):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(mode.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


if __name__ == "__main__":
    print(setup_fingerprint())
