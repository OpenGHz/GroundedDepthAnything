"""Validate the sibling repositories required by the locked Pixi manifest."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("GDA_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])).resolve()
PINNED = {
    "sam3": "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da",
    "Depth-Anything-3": "2c21ea849ceec7b469a3e62ea0c0e270afc3281a",
    "sdf_compute/thirdparty/grounded_sam_2": "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28",
}


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    errors: list[str] = []
    for relative, expected in PINNED.items():
        path = ROOT / relative
        if not (path / ".git").exists() and not (path / "HEAD").exists():
            errors.append(f"missing git repository: {path}")
            continue
        try:
            actual = git_head(path)
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"cannot inspect {path}: {exc}")
            continue
        if actual != expected:
            errors.append(f"{path}: expected {expected}, found {actual}")

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise SystemExit(
            "Workspace repositories do not match the Pixi lock pins. "
            "Fetch the pinned commits without overwriting local changes, then retry:\n" + details
        )
    print(f"workspace pins verified under {ROOT}")


if __name__ == "__main__":
    main()
