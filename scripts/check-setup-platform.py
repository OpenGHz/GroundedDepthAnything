"""Prevent incompatible GPU targets from sharing one mutable Pixi prefix."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKER = PROJECT_ROOT / ".pixi" / "gda-platform"
VALID_PLATFORMS = {"h200", "b300"}


def ensure_platform(platform: str, marker: Path = MARKER) -> None:
    """Bind a checkout's Pixi prefix to exactly one GPU target."""

    if platform not in VALID_PLATFORMS:
        raise ValueError(f"platform must be one of {sorted(VALID_PLATFORMS)}, found {platform!r}")
    if marker.exists():
        existing = marker.read_text(encoding="utf-8").strip()
        if existing != platform:
            raise RuntimeError(
                f"This checkout's .pixi prefix is bound to {existing!r}, not {platform!r}. "
                "Use a separate checkout for each GPU target; this avoids replacing one "
                "target's PyTorch and SAM2 extension with the other target's binaries."
            )
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(platform + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check-setup-platform.py {h200|b300}")
    ensure_platform(sys.argv[1])
    print(f"Pixi prefix platform: {sys.argv[1]}")
