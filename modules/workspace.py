"""Locate the sibling model repositories used by the Pixi workspace."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Return the configured EIA root, or infer it from the current layout."""

    configured = os.environ.get("GDA_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    module_path = Path(__file__).resolve()
    current = Path.cwd().resolve()
    candidates = [
        module_path.parents[2],
        current,
        *current.parents,
        *module_path.parents,
    ]
    for candidate in candidates:
        if (candidate / "sam3").exists() or (candidate / "Depth-Anything-3").exists():
            return candidate
    return module_path.parents[2]
