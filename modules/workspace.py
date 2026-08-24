"""Resolve repository-owned third-party sources and model-cache paths."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""Root of the GDA source checkout."""


def third_party_root() -> Path:
    """Return the GDA-owned submodule root, with an explicit developer override."""

    configured = os.environ.get("GDA_THIRD_PARTY_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "third_party"


def cache_root() -> Path:
    """Return the writable cache used for downloaded GDA model artifacts."""

    configured = os.environ.get("GDA_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return (base / "gda").resolve()
