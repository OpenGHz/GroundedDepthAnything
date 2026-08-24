"""Ensure the pinned public SAM2.1 Hiera-L checkpoint is present and intact."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

CHECKPOINT = (
    Path(os.environ.get("GDA_WORKSPACE_ROOT", Path(__file__).resolve().parents[2]))
    / "sdf_compute"
    / "thirdparty"
    / "grounded_sam_2"
    / "checkpoints"
    / "sam2.1_hiera_large.pt"
)
URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists():
        actual = sha256(CHECKPOINT)
        if actual == SHA256:
            print(f"SAM2 checkpoint verified: {CHECKPOINT}")
            return
        raise SystemExit(
            f"Refusing to overwrite an existing checkpoint with the wrong SHA256: "
            f"{CHECKPOINT} ({actual})"
        )

    partial = CHECKPOINT.with_suffix(CHECKPOINT.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    print(f"downloading SAM2 checkpoint from {URL}")
    try:
        urllib.request.urlretrieve(URL, partial)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    actual = sha256(partial)
    if actual != SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded checkpoint SHA256 mismatch: {actual}")
    partial.replace(CHECKPOINT)
    print(f"SAM2 checkpoint installed: {CHECKPOINT}")


if __name__ == "__main__":
    main()
