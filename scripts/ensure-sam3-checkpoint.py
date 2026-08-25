"""Ensure the pinned SAM3 image checkpoint is cached through ModelScope."""

from __future__ import annotations

from gda.modules.sam3_checkpoint import download_sam3_checkpoint


def main() -> None:
    checkpoint = download_sam3_checkpoint()
    print(f"SAM3 checkpoint verified: {checkpoint}")


if __name__ == "__main__":
    main()
