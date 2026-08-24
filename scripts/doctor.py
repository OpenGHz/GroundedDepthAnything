"""Validate the installed runtime and compiled SAM2 extension."""

from __future__ import annotations

import os

import torch


def main() -> None:
    import depth_anything_3  # noqa: F401
    import sam2
    import sam2._C
    import sam3  # noqa: F401
    import xformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run doctor on an NVIDIA node")
    capability = torch.cuda.get_device_capability()
    expected = os.environ.get("GDA_EXPECTED_CUDA_ARCH")
    if expected and capability != tuple(map(int, expected.split("."))):
        raise RuntimeError(f"expected CUDA capability {expected}, found {capability}")
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name()}")
    print(f"capability={capability[0]}.{capability[1]}")
    print(f"xformers={xformers.__version__}")
    print(f"sam2_cuda_extension={sam2._C.__file__}")


if __name__ == "__main__":
    main()
