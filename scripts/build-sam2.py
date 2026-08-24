"""Build the vendored SAM2 CUDA extension for the detected GPU architecture."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = (
    Path(os.environ.get("GDA_WORKSPACE_ROOT", PROJECT_ROOT.parent)).expanduser().resolve()
)
SAM2_ROOT = WORKSPACE_ROOT / "sdf_compute" / "thirdparty" / "grounded_sam_2"
SAM2_PATCH = PROJECT_ROOT / "scripts" / "patches" / "grounded-sam2-cuda-arch.patch"


def ensure_cuda_arch_patch() -> None:
    """Apply the pinned SAM2 build patch once, without discarding local work."""

    reverse_check = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(SAM2_PATCH)],
        cwd=SAM2_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if reverse_check.returncode == 0:
        return

    forward_check = subprocess.run(
        ["git", "apply", "--check", str(SAM2_PATCH)],
        cwd=SAM2_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if forward_check.returncode != 0:
        raise RuntimeError(
            "Cannot apply the pinned SAM2 CUDA architecture patch without overwriting "
            f"local changes in {SAM2_ROOT}.\n{forward_check.stderr.strip()}"
        )
    subprocess.run(
        ["git", "apply", str(SAM2_PATCH)],
        cwd=SAM2_ROOT,
        check=True,
    )


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


def main() -> None:
    ensure_cuda_arch_patch()
    arch = detect_arch()
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = arch
    env["SAM2_BUILD_CUDA"] = "1"
    env["SAM2_BUILD_ALLOW_ERRORS"] = "0"
    print(f"building SAM2 CUDA extension for TORCH_CUDA_ARCH_LIST={arch}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(SAM2_ROOT),
        ],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
