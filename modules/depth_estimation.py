"""Depth estimation module (single image) based on Depth-Anything-3.

This file contains:
- a configurable module class (single `config` argument, pydantic BaseModel)
- a CLI entry implemented with pydantic_settings

Outputs:
- depth.npy: float32 depth map, shape [H, W]
- depth.png: colorized visualization (default)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DA3_SRC = _REPO_ROOT / "Depth-Anything-3" / "src"
if not _DA3_SRC.exists():
    raise FileNotFoundError(
        f"Depth-Anything-3 source path not found: {_DA3_SRC}. "
        "Please check the workspace layout."
    )
if str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))

from depth_anything_3.api import DepthAnything3  # noqa: E402


class DepthEstimationConfig(BaseModel):
    """Configuration for Depth-Anything-3 depth estimation."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "depth-anything/DA3-LARGE"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    colormap: Literal["turbo", "inferno", "magma", "viridis", "jet"] = "turbo"
    hf_local_files_only: bool = False


class DepthEstimatorDA3:
    """Estimate depth for a single image using Depth-Anything-3."""

    def __init__(self, config: DepthEstimationConfig):
        """Create the estimator.

        Args:
            config: Model/device/visualization configuration.
        """

        self.config = config
        self.device = torch.device(config.device)

        logger = logging.getLogger("gda.depth")
        t0 = time.perf_counter()
        logger.info("loading DA3 model=%s device=%s", config.model_name, config.device)
        self.model = DepthAnything3.from_pretrained(
            config.model_name,
            local_files_only=bool(config.hf_local_files_only),
        ).to(self.device)
        logger.info("DA3 loaded in %.2fs", time.perf_counter() - t0)

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray | Image.Image) -> np.ndarray:
        """Predict a depth map for a single image.

        Args:
            image_rgb: Input image as RGB (numpy uint8 array [H,W,3] or PIL Image).

        Returns:
            Depth map as float32 numpy array with shape [H, W].
        """

        if isinstance(image_rgb, Image.Image):
            image_in: np.ndarray | Image.Image = image_rgb.convert("RGB")
        else:
            arr = np.asarray(image_rgb)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"image_rgb must have shape [H,W,3], got {arr.shape}")
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            image_in = arr

        prediction = self.model.inference(
            [image_in],
            export_format="mini_npz",
            export_dir=None,
        )
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        return depth

    @torch.no_grad()
    def predict_from_path(self, image_path: str | Path) -> np.ndarray:
        """Backward-compatible helper: load image from path then call predict()."""

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.predict(image)

    def colorize(self, depth: np.ndarray) -> np.ndarray:
        """Convert a depth map into a colorized visualization.

        Args:
            depth: Depth map, shape [H, W], float-like.

        Returns:
            Color image in BGR uint8 format, shape [H, W, 3].
        """

        depth = np.asarray(depth)
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            gray = np.zeros(depth.shape, dtype=np.uint8)
        else:
            d = depth[valid]
            lo = float(np.percentile(d, 5))
            hi = float(np.percentile(d, 95))
            if hi <= lo:
                hi = lo + 1e-6
            norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
            gray = (norm * 255.0).astype(np.uint8)

        cmap = {
            "turbo": cv2.COLORMAP_TURBO,
            "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "jet": cv2.COLORMAP_JET,
        }[self.config.colormap]
        return cv2.applyColorMap(gray, cmap)


class DepthEstimationCLI(BaseSettings):
    """CLI arguments for depth estimation."""

    model_config = SettingsConfigDict(cli_parse_args=True, extra="ignore")

    image: Path
    output_dir: Path | None = None

    model_name: str = "depth-anything/DA3-LARGE"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    colormap: Literal["turbo", "inferno", "magma", "viridis", "jet"] = "turbo"

    save_npy: bool = True
    save_png: bool = True


def _resolve_output_dir(image_path: Path, output_dir: Path | None) -> Path:
    """Resolve output directory.

    If output_dir is not provided, it defaults to the input image directory.
    """

    return output_dir if output_dir is not None else image_path.parent


def main() -> None:
    """CLI entrypoint."""

    args = DepthEstimationCLI()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    out_dir = _resolve_output_dir(args.image, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    estimator = DepthEstimatorDA3(
        DepthEstimationConfig(
            model_name=args.model_name,
            device=args.device,
            colormap=args.colormap,
        )
    )

    image = Image.open(args.image).convert("RGB")
    depth = estimator.predict(image)

    if args.save_npy:
        np.save(out_dir / "depth.npy", depth)

    if args.save_png:
        depth_vis = estimator.colorize(depth)
        cv2.imwrite(str(out_dir / "depth.png"), depth_vis)

    print(f"[OK] depth saved under: {out_dir}")


if __name__ == "__main__":
    main()
