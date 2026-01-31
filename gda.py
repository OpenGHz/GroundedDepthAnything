"""Main pipeline module: single image -> depth + detections + masks.

This file contains:
- a pipeline class (single `config` argument, pydantic BaseModel)
- a CLI entry implemented with pydantic_settings

Outputs (default names under output_dir):
- depth.npy, depth.png
- detections.json, detections_vis.png
- masks.npz, masks_vis.png, masks_meta.json
- depth_with_masks.png (overlay masks on colorized depth)

Notes:
- Follows prompts/prepare.md:
    - class initialization takes exactly one `config` argument (pydantic BaseModel)
    - CLI arguments are parsed via pydantic_settings
    - output_dir defaults to the input image directory
"""

from __future__ import annotations

import os
import sys


def _parse_bool_arg(val: str) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _early_configure_hf_offline_from_argv(argv: list[str]) -> None:
    def _get_flag(name: str) -> bool:
        for i, tok in enumerate(argv):
            if tok == name:
                if i + 1 < len(argv):
                    return _parse_bool_arg(argv[i + 1])
                return True
            if tok.startswith(name + "="):
                return _parse_bool_arg(tok.split("=", 1)[1])
        return False

    if _get_flag("--hf_offline") or _get_flag("--hf_local_files_only"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_early_configure_hf_offline_from_argv(sys.argv)

import logging
import json
import re
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from gda.datatypes import DepthAndSegResult
from gda.modules.depth_estimation import DepthEstimationConfig, DepthEstimatorDA3
from gda.modules.object_detection import _draw_boxes
from gda.modules.object_detection import GroundingDinoDetector, ObjectDetectionConfig
from gda.modules.object_segmentation import (
    ObjectSegmentationConfig,
    Sam2BoxSegmentor,
    Sam3BoxSegmentor,
)


class PipelineConfig(BaseModel):
    """Configuration for the end-to-end pipeline."""

    model_config = ConfigDict(extra="forbid")

    depth: DepthEstimationConfig = DepthEstimationConfig()
    det: ObjectDetectionConfig = ObjectDetectionConfig()
    seg: ObjectSegmentationConfig = ObjectSegmentationConfig()


class ImageDepthAndSegPipeline:
    """End-to-end pipeline for a single image.

    Given an image and prompts, it produces:
    - depth map
    - bbox detections
    - segmentation masks (SAM2 default)
    """

    def __init__(self, config: PipelineConfig):
        """Create the pipeline.

        Args:
            config: Nested configs for depth/detection/segmentation.
        """

        logger = logging.getLogger("gda.pipeline")
        self.config = config

        logger.info("init: depth estimator")
        t0 = time.perf_counter()
        self.depth_estimator = DepthEstimatorDA3(config.depth)
        logger.info("init: depth estimator done in %.2fs", time.perf_counter() - t0)

        logger.info("init: detector")
        t0 = time.perf_counter()
        self.detector = GroundingDinoDetector(config.det)
        logger.info("init: detector done in %.2fs", time.perf_counter() - t0)

        logger.info("init: segmentor backend=%s", config.seg.backend)
        t0 = time.perf_counter()
        if config.seg.backend == "sam3":
            self.segmentor = Sam3BoxSegmentor(config.seg)
        else:
            self.segmentor = Sam2BoxSegmentor(config.seg)
        logger.info("init: segmentor done in %.2fs", time.perf_counter() - t0)

    def process(self, image_rgb: np.ndarray, prompts: list[str]) -> DepthAndSegResult:
        """Process one image from an in-memory RGB array.

        Args:
            image_rgb: Input image as RGB uint8 array, shape [H, W, 3].
            prompts: List of prompt strings.

        Returns:
            (depth, det, seg_meta, masks)
            - depth: float32 array [Hd, Wd]
            - det: detection dict (json-serializable)
            - seg_meta: mask metadata dict (json-serializable)
            - masks: bool array [N, Hm, Wm]
        """

        logger = logging.getLogger("gda.pipeline")

        t0 = time.perf_counter()
        logger.info("depth: start")
        depth = self.depth_estimator.predict(image_rgb)
        logger.info(
            "depth: done in %.2fs (shape=%s)",
            time.perf_counter() - t0,
            tuple(depth.shape),
        )

        t0 = time.perf_counter()
        logger.info("detect: start (prompts=%d)", len(prompts))
        det = self.detector.detect(image_rgb, prompts)
        logger.info(
            "detect: done in %.2fs (boxes=%d)",
            time.perf_counter() - t0,
            len(det.boxes_xyxy),
        )

        t0 = time.perf_counter()
        logger.info("segment: start")
        seg = self.segmentor.segment(image_rgb, det)
        logger.info(
            "segment: done in %.2fs (masks=%d)",
            time.perf_counter() - t0,
            int(seg.masks.shape[0]),
        )
        return DepthAndSegResult(depth=depth, det=det, seg=seg)

    def run(self, image_path: str | Path, prompts: list[str]) -> DepthAndSegResult:
        """Backward-compatible wrapper: loads image from path then calls process()."""

        image = Image.open(image_path).convert("RGB")
        image_rgb = np.array(image, dtype=np.uint8)
        return self.process(image_rgb, prompts)


class GDACLI(BaseSettings):
    """CLI arguments for the main pipeline."""

    model_config = SettingsConfigDict(cli_parse_args=True, extra="ignore")

    image: Path
    prompts: str
    output_dir: Path | None = None

    device: str = "cuda"  # best-effort shared default

    # HuggingFace / download controls
    hf_offline: bool = False
    hf_local_files_only: bool = False

    # depth
    depth_model_name: str = "depth-anything/DA3-LARGE"
    depth_colormap: Literal["turbo", "inferno", "magma", "viridis", "jet"] = "turbo"

    # detection
    det_model_id: str = "IDEA-Research/grounding-dino-base"
    box_th: float = 0.25
    text_th: float = 0.3

    # segmentation
    seg_backend: Literal["sam2", "sam3"] = "sam2"
    sam2_checkpoint: Path | None = None
    sam2_model_cfg: str | None = None

    sam3_checkpoint: Path | None = None
    sam3_load_from_hf: bool = False


def _parse_prompts(prompts: str) -> list[str]:
    """Parse prompts string into a list.

    Accepts separators: ',', ';' and newlines.

    Note: when running via `conda run`, unquoted semicolons may be interpreted
    by the shell; commas are safer.
    """

    items = [p.strip() for p in re.split(r"[;,\n]+", prompts)]
    return [p for p in items if p]


def _resolve_output_dir(image_path: Path, output_dir: Path | None) -> Path:
    """Resolve output directory.

    If output_dir is not provided, it defaults to the input image directory.
    """

    return output_dir if output_dir is not None else image_path.parent


def _overlay_masks(image_bgr: np.ndarray, masks: np.ndarray, prompt_ids: list[int]) -> np.ndarray:
    """Overlay masks on a BGR image."""

    out = image_bgr.copy()
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
    ]

    for i in range(masks.shape[0]):
        mask = masks[i]
        pid = prompt_ids[i] if i < len(prompt_ids) else -1
        color = palette[pid % len(palette)] if pid >= 0 else (255, 255, 255)
        bgr = (int(color[2]), int(color[1]), int(color[0]))

        overlay = np.zeros_like(out)
        overlay[mask] = bgr
        out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0)

    return out


def main() -> None:
    """CLI entrypoint."""

    args = GDACLI()

    if args.hf_offline or args.hf_local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    prompts_list = _parse_prompts(args.prompts)
    if not prompts_list:
        raise ValueError("--prompts must contain at least one item")

    out_dir = _resolve_output_dir(args.image, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_cfg_kwargs: dict = {
        "backend": args.seg_backend,
        "device": args.device,
        "sam3_checkpoint": args.sam3_checkpoint,
        "sam3_load_from_hf": args.sam3_load_from_hf,
    }
    if args.sam2_checkpoint is not None:
        seg_cfg_kwargs["sam2_checkpoint"] = args.sam2_checkpoint
    if args.sam2_model_cfg is not None:
        seg_cfg_kwargs["sam2_model_cfg"] = args.sam2_model_cfg

    config = PipelineConfig(
        depth=DepthEstimationConfig(
            model_name=args.depth_model_name,
            device=args.device,
            colormap=args.depth_colormap,
            hf_local_files_only=args.hf_local_files_only,
        ),
        det=ObjectDetectionConfig(
            model_id=args.det_model_id,
            device=args.device,
            box_threshold=args.box_th,
            text_threshold=args.text_th,
            hf_local_files_only=args.hf_local_files_only,
        ),
        seg=ObjectSegmentationConfig(**seg_cfg_kwargs),
    )

    pipeline = ImageDepthAndSegPipeline(config)

    image = Image.open(args.image).convert("RGB")
    image_rgb = np.array(image, dtype=np.uint8)
    result = pipeline.process(image_rgb, prompts_list)
    depth = result.depth
    det = result.det
    seg = result.seg

    # Save depth
    np.save(out_dir / "depth.npy", depth.astype(np.float32))
    depth_vis = pipeline.depth_estimator.colorize(depth)
    cv2.imwrite(str(out_dir / "depth.png"), depth_vis)

    # Save detections
    (out_dir / "detections.json").write_text(
        json.dumps(det.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    det_vis = _draw_boxes(image, det)
    det_vis.save(out_dir / "detections_vis.png")

    # Save masks
    np.savez_compressed(
        out_dir / "masks.npz",
        masks=seg.masks.astype(np.bool_),
        boxes_xyxy=np.asarray(seg.boxes_xyxy, dtype=np.float32),
        prompt_ids=np.asarray(seg.prompt_ids, dtype=np.int32),
        scores=np.asarray(seg.scores, dtype=np.float32),
        prompts=np.asarray(seg.prompts, dtype=object),
        image_size=np.asarray([seg.image_size[0], seg.image_size[1]], dtype=np.int32),
    )
    (out_dir / "masks_meta.json").write_text(
        json.dumps(seg.meta_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save visualizations
    rgb = np.array(image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    masks_vis = _overlay_masks(bgr, seg.masks, seg.prompt_ids.tolist())
    cv2.imwrite(str(out_dir / "masks_vis.png"), masks_vis)

    # Depth prediction may be at a different resolution than the input image.
    # For visualization, resize the colorized depth to the mask/image size.
    depth_vis_for_overlay = depth_vis
    if seg.masks.ndim == 3:
        target_h, target_w = int(seg.masks.shape[1]), int(seg.masks.shape[2])
        if depth_vis_for_overlay.shape[0] != target_h or depth_vis_for_overlay.shape[1] != target_w:
            depth_vis_for_overlay = cv2.resize(
                depth_vis_for_overlay,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )

    depth_with_masks = _overlay_masks(depth_vis_for_overlay, seg.masks, seg.prompt_ids.tolist())
    cv2.imwrite(str(out_dir / "depth_with_masks.png"), depth_with_masks)

    print(f"[OK] outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
