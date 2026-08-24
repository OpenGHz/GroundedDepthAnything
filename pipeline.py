"""Main pipeline: single image -> depth + grounded instance masks.

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

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import CliApp

from gda.datatypes import DepthAndSegResult
from gda.modules.depth_estimation import (
    DEFAULT_DA3_MODEL_ID,
    DEFAULT_DA3_MODEL_REVISION,
    DepthEstimationConfig,
    DepthEstimatorDA3,
)
from gda.modules.grounded_segmentation import (
    DEFAULT_SAM3_MODEL_REVISION,
    AutocastDtype,
    GroundedBackend,
    GroundedSegmentationConfig,
    GroundingDinoSam2Config,
    Sam3ConceptSegmentationConfig,
    build_grounded_segmentor,
)
from gda.modules.object_detection import (
    DEFAULT_GROUNDING_DINO_MODEL_ID,
    DEFAULT_GROUNDING_DINO_MODEL_REVISION,
    ObjectDetectionConfig,
    _draw_boxes,
)
from gda.modules.object_segmentation import ObjectSegmentationConfig, Sam2AutocastDtype


class PipelineConfig(BaseModel, frozen=True):
    """Configuration for the end-to-end pipeline."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    depth: DepthEstimationConfig = DepthEstimationConfig()
    """Depth-Anything-3 configuration."""

    grounded: GroundedSegmentationConfig = GroundedSegmentationConfig()
    """Text-grounded instance-segmentation configuration."""


class ImageDepthAndSegPipeline:
    """End-to-end pipeline for a single image.

    Given an image and prompts, it produces:
    - depth map
    - bbox detections
    - segmentation masks (native SAM3 by default)
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

        logger.info("init: grounded segmentor backend=%s", config.grounded.backend)
        t0 = time.perf_counter()
        self.grounded_segmentor = build_grounded_segmentor(config.grounded)
        logger.info("init: grounded segmentor done in %.2fs", time.perf_counter() - t0)

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
        logger.info("grounded segmentation: start (prompts=%d)", len(prompts))
        grounded = self.grounded_segmentor.segment(image_rgb, prompts)
        logger.info(
            "grounded segmentation: done in %.2fs (instances=%d)",
            time.perf_counter() - t0,
            len(grounded.det.boxes_xyxy),
        )
        return DepthAndSegResult(depth=depth, det=grounded.det, seg=grounded.seg)

    def run(self, image_path: str | Path, prompts: list[str]) -> DepthAndSegResult:
        """Backward-compatible wrapper: loads image from path then calls process()."""

        image = Image.open(image_path).convert("RGB")
        image_rgb = np.array(image, dtype=np.uint8)
        return self.process(image_rgb, prompts)


class GDAArgs(BaseModel, frozen=True):
    """CLI arguments for the main pipeline."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    image: Path
    """Input image path."""

    prompts: str
    """Comma-, semicolon-, or newline-separated text concepts."""

    output_dir: Path | None = None
    """Output directory; defaults to the input image directory."""

    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    """Shared inference device."""

    hf_offline: bool = False
    """Disable Hugging Face network access."""

    hf_local_files_only: bool = False
    """Load Hugging Face models from local files only."""

    depth_model_name: str = DEFAULT_DA3_MODEL_ID
    """Depth-Anything-3 model id."""

    depth_model_revision: str | None = DEFAULT_DA3_MODEL_REVISION
    """Exact Depth-Anything-3 Hugging Face revision."""

    depth_colormap: Literal["turbo", "inferno", "magma", "viridis", "jet"] = "turbo"
    """Depth visualization colormap."""

    det_model_id: str = DEFAULT_GROUNDING_DINO_MODEL_ID
    """GroundingDINO model id used by the SAM2 backend."""

    det_model_revision: str | None = DEFAULT_GROUNDING_DINO_MODEL_REVISION
    """Exact GroundingDINO Hugging Face revision used by the SAM2 backend."""

    box_th: float = Field(default=0.25, ge=0.0, le=1.0)
    """GroundingDINO box threshold."""

    text_th: float = Field(default=0.3, ge=0.0, le=1.0)
    """GroundingDINO text threshold."""

    seg_backend: GroundedBackend = "sam3"
    """Use native SAM3 or the GroundingDINO+SAM2.1 baseline."""

    sam2_checkpoint: Path | None = None
    """Optional SAM2.1 checkpoint override."""

    sam2_model_cfg: str | None = None
    """Optional SAM2.1 model configuration override."""

    sam2_autocast_dtype: Sam2AutocastDtype = "bfloat16"
    """SAM2 CUDA autocast dtype."""

    sam3_checkpoint: Path | None = None
    """Optional local SAM3 image checkpoint."""

    sam3_load_from_hf: bool = True
    """Download gated SAM3 weights when no checkpoint is supplied."""

    sam3_model_revision: str = DEFAULT_SAM3_MODEL_REVISION
    """Exact SAM3 Hugging Face revision used when downloading the gated checkpoint."""

    sam3_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """SAM3 instance confidence threshold."""

    sam3_resolution: Literal[1008] = 1008
    """SAM3 square inference resolution."""

    sam3_compile: bool = False
    """Compile supported SAM3 image components."""

    sam3_autocast_dtype: AutocastDtype = "bfloat16"
    """SAM3 CUDA autocast dtype."""

    sam3_deduplicate_mask_iou: float | None = Field(default=0.9, gt=0.0, le=1.0)
    """Cross-prompt SAM3 duplicate suppression threshold."""


def _parse_prompts(prompts: str) -> list[str]:
    """Parse prompts string into a list.

    Accepts separators: ',', ';' and newlines.

    Note: when passing prompts through a shell runner, unquoted semicolons may be
    interpreted as command separators; commas are safer.
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


def main(cli_args: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = CliApp.run(GDAArgs, cli_args=cli_args)

    local_files_only = args.hf_offline or args.hf_local_files_only
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    prompts_list = _parse_prompts(args.prompts)
    if not prompts_list:
        raise ValueError("--prompts must contain at least one item")
    if local_files_only and args.seg_backend == "sam3" and args.sam3_checkpoint is None:
        raise ValueError("Offline SAM3 inference requires --sam3-checkpoint")

    out_dir = _resolve_output_dir(args.image, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sam2_config_kwargs: dict = {
        "device": args.device,
        "autocast_dtype": args.sam2_autocast_dtype,
    }
    if args.sam2_checkpoint is not None:
        sam2_config_kwargs["sam2_checkpoint"] = args.sam2_checkpoint
    if args.sam2_model_cfg is not None:
        sam2_config_kwargs["sam2_model_cfg"] = args.sam2_model_cfg

    config = PipelineConfig(
        depth=DepthEstimationConfig(
            model_name=args.depth_model_name,
            model_revision=args.depth_model_revision,
            device=args.device,
            colormap=args.depth_colormap,
            hf_local_files_only=local_files_only,
        ),
        grounded=GroundedSegmentationConfig(
            backend=args.seg_backend,
            grounded_sam2=GroundingDinoSam2Config(
                detector=ObjectDetectionConfig(
                    model_id=args.det_model_id,
                    model_revision=args.det_model_revision,
                    device=args.device,
                    box_threshold=args.box_th,
                    text_threshold=args.text_th,
                    hf_local_files_only=local_files_only,
                ),
                segmentor=ObjectSegmentationConfig(**sam2_config_kwargs),
            ),
            sam3=Sam3ConceptSegmentationConfig(
                device=args.device,
                checkpoint=args.sam3_checkpoint,
                load_from_hf=args.sam3_load_from_hf,
                model_revision=args.sam3_model_revision,
                confidence_threshold=args.sam3_confidence_threshold,
                resolution=args.sam3_resolution,
                compile=args.sam3_compile,
                autocast_dtype=args.sam3_autocast_dtype,
                deduplicate_mask_iou=args.sam3_deduplicate_mask_iou,
            ),
        ),
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
        **(
            {"prompt_matches": np.asarray(seg.prompt_matches, dtype=object)}
            if seg.prompt_matches is not None
            else {}
        ),
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
        if (
            depth_vis_for_overlay.shape[0] != target_h
            or depth_vis_for_overlay.shape[1] != target_w
        ):
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
