"""Integrated image -> depth + grounded masks -> per-mask positions.

This file contains:
- a pipeline class (single `config` argument, pydantic BaseModel)
- a CLI entry implemented with pydantic_settings

Outputs (default names under output_dir):
- (optional intermediates, controlled by --save_intermediate)
  - depth.npy, depth.png
  - detections.json, detections_vis.png
  - masks.npz, masks_vis.png, masks_meta.json
  - depth_with_masks.png
- positions.npz, positions.json

Notes:
- Follows prompts/prepare.md:
  - class initialization takes exactly one `config` argument (pydantic BaseModel)
  - CLI arguments are parsed via pydantic_settings
  - output_dir defaults to the input image directory
- Depth prediction may differ in resolution; position representation resizes depth to mask size.
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

from gda.datatypes import ImageToPositionsResult
from gda.modules.depth_estimation import (
    DEFAULT_DA3_MODEL_ID,
    DEFAULT_DA3_MODEL_REVISION,
    DepthEstimationConfig,
)
from gda.modules.grounded_segmentation import (
    DEFAULT_SAM3_MODEL_REVISION,
    AutocastDtype,
    GroundedBackend,
    GroundedSegmentationConfig,
    GroundingDinoSam2Config,
    Sam3ConceptSegmentationConfig,
)
from gda.modules.object_detection import (
    DEFAULT_GROUNDING_DINO_MODEL_ID,
    DEFAULT_GROUNDING_DINO_MODEL_REVISION,
    ObjectDetectionConfig,
    _draw_boxes,
)
from gda.modules.object_segmentation import ObjectSegmentationConfig, Sam2AutocastDtype
from gda.modules.pointcloud_generation import (
    PointCloudGenerationConfig,
    PointCloudGenerator,
    load_k,
    save_pointcloud_ply,
)
from gda.modules.position_representation import (
    MaskPositionRepresentor,
    PositionRepresentationConfig,
)
from gda.pipeline import ImageDepthAndSegPipeline, PipelineConfig


class ImageToPositionsConfig(BaseModel, frozen=True):
    """Configuration for the integrated pipeline (image -> positions)."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    pipeline: PipelineConfig = PipelineConfig()
    """Depth and grounded-segmentation pipeline configuration."""

    pos: PositionRepresentationConfig = PositionRepresentationConfig()
    """Mask-to-representative-position configuration."""

    save_intermediate: bool = True
    """Save depth, detections, masks, and visualizations."""


class ImageToPositionsPipeline:
    """End-to-end pipeline for a single image to per-mask positions."""

    def __init__(self, config: ImageToPositionsConfig):
        self.config = config
        self.pipeline = ImageDepthAndSegPipeline(config.pipeline)
        self.representor = MaskPositionRepresentor(config.pos)

    def process(self, image_rgb: np.ndarray, prompts: list[str]) -> ImageToPositionsResult:
        result = self.pipeline.process(image_rgb, prompts)
        positions = self.representor.compute(
            depth=result.depth,
            masks=result.seg.masks,
            prompts=result.det.prompts,
            prompt_ids=result.seg.prompt_ids.tolist(),
        )

        return ImageToPositionsResult(
            depth=result.depth,
            det=result.det,
            seg=result.seg,
            positions=positions,
        )

    def run(self, image_path: str | Path, prompts: list[str]) -> ImageToPositionsResult:
        """Backward-compatible wrapper: loads image from path then calls process()."""

        image = Image.open(image_path).convert("RGB")
        image_rgb = np.array(image, dtype=np.uint8)
        return self.process(image_rgb, prompts)


class ImageToPositionsArgs(BaseModel, frozen=True):
    """CLI arguments for the integrated pipeline."""

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
    """Depth-Anything-3 model repository or local directory."""

    depth_model_revision: str | None = DEFAULT_DA3_MODEL_REVISION
    """Exact Depth-Anything-3 Hugging Face revision."""

    depth_colormap: Literal["turbo", "inferno", "magma", "viridis", "jet"] = "turbo"
    """Depth visualization colormap."""

    det_model_id: str = DEFAULT_GROUNDING_DINO_MODEL_ID
    """GroundingDINO model repository or local directory."""

    det_model_revision: str | None = DEFAULT_GROUNDING_DINO_MODEL_REVISION
    """Exact GroundingDINO Hugging Face revision."""

    box_th: float = 0.25
    """GroundingDINO box threshold."""

    text_th: float = 0.3
    """GroundingDINO text threshold."""

    seg_backend: GroundedBackend = "sam3"
    """Grounded segmentation backend."""

    sam2_checkpoint: Path | None = None
    """Optional SAM2.1 checkpoint override."""

    sam2_model_cfg: str | None = None
    """Optional SAM2.1 model configuration override."""

    sam2_autocast_dtype: Sam2AutocastDtype = "bfloat16"
    """SAM2 CUDA autocast dtype."""

    sam3_checkpoint: Path | None = None
    """Optional local SAM3 checkpoint."""

    sam3_load_from_hf: bool = True
    """Download gated SAM3 weights when no checkpoint is supplied."""

    sam3_model_revision: str = DEFAULT_SAM3_MODEL_REVISION
    """Exact SAM3 Hugging Face revision."""

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

    min_depth: float = 1e-6
    """Minimum valid depth used for representative positions."""

    max_depth: float | None = None
    """Optional maximum valid depth used for representative positions."""

    save_intermediate: bool = True
    """Save depth, box, and mask intermediate artifacts."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    """Application log level."""

    log_file: Path | None = None
    """Optional additional log file."""

    make_pointcloud: bool = False
    """Generate a point cloud from depth and camera intrinsics."""

    visualize_pointcloud: bool = False
    """Open an interactive point-cloud viewer."""

    visualize_seconds: float | None = None
    """Optional point-cloud viewer duration in seconds."""

    k_file: Path | None = None
    """Optional camera-intrinsics matrix file."""

    fx: float | None = None
    """Camera focal length along x when no matrix file is provided."""

    fy: float | None = None
    """Camera focal length along y when no matrix file is provided."""

    cx: float | None = None
    """Camera principal point x when no matrix file is provided."""

    cy: float | None = None
    """Camera principal point y when no matrix file is provided."""

    pc_min_depth: float = 1e-6
    """Minimum depth retained in the point cloud."""

    pc_max_depth: float | None = None
    """Optional maximum depth retained in the point cloud."""

    pc_use_masks_union_only: bool = True
    """Restrict the point cloud to the union of detected masks."""

    pc_include_colors: bool = True
    """Include RGB values in point-cloud output."""

    pc_save_ply: bool = True
    """Save generated point clouds as PLY files."""


def _parse_prompts(prompts: str) -> list[str]:
    items = [p.strip() for p in re.split(r"[;,\n]+", prompts)]
    return [p for p in items if p]


def _resolve_output_dir(image_path: Path, output_dir: Path | None) -> Path:
    return output_dir if output_dir is not None else image_path.parent


def _overlay_masks(image_bgr: np.ndarray, masks: np.ndarray, prompt_ids: list[int]) -> np.ndarray:
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


def _save_positions(
    out_dir: Path, pos_meta: dict, rep_uvs: np.ndarray, rep_depths: np.ndarray, valids: np.ndarray
) -> None:
    np.savez_compressed(
        out_dir / "positions.npz",
        rep_uvs=rep_uvs,
        rep_depths=rep_depths,
        valids=valids,
        meta=json.dumps(pos_meta, ensure_ascii=False),
    )

    items: list[dict] = []
    for i in range(rep_uvs.shape[0]):
        items.append(
            {
                "mask_index": int(i),
                "valid": bool(valids[i]),
                "rep_uv": [int(rep_uvs[i, 0]), int(rep_uvs[i, 1])],
                "rep_depth": float(rep_depths[i]) if np.isfinite(rep_depths[i]) else None,
            }
        )

    (out_dir / "positions.json").write_text(
        json.dumps({"meta": pos_meta, "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _make_k_from_args(args: ImageToPositionsArgs) -> np.ndarray:
    if args.k_file is not None:
        return load_k(args.k_file)
    if None in (args.fx, args.fy, args.cx, args.cy):
        raise ValueError("Provide either --k_file or all of --fx --fy --cx --cy")
    fx, fy, cx, cy = float(args.fx), float(args.fy), float(args.cx), float(args.cy)
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def main(cli_args: list[str] | None = None) -> None:
    args = CliApp.run(ImageToPositionsArgs, cli_args=cli_args)

    local_files_only = args.hf_offline or args.hf_local_files_only
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("gda.image_to_positions")

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image = Image.open(args.image).convert("RGB")
    image_rgb = np.array(image, dtype=np.uint8)

    prompts_list = _parse_prompts(args.prompts)
    if not prompts_list:
        raise ValueError("--prompts must contain at least one item")
    if local_files_only and args.seg_backend == "sam3" and args.sam3_checkpoint is None:
        raise ValueError("Offline SAM3 inference requires --sam3-checkpoint")

    out_dir = _resolve_output_dir(args.image, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.log_file is not None:
        fh = logging.FileHandler(args.log_file)
        fh.setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    logger.info("start")
    logger.info("image=%s", args.image)
    logger.info("output_dir=%s", out_dir)
    logger.info("device=%s", args.device)
    logger.info("hf_offline=%s hf_local_files_only=%s", args.hf_offline, args.hf_local_files_only)
    logger.info("prompts(%d)=%s", len(prompts_list), prompts_list)
    logger.info("seg_backend=%s", args.seg_backend)
    logger.info("save_intermediate=%s", args.save_intermediate)
    logger.info(
        "make_pointcloud=%s visualize_pointcloud=%s",
        args.make_pointcloud,
        args.visualize_pointcloud,
    )

    sam2_config_kwargs: dict = {
        "device": args.device,
        "autocast_dtype": args.sam2_autocast_dtype,
    }
    if args.sam2_checkpoint is not None:
        sam2_config_kwargs["sam2_checkpoint"] = args.sam2_checkpoint
    if args.sam2_model_cfg is not None:
        sam2_config_kwargs["sam2_model_cfg"] = args.sam2_model_cfg

    config = ImageToPositionsConfig(
        pipeline=PipelineConfig(
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
        ),
        pos=PositionRepresentationConfig(min_depth=args.min_depth, max_depth=args.max_depth),
        save_intermediate=args.save_intermediate,
    )

    runner = ImageToPositionsPipeline(config)
    t0 = time.perf_counter()
    result = runner.process(image_rgb, prompts_list)
    t1 = time.perf_counter()
    logger.info("pipeline+positions done in %.3fs", t1 - t0)

    depth: np.ndarray = result.depth
    det = result.det
    seg = result.seg

    pos_meta: dict = result.positions.meta
    rep_uvs: np.ndarray = result.positions.rep_uvs
    rep_depths: np.ndarray = result.positions.rep_depths
    valids: np.ndarray = result.positions.valids

    # Always save positions
    _save_positions(out_dir, pos_meta, rep_uvs, rep_depths, valids)
    logger.info("positions saved: %s", out_dir / "positions.npz")

    if config.save_intermediate:
        # Save depth
        np.save(out_dir / "depth.npy", depth.astype(np.float32))
        depth_vis = runner.pipeline.depth_estimator.colorize(depth)
        cv2.imwrite(str(out_dir / "depth.png"), depth_vis)
        logger.info("depth saved")

        # Save detections
        (out_dir / "detections.json").write_text(
            json.dumps(det.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("detections saved")

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
        logger.info("masks saved")

        # Visualizations
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        prompt_ids_list = seg.prompt_ids.tolist()
        masks_vis = _overlay_masks(bgr, seg.masks, [int(x) for x in prompt_ids_list])
        cv2.imwrite(str(out_dir / "masks_vis.png"), masks_vis)

        # Overlay on depth visualization (resize depth_vis to mask size if needed)
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
        depth_with_masks = _overlay_masks(
            depth_vis_for_overlay, seg.masks, [int(x) for x in prompt_ids_list]
        )
        cv2.imwrite(str(out_dir / "depth_with_masks.png"), depth_with_masks)
        logger.info("visualizations saved")

    make_pointcloud = args.make_pointcloud or args.visualize_pointcloud

    if make_pointcloud:
        logger.info("pointcloud generation start")
        k = _make_k_from_args(args)
        pc_cfg = PointCloudGenerationConfig(
            min_depth=args.pc_min_depth,
            max_depth=args.pc_max_depth,
            use_masks_union_only=args.pc_use_masks_union_only,
            include_colors=args.pc_include_colors,
            save_ply=args.pc_save_ply,
        )
        generator = PointCloudGenerator(pc_cfg)

        # Align depth to mask size for consistent pixel coords / K
        depth_for_pc = depth.astype(np.float32)
        image_for_pc = image_rgb
        masks_for_pc = seg.masks
        if seg.masks.ndim == 3:
            target_h, target_w = int(seg.masks.shape[1]), int(seg.masks.shape[2])
            if depth_for_pc.shape[0] != target_h or depth_for_pc.shape[1] != target_w:
                depth_for_pc = cv2.resize(
                    depth_for_pc, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
            if image_for_pc.shape[0] != target_h or image_for_pc.shape[1] != target_w:
                image_for_pc = cv2.resize(
                    image_for_pc, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )

        pc = generator.generate(
            depth=depth_for_pc,
            k=k,
            image_rgb=image_for_pc,
            masks=masks_for_pc,
            rep_uvs=rep_uvs,
            rep_valids=valids,
        )

        np.savez_compressed(out_dir / "pointcloud.npz", **pc.to_npz_dict())
        logger.info(
            "pointcloud npz saved: %s (points=%d)",
            out_dir / "pointcloud.npz",
            int(pc.points_xyz.shape[0]),
        )

        if pc_cfg.save_ply:
            save_pointcloud_ply(out_dir / "pointcloud.ply", pc)
            logger.info("pointcloud ply saved: %s", out_dir / "pointcloud.ply")

        if args.visualize_pointcloud:
            from gda.modules.pointcloud_visualization import (
                PointCloudVisualizationConfig,
                PointCloudVisualizer,
            )

            if args.visualize_seconds is None:
                logger.warning(
                    "opening Open3D GUI (this is blocking until the window is closed). "
                    "Set --visualize_seconds N to auto-close."
                )
            else:
                logger.info(
                    "opening Open3D GUI with auto-close: %.2fs", float(args.visualize_seconds)
                )

            vis = PointCloudVisualizer(
                PointCloudVisualizationConfig(timeout_sec=args.visualize_seconds)
            )
            vis.visualize(
                points_xyz=pc.points_xyz,
                colors=pc.colors,
                rep_point_indices=pc.rep_point_indices,
            )

    logger.info("done")

    print(f"[OK] outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
