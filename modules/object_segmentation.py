"""Box-prompted object segmentation for a single image.

This module consumes:
- an image
- detection results (boxes) produced by object_detection.py

It outputs:
- masks.npz (boolean masks and metadata)
- masks_vis.png (overlay visualization)

Backend:
- SAM 2.1 from the local Grounded-SAM-2 checkout.

Notes:
- This module follows prompts/prepare.md:
    - class initialization takes exactly one `config` argument (pydantic BaseModel)
    - CLI arguments are parsed via pydantic_settings
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict
from pydantic_settings import CliApp

from gda.datatypes import DetectionResult, SegmentationResult
from gda.modules.workspace import workspace_root

_REPO_ROOT = workspace_root()

_DEFAULT_SAM2_CHECKPOINT = (
    _REPO_ROOT
    / "sdf_compute"
    / "thirdparty"
    / "grounded_sam_2"
    / "checkpoints"
    / "sam2.1_hiera_large.pt"
)
_DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

Sam2AutocastDtype = Literal["none", "bfloat16", "float16"]


def _import_sam2_components():
    """Import SAM2 lazily so SAM3-only commands do not require it."""

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        sam2_root = _REPO_ROOT / "sdf_compute" / "thirdparty" / "grounded_sam_2"
        if not sam2_root.exists():
            raise FileNotFoundError(
                f"SAM2 repo path not found: {sam2_root}. Please check the workspace layout."
            )

        import sys

        sys.path.insert(0, str(sam2_root))
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise ImportError(
                "Failed to import SAM2. Run `pixi install` from the gda repository."
            ) from exc

    return build_sam2, SAM2ImagePredictor


class ObjectSegmentationConfig(BaseModel, frozen=True):
    """Configuration for mask segmentation from boxes."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Torch device used by SAM2."""

    sam2_checkpoint: Path = _DEFAULT_SAM2_CHECKPOINT
    """Path to the SAM2.1 checkpoint."""

    sam2_model_cfg: str = _DEFAULT_SAM2_MODEL_CFG
    """Hydra model configuration bundled with SAM2."""

    sam2_multimask_output: bool = False
    """Generate three candidates per box and retain the best predicted-IoU mask."""

    autocast_dtype: Sam2AutocastDtype = "bfloat16"
    """CUDA automatic mixed-precision dtype; CPU inference ignores this setting."""


class Sam2BoxSegmentor:
    """Segment objects from bounding boxes using SAM2 image predictor."""

    def __init__(self, config: ObjectSegmentationConfig):
        """Create segmentor.

        Args:
            config: SAM2 checkpoint/config/device options.
        """

        self.config = config
        if not config.sam2_checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {config.sam2_checkpoint}")

        logger = logging.getLogger("gda.seg")
        logger.info(
            "building SAM2 model_cfg=%s ckpt=%s device=%s",
            config.sam2_model_cfg,
            config.sam2_checkpoint,
            config.device,
        )
        t0 = time.perf_counter()
        build_sam2, sam2_image_predictor = _import_sam2_components()
        self.model = build_sam2(
            config.sam2_model_cfg,
            ckpt_path=str(config.sam2_checkpoint),
            device=config.device,
        )
        self.predictor = sam2_image_predictor(self.model)
        logger.info("SAM2 ready in %.2fs", time.perf_counter() - t0)

    def _autocast_context(self):
        if not str(self.config.device).startswith("cuda") or self.config.autocast_dtype == "none":
            return nullcontext()
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.config.autocast_dtype]
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.no_grad()
    def segment(
        self, image_rgb: np.ndarray | Image.Image, det: DetectionResult | dict[str, Any]
    ) -> SegmentationResult:
        """Generate masks from detection boxes.

        Args:
            image_rgb: Input image as RGB (numpy uint8 array [H,W,3] or PIL Image).
            det: Detection result dict (from object_detection.py). Must contain:
                - image_size: [H, W]
                - prompts: list[str]
                - boxes_xyxy: list[list[float]]
                - prompt_ids: list[int]

        Returns:
            (meta, masks)
            - meta: JSON-serializable dict with basic metadata and mask scores
            - masks: boolean numpy array with shape [N, H, W]
        """

        if isinstance(image_rgb, Image.Image):
            image_np = np.array(image_rgb.convert("RGB"), dtype=np.uint8)
        else:
            image_np = np.asarray(image_rgb)
            if image_np.ndim != 3 or image_np.shape[2] != 3:
                raise ValueError(f"image_rgb must have shape [H,W,3], got {image_np.shape}")
            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8)
        h, w = image_np.shape[:2]

        if isinstance(det, dict):
            det_obj = DetectionResult.from_json_dict(det)
        else:
            det_obj = det

        boxes = np.asarray(det_obj.boxes_xyxy, dtype=np.float32)
        if tuple(map(int, det_obj.image_size)) != (int(h), int(w)):
            raise ValueError(
                "Detection image_size does not match the input image: "
                f"detection={det_obj.image_size}, image={(h, w)}"
            )
        num_detections = len(boxes) if boxes.ndim > 0 else 0
        if not (
            len(np.asarray(det_obj.scores).reshape(-1))
            == len(np.asarray(det_obj.prompt_ids).reshape(-1))
            == len(det_obj.labels)
            == num_detections
        ):
            raise ValueError(
                "Detection boxes, scores, prompt_ids, and labels must have equal length"
            )
        if boxes.size == 0:
            masks_arr = np.zeros((0, h, w), dtype=bool)
            scores_np = np.zeros((0,), dtype=np.float32)
        else:
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("det.boxes_xyxy must be N x 4")
            if not np.isfinite(boxes).all():
                raise ValueError("det.boxes_xyxy must contain only finite coordinates")

            with self._autocast_context():
                self.predictor.set_image(image_np)
                masks, scores_np, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=self.config.sam2_multimask_output,
                )

            masks, scores_np = self._normalize_predictor_output(
                masks=masks,
                scores=scores_np,
                num_boxes=len(boxes),
                image_size=(h, w),
            )

            if self.config.sam2_multimask_output:
                # masks: [N, C, H, W], scores: [N, C]
                best = np.argmax(scores_np, axis=1)
                masks = masks[np.arange(masks.shape[0]), best]
                scores_np = scores_np[np.arange(scores_np.shape[0]), best]
            else:
                # The predictor still exposes a candidate dimension for some
                # SAM2 versions, even when multimask output is disabled.
                masks = masks[:, 0]
                scores_np = scores_np[:, 0]

            masks_arr = masks.astype(bool, copy=False)
            scores_np = np.asarray(scores_np, dtype=np.float32).reshape(-1)

        return SegmentationResult(
            backend="sam2",
            image_size=(int(h), int(w)),
            prompts=list(det_obj.prompts),
            boxes_xyxy=np.asarray(det_obj.boxes_xyxy, dtype=np.float32),
            prompt_ids=np.asarray(det_obj.prompt_ids, dtype=np.int32),
            scores=scores_np,
            masks=masks_arr.astype(bool),
            prompt_matches=det_obj.prompt_matches,
        )

    @staticmethod
    def _normalize_predictor_output(
        masks: Any,
        scores: Any,
        num_boxes: int,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize SAM2 predictor output to ``[N,C,H,W]`` and ``[N,C]``.

        SAM2 releases differ in whether singleton batch and candidate axes are
        squeezed. In particular, one box with ``multimask_output=True`` may be
        returned as ``[C,H,W]``/``[C]`` rather than ``[1,C,H,W]``/``[1,C]``.
        Normalizing before selecting the best candidate keeps both paths safe.
        """

        def to_numpy(value: Any) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu()
                if tensor.is_floating_point() and tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                return tensor.numpy()
            return np.asarray(value)

        height, width = image_size
        masks_np = to_numpy(masks)
        scores_np = to_numpy(scores)

        if num_boxes <= 0:
            return (
                np.zeros((0, 1, height, width), dtype=bool),
                np.zeros((0, 1), dtype=np.float32),
            )

        if masks_np.ndim == 2:
            masks_np = masks_np[None, None]
        elif masks_np.ndim == 3:
            if masks_np.shape[-2:] != (height, width):
                raise RuntimeError(f"Unexpected SAM2 mask shape: {masks_np.shape}")
            if num_boxes == 1:
                # Squeezed batch: [C,H,W].
                masks_np = masks_np[None]
            elif masks_np.shape[0] == num_boxes:
                # Squeezed candidate axis: [N,H,W].
                masks_np = masks_np[:, None]
            else:
                raise RuntimeError(
                    f"SAM2 returned {masks_np.shape[0]} masks for {num_boxes} boxes"
                )
        elif masks_np.ndim == 4:
            if masks_np.shape[0] != num_boxes:
                raise RuntimeError(
                    f"SAM2 returned {masks_np.shape[0]} mask batches for {num_boxes} boxes"
                )
        else:
            raise RuntimeError(f"Unexpected SAM2 mask shape: {masks_np.shape}")

        if masks_np.shape[0] != num_boxes or masks_np.shape[-2:] != (height, width):
            raise RuntimeError(f"Unexpected SAM2 mask shape: {masks_np.shape}")
        masks_np = masks_np.astype(bool, copy=False)
        candidates = masks_np.shape[1]

        if scores_np.ndim == 0:
            scores_np = scores_np.reshape(1, 1)
        elif scores_np.ndim == 1:
            if num_boxes == 1:
                # Squeezed batch: [C].
                scores_np = scores_np[None]
            elif scores_np.shape[0] == num_boxes:
                # Squeezed candidate axis: [N].
                scores_np = scores_np[:, None]
            else:
                raise RuntimeError(
                    f"SAM2 returned {scores_np.shape[0]} scores for {num_boxes} boxes"
                )
        elif scores_np.ndim != 2:
            raise RuntimeError(f"Unexpected SAM2 score shape: {scores_np.shape}")

        if scores_np.shape != (num_boxes, candidates):
            raise RuntimeError(
                "SAM2 returned inconsistent masks and scores: "
                f"masks={masks_np.shape}, scores={scores_np.shape}"
            )
        return masks_np, scores_np.astype(np.float32, copy=False)

    @torch.no_grad()
    def segment_from_path(
        self, image_path: str | Path, det: DetectionResult | dict[str, Any]
    ) -> SegmentationResult:
        """Backward-compatible helper: load image from path then call segment()."""

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.segment(image, det)


class ObjectSegmentationArgs(BaseModel):
    """CLI arguments for object segmentation."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    image: Path
    """Input image path."""

    detection_json: Path
    """GroundingDINO detection JSON containing XYXY boxes."""

    output_dir: Path | None = None
    """Output directory; defaults to the image directory."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Torch device used by SAM2."""

    sam2_checkpoint: Path = _DEFAULT_SAM2_CHECKPOINT
    """Path to the SAM2.1 checkpoint."""

    sam2_model_cfg: str = _DEFAULT_SAM2_MODEL_CFG
    """SAM2 Hydra model configuration."""

    sam2_multimask_output: bool = False
    """Retain the best of three candidate masks per box."""

    sam2_autocast_dtype: Sam2AutocastDtype = "bfloat16"
    """SAM2 CUDA autocast dtype."""


def _resolve_output_dir(image_path: Path, output_dir: Path | None) -> Path:
    """Resolve output directory.

    If output_dir is not provided, it defaults to the input image directory.
    """

    return output_dir if output_dir is not None else image_path.parent


def _overlay_masks(image_rgb: np.ndarray, masks: np.ndarray, prompt_ids: list[int]) -> np.ndarray:
    """Overlay masks on an RGB image."""

    out = image_rgb.copy()
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
        # cv2 uses BGR; convert palette RGB->BGR
        bgr = (int(color[2]), int(color[1]), int(color[0]))

        overlay = np.zeros_like(out)
        overlay[mask] = bgr
        out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0)

    return out


def main(cli_args: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = CliApp.run(ObjectSegmentationArgs, cli_args=cli_args)
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.detection_json.exists():
        raise FileNotFoundError(f"Detection json not found: {args.detection_json}")

    det_dict = json.loads(args.detection_json.read_text(encoding="utf-8"))
    det = DetectionResult.from_json_dict(det_dict)

    image = Image.open(args.image).convert("RGB")
    image_rgb = np.array(image, dtype=np.uint8)

    out_dir = _resolve_output_dir(args.image, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ObjectSegmentationConfig(
        device=args.device,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        sam2_multimask_output=args.sam2_multimask_output,
        autocast_dtype=args.sam2_autocast_dtype,
    )
    segmentor = Sam2BoxSegmentor(config)

    seg = segmentor.segment(image_rgb, det)

    np.savez_compressed(
        out_dir / "masks.npz",
        masks=seg.masks.astype(np.bool_),
        boxes_xyxy=np.asarray(seg.boxes_xyxy, dtype=np.float32),
        prompt_ids=np.asarray(seg.prompt_ids, dtype=np.int32),
        scores=np.asarray(seg.scores, dtype=np.float32),
        prompts=np.asarray(seg.prompts, dtype=object),
        image_size=np.asarray([seg.image_size[0], seg.image_size[1]], dtype=np.int32),
    )

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    vis = _overlay_masks(bgr, seg.masks, seg.prompt_ids.tolist())
    cv2.imwrite(str(out_dir / "masks_vis.png"), vis)

    (out_dir / "masks_meta.json").write_text(
        json.dumps(seg.meta_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] masks saved under: {out_dir}")


if __name__ == "__main__":
    main()
