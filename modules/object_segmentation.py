"""Object segmentation (mask) module for a single image.

This module consumes:
- an image
- detection results (boxes) produced by object_detection.py

It outputs:
- masks.npz (boolean masks and metadata)
- masks_vis.png (overlay visualization)

Backends:
- sam2 (default): uses the local Grounded-SAM-2 code under sdf_compute/thirdparty.
- sam3: uses the local sam3 repo.

Notes:
- This module follows prompts/prepare.md:
    - class initialization takes exactly one `config` argument (pydantic BaseModel)
    - CLI arguments are parsed via pydantic_settings
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


_REPO_ROOT = Path(__file__).resolve().parents[2]

_SAM2_ROOT = _REPO_ROOT / "sdf_compute" / "thirdparty" / "grounded_sam_2"
if not _SAM2_ROOT.exists():
    raise FileNotFoundError(
        f"SAM2 repo path not found: {_SAM2_ROOT}. Please check the workspace layout."
    )
if str(_SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM2_ROOT))


_DEFAULT_SAM2_CHECKPOINT = (
    _REPO_ROOT
    / "sdf_compute"
    / "thirdparty"
    / "grounded_sam_2"
    / "checkpoints"
    / "sam2.1_hiera_large.pt"
)
_DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


def _import_sam3_builder():
    """Import SAM3 builder lazily.

    SAM3 is an optional backend; importing it at module import time would:
    - slow down `--help`
    - trigger warnings
    - fail even when using the default SAM2 backend
    """

    sam3_repo_root = _REPO_ROOT / "sam3"
    if not sam3_repo_root.exists():
        raise FileNotFoundError(
            f"sam3 repo path not found: {sam3_repo_root}. Please check the workspace layout."
        )
    if str(sam3_repo_root) not in sys.path:
        sys.path.insert(0, str(sam3_repo_root))

    try:
        from sam3.model_builder import build_sam3_image_model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Failed to import sam3. Please ensure the sam3 repo dependencies are installed."
        ) from e

    return build_sam3_image_model


class ObjectSegmentationConfig(BaseModel):
    """Configuration for mask segmentation from boxes."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["sam2", "sam3"] = "sam2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # sam2
    sam2_checkpoint: Path = _DEFAULT_SAM2_CHECKPOINT
    sam2_model_cfg: str = _DEFAULT_SAM2_MODEL_CFG
    sam2_multimask_output: bool = False

    # sam3
    sam3_checkpoint: Path | None = None
    sam3_load_from_hf: bool = False
    sam3_multimask_output: bool = False


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

        device = config.device
        self.model = build_sam2(
            config.sam2_model_cfg,
            ckpt_path=str(config.sam2_checkpoint),
            device=device,
        )
        self.predictor = SAM2ImagePredictor(self.model)

    @torch.no_grad()
    def segment(self, image_path: str | Path, det: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        """Generate masks from detection boxes.

        Args:
            image_path: Path to an image.
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

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)
        h, w = image_np.shape[:2]

        boxes = np.asarray(det.get("boxes_xyxy", []), dtype=np.float32)
        if boxes.size == 0:
            masks_arr = np.zeros((0, h, w), dtype=bool)
            scores = []
        else:
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("det['boxes_xyxy'] must be N x 4")

            self.predictor.set_image(image_np)
            masks, scores_np, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=self.config.sam2_multimask_output,
            )

            # Normalize shapes
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
                scores_np = np.asarray(scores_np).reshape(-1)

            if self.config.sam2_multimask_output:
                # masks: [N, C, H, W], scores: [N, C]
                best = np.argmax(scores_np, axis=1)
                masks = masks[np.arange(masks.shape[0]), best]
                scores_np = scores_np[np.arange(scores_np.shape[0]), best]

            masks_arr = masks.astype(bool)
            scores = [float(x) for x in np.asarray(scores_np).reshape(-1).tolist()]

        meta = {
            "backend": "sam2",
            "image_size": [h, w],
            "prompts": det.get("prompts", []),
            "boxes_xyxy": det.get("boxes_xyxy", []),
            "prompt_ids": det.get("prompt_ids", []),
            "scores": scores,
            "_masks_shape": list(masks_arr.shape),
        }
        return meta, masks_arr


class Sam3BoxSegmentor:
    """Segment objects from bounding boxes using SAM3 interactive predictor."""

    def __init__(self, config: ObjectSegmentationConfig):
        """Create segmentor.

        Args:
            config: SAM3 checkpoint/device options.
        """

        self.config = config
        build_sam3_image_model = _import_sam3_builder()

        self.model = build_sam3_image_model(
            device="cuda" if str(config.device).startswith("cuda") else "cpu",
            checkpoint_path=str(config.sam3_checkpoint) if config.sam3_checkpoint else None,
            load_from_HF=config.sam3_load_from_hf,
            enable_segmentation=False,
            enable_inst_interactivity=True,
        )
        predictor = getattr(self.model, "inst_interactive_predictor", None)
        if predictor is None:
            raise RuntimeError("SAM3 model does not expose inst_interactive_predictor")
        self.predictor = predictor

    @torch.no_grad()
    def segment(self, image_path: str | Path, det: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        """Generate masks from detection boxes using SAM3.

        Args:
            image_path: Path to an image.
            det: Detection result dict (from object_detection.py).

        Returns:
            (meta, masks)
            - meta: JSON-serializable dict
            - masks: boolean numpy array with shape [N, H, W]
        """

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)
        h, w = image_np.shape[:2]

        boxes = np.asarray(det.get("boxes_xyxy", []), dtype=np.float32)
        if boxes.size == 0:
            masks_arr = np.zeros((0, h, w), dtype=bool)
            scores = []
        else:
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("det['boxes_xyxy'] must be N x 4")

            self.predictor.set_image(image_np)

            masks_list = []
            scores = []
            for box in boxes:
                masks, ious, _ = self.predictor.predict(
                    box=box,
                    multimask_output=self.config.sam3_multimask_output,
                    return_logits=False,
                    normalize_coords=True,
                )
                if masks.ndim != 3:
                    raise RuntimeError("Unexpected mask shape from SAM3 predictor")
                masks_list.append(masks[0].astype(bool))
                scores.append(float(ious[0]) if np.size(ious) > 0 else 0.0)

            masks_arr = np.stack(masks_list, axis=0)

        meta = {
            "backend": "sam3",
            "image_size": [h, w],
            "prompts": det.get("prompts", []),
            "boxes_xyxy": det.get("boxes_xyxy", []),
            "prompt_ids": det.get("prompt_ids", []),
            "scores": scores,
            "_masks_shape": list(masks_arr.shape),
        }
        return meta, masks_arr


class ObjectSegmentationCLI(BaseSettings):
    """CLI arguments for object segmentation."""

    model_config = SettingsConfigDict(cli_parse_args=True, extra="ignore")

    image: Path
    detection_json: Path
    output_dir: Path | None = None

    backend: Literal["sam2", "sam3"] = "sam2"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    sam2_checkpoint: Path = _DEFAULT_SAM2_CHECKPOINT
    sam2_model_cfg: str = _DEFAULT_SAM2_MODEL_CFG
    sam2_multimask_output: bool = False

    sam3_checkpoint: Path | None = None
    sam3_load_from_hf: bool = False
    sam3_multimask_output: bool = False


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


def main() -> None:
    """CLI entrypoint."""

    args = ObjectSegmentationCLI()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.detection_json.exists():
        raise FileNotFoundError(f"Detection json not found: {args.detection_json}")

    det = json.loads(args.detection_json.read_text(encoding="utf-8"))

    out_dir = _resolve_output_dir(args.image, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ObjectSegmentationConfig(
        backend=args.backend,
        device=args.device,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        sam2_multimask_output=args.sam2_multimask_output,
        sam3_checkpoint=args.sam3_checkpoint,
        sam3_load_from_hf=args.sam3_load_from_hf,
        sam3_multimask_output=args.sam3_multimask_output,
    )

    if config.backend == "sam3":
        if config.sam3_load_from_hf and config.sam3_checkpoint is None:
            raise RuntimeError(
                "SAM3 backend requires checkpoint access. Either provide --sam3_checkpoint "
                "to a local checkpoint, or authenticate to HuggingFace and use --sam3_load_from_hf true."
            )
        segmentor = Sam3BoxSegmentor(config)
    else:
        segmentor = Sam2BoxSegmentor(config)

    meta, masks = segmentor.segment(args.image, det)

    np.savez_compressed(
        out_dir / "masks.npz",
        masks=masks.astype(np.bool_),
        boxes_xyxy=np.asarray(det.get("boxes_xyxy", []), dtype=np.float32),
        prompt_ids=np.asarray(det.get("prompt_ids", []), dtype=np.int32),
        scores=np.asarray(meta.get("scores", []), dtype=np.float32),
        prompts=np.asarray(det.get("prompts", []), dtype=object),
        image_size=np.asarray(det.get("image_size", []), dtype=np.int32),
    )

    image = Image.open(args.image).convert("RGB")
    rgb = np.array(image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis = _overlay_masks(bgr, masks, det.get("prompt_ids", []))
    cv2.imwrite(str(out_dir / "masks_vis.png"), vis)

    (out_dir / "masks_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] masks saved under: {out_dir}")


if __name__ == "__main__":
    main()
