"""End-to-end text-grounded instance segmentation backends."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import CliApp

from gda.datatypes import (
    DetectionResult,
    GroundedSegmentationResult,
    SegmentationResult,
)
from gda.modules.object_detection import (
    DEFAULT_GROUNDING_DINO_MODEL_ID,
    DEFAULT_GROUNDING_DINO_MODEL_REVISION,
    GroundingDinoDetector,
    ObjectDetectionConfig,
    _draw_boxes,
)
from gda.modules.object_segmentation import (
    _DEFAULT_SAM2_CHECKPOINT,
    _DEFAULT_SAM2_MODEL_CFG,
    ObjectSegmentationConfig,
    Sam2AutocastDtype,
    Sam2BoxSegmentor,
    _overlay_masks,
)
from gda.modules.sam3_checkpoint import (
    DEFAULT_SAM3_HUGGINGFACE_REVISION,
    DEFAULT_SAM3_MODELSCOPE_REVISION,
    download_sam3_checkpoint,
)
from gda.modules.workspace import third_party_root

_THIRD_PARTY_ROOT = third_party_root()

GroundedBackend = Literal["sam2", "sam3"]
AutocastDtype = Literal["none", "bfloat16", "float16"]
DEFAULT_SAM3_MODEL_REVISION = DEFAULT_SAM3_HUGGINGFACE_REVISION


def _import_sam3_components():
    """Import official SAM3 components lazily."""

    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError:
        sam3_root = _THIRD_PARTY_ROOT / "sam3"
        if not (sam3_root / "sam3").is_dir():
            raise FileNotFoundError(
                f"SAM3 submodule not found: {sam3_root}. Run "
                "`git submodule update --init --recursive`."
            )
        sys.path.insert(0, str(sam3_root))
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:  # pragma: no cover - optional heavyweight runtime
            raise ImportError(
                "Failed to import SAM3. Run `pixi install` from the gda repository."
            ) from exc

    return build_sam3_image_model, Sam3Processor


def _as_rgb_array(image_rgb: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image_rgb, Image.Image):
        return np.asarray(image_rgb.convert("RGB"), dtype=np.uint8)

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape [H,W,3], got {image.shape}")
    return image.astype(np.uint8, copy=False)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.is_floating_point() and tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


class Sam3ConceptSegmentationConfig(BaseModel, frozen=True):
    """Configuration for native SAM3 promptable concept segmentation."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    """Device used for official SAM3 image inference."""

    checkpoint: Path | None = None
    """Optional local SAM3 image checkpoint."""

    load_from_hf: bool = False
    """Use Hugging Face instead of the default ModelScope checkpoint provider."""

    model_revision: str = DEFAULT_SAM3_MODEL_REVISION
    """Exact Hugging Face revision used when that optional provider is selected."""

    modelscope_revision: str = DEFAULT_SAM3_MODELSCOPE_REVISION
    """Exact ModelScope revision used by the default checkpoint provider."""

    local_files_only: bool = False
    """Require the selected provider's pinned checkpoint to already be cached."""

    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Minimum concept-presence-adjusted score retained as an instance."""

    resolution: Literal[1008] = 1008
    """Square image resolution used by the official SAM3 processor."""

    compile: bool = False
    """Compile supported SAM3 image components with torch.compile."""

    autocast_dtype: AutocastDtype = "bfloat16"
    """CUDA automatic mixed-precision dtype; H200 should use bfloat16."""

    deduplicate_mask_iou: float | None = Field(default=0.9, gt=0.0, le=1.0)
    """Cross-prompt mask IoU threshold; set to null to preserve duplicate matches."""


class GroundingDinoSam2Config(BaseModel, frozen=True):
    """Configuration for the GroundingDINO to SAM2.1 baseline."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    detector: ObjectDetectionConfig = ObjectDetectionConfig()
    """Open-vocabulary box detector configuration."""

    segmentor: ObjectSegmentationConfig = ObjectSegmentationConfig()
    """SAM2.1 box-to-mask configuration."""


class GroundedSegmentationConfig(BaseModel, frozen=True):
    """Configuration selecting one grounded instance-segmentation chain."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    backend: GroundedBackend = "sam3"
    """Use the stable GroundingDINO+SAM2 chain or native SAM3 concept segmentation."""

    grounded_sam2: GroundingDinoSam2Config = GroundingDinoSam2Config()
    """Configuration used when backend is sam2."""

    sam3: Sam3ConceptSegmentationConfig = Sam3ConceptSegmentationConfig()
    """Configuration used when backend is sam3."""


class GroundedSegmentor(Protocol):
    """Shared runtime interface for grounded instance segmentors."""

    def segment(
        self, image_rgb: np.ndarray | Image.Image, prompts: list[str]
    ) -> GroundedSegmentationResult: ...


class GroundingDinoSam2Segmentor:
    """Detect with GroundingDINO and refine each box with SAM2.1."""

    def __init__(self, config: GroundingDinoSam2Config):
        self.config = config
        self.detector = GroundingDinoDetector(config.detector)
        self.segmentor = Sam2BoxSegmentor(config.segmentor)

    def segment(
        self, image_rgb: np.ndarray | Image.Image, prompts: list[str]
    ) -> GroundedSegmentationResult:
        det = self.detector.detect(image_rgb, prompts)
        seg = self.segmentor.segment(image_rgb, det)
        return GroundedSegmentationResult(det=det, seg=seg)


class Sam3ConceptSegmentor:
    """Run native SAM3 text-to-mask inference with one shared image encoding."""

    def __init__(self, config: Sam3ConceptSegmentationConfig):
        self.config = config
        build_model, processor_type = _import_sam3_components()

        checkpoint = config.checkpoint
        if checkpoint is None:
            checkpoint = download_sam3_checkpoint(
                load_from_hf=config.load_from_hf,
                modelscope_revision=config.modelscope_revision,
                huggingface_revision=config.model_revision,
                local_files_only=config.local_files_only,
            )
        elif not checkpoint.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

        logger = logging.getLogger("gda.grounded.sam3")
        logger.info(
            "building SAM3 image checkpoint=%s device=%s compile=%s",
            checkpoint,
            config.device,
            config.compile,
        )
        started = time.perf_counter()
        self.model = build_model(
            device=config.device,
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=config.compile,
        )
        self.processor = processor_type(
            self.model,
            resolution=config.resolution,
            device=config.device,
            confidence_threshold=config.confidence_threshold,
        )
        logger.info("SAM3 ready in %.2fs", time.perf_counter() - started)

    def _autocast_context(self):
        if self.config.device != "cuda" or self.config.autocast_dtype == "none":
            return nullcontext()
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.config.autocast_dtype]
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.inference_mode()
    def segment(
        self, image_rgb: np.ndarray | Image.Image, prompts: list[str]
    ) -> GroundedSegmentationResult:
        clean_prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
        if not clean_prompts:
            raise ValueError("prompts must be non-empty")

        image = _as_rgb_array(image_rgb)
        height, width = image.shape[:2]
        boxes_parts: list[np.ndarray] = []
        scores_parts: list[np.ndarray] = []
        masks_parts: list[np.ndarray] = []
        prompt_id_parts: list[np.ndarray] = []
        labels: list[str] = []

        with self._autocast_context():
            state = self.processor.set_image(Image.fromarray(image, mode="RGB"))
            for prompt_id, prompt in enumerate(clean_prompts):
                self.processor.reset_all_prompts(state)
                output = self.processor.set_text_prompt(state=state, prompt=prompt)
                boxes, scores, masks = self._normalize_output(
                    output=output,
                    image_size=(height, width),
                )
                count = len(scores)
                if count == 0:
                    continue
                boxes_parts.append(boxes)
                scores_parts.append(scores)
                masks_parts.append(masks)
                prompt_id_parts.append(np.full(count, prompt_id, dtype=np.int32))
                labels.extend([prompt] * count)

        if boxes_parts:
            boxes = np.concatenate(boxes_parts, axis=0)
            scores = np.concatenate(scores_parts, axis=0)
            masks = np.concatenate(masks_parts, axis=0)
            prompt_ids = np.concatenate(prompt_id_parts, axis=0)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            masks = np.zeros((0, height, width), dtype=bool)
            prompt_ids = np.zeros((0,), dtype=np.int32)

        prompt_matches: list[list[int]] | None = None
        if self.config.deduplicate_mask_iou is not None and len(scores) > 1:
            boxes, scores, masks, prompt_ids, labels, prompt_matches = self._deduplicate(
                boxes=boxes,
                scores=scores,
                masks=masks,
                prompt_ids=prompt_ids,
                labels=labels,
                threshold=self.config.deduplicate_mask_iou,
            )

        det = DetectionResult(
            image_size=(height, width),
            prompts=clean_prompts,
            boxes_xyxy=boxes,
            scores=scores,
            prompt_ids=prompt_ids,
            labels=labels,
            prompt_matches=prompt_matches,
        )
        seg = SegmentationResult(
            backend="sam3",
            image_size=(height, width),
            prompts=clean_prompts,
            boxes_xyxy=boxes,
            prompt_ids=prompt_ids,
            scores=scores,
            masks=masks,
            prompt_matches=prompt_matches,
        )
        return GroundedSegmentationResult(det=det, seg=seg)

    @staticmethod
    def _normalize_output(
        output: dict[str, Any], image_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image_size
        boxes = _to_numpy(output["boxes"]).astype(np.float32, copy=False).reshape(-1, 4)
        scores = _to_numpy(output["scores"]).astype(np.float32, copy=False).reshape(-1)
        masks = _to_numpy(output["masks"])
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim == 2:
            masks = masks[None]
        if masks.size == 0:
            masks = np.zeros((0, height, width), dtype=bool)
        if masks.ndim != 3 or masks.shape[1:] != (height, width):
            raise RuntimeError(f"Unexpected SAM3 mask shape: {masks.shape}")
        if not (len(boxes) == len(scores) == len(masks)):
            raise RuntimeError(
                "SAM3 returned inconsistent boxes, scores, and masks: "
                f"{len(boxes)}, {len(scores)}, {len(masks)}"
            )
        return boxes, scores, masks.astype(bool, copy=False)

    @staticmethod
    def _deduplicate(
        boxes: np.ndarray,
        scores: np.ndarray,
        masks: np.ndarray,
        prompt_ids: np.ndarray,
        labels: list[str],
        threshold: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[str],
        list[list[int]],
    ]:
        order = np.argsort(-scores, kind="stable")
        kept: list[int] = []
        matches: dict[int, set[int]] = {}

        for candidate_index in order.tolist():
            duplicate_of: int | None = None
            candidate_mask = masks[candidate_index]
            for kept_index in kept:
                if prompt_ids[kept_index] == prompt_ids[candidate_index]:
                    continue
                intersection = np.logical_and(candidate_mask, masks[kept_index]).sum()
                if intersection == 0:
                    continue
                union = np.logical_or(candidate_mask, masks[kept_index]).sum()
                if union > 0 and float(intersection / union) >= threshold:
                    duplicate_of = kept_index
                    break

            if duplicate_of is None:
                kept.append(candidate_index)
                matches[candidate_index] = {int(prompt_ids[candidate_index])}
            else:
                matches[duplicate_of].add(int(prompt_ids[candidate_index]))

        kept.sort()
        return (
            boxes[kept],
            scores[kept],
            masks[kept],
            prompt_ids[kept],
            [labels[index] for index in kept],
            [sorted(matches[index]) for index in kept],
        )


def build_grounded_segmentor(config: GroundedSegmentationConfig) -> GroundedSegmentor:
    """Instantiate the selected grounded segmentation backend."""

    if config.backend == "sam3":
        return Sam3ConceptSegmentor(config.sam3)
    return GroundingDinoSam2Segmentor(config.grounded_sam2)


class GroundedSegmentationArgs(BaseModel, frozen=True):
    """CLI arguments for text-grounded instance segmentation."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    image: Path
    """Input image path."""

    prompts: str
    """Comma-, semicolon-, or newline-separated concept prompts."""

    output_dir: Path | None = None
    """Output directory; defaults to the image directory."""

    backend: GroundedBackend = "sam3"
    """Grounded segmentation backend."""

    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    """Inference device."""

    hf_offline: bool = False
    """Use only locally cached Hugging Face files."""

    det_model_id: str = DEFAULT_GROUNDING_DINO_MODEL_ID
    """GroundingDINO model id used by the SAM2 backend."""

    det_model_revision: str | None = DEFAULT_GROUNDING_DINO_MODEL_REVISION
    """Exact GroundingDINO Hugging Face revision used by the SAM2 backend."""

    box_th: float = Field(default=0.25, ge=0.0, le=1.0)
    """GroundingDINO box threshold."""

    text_th: float = Field(default=0.3, ge=0.0, le=1.0)
    """GroundingDINO text threshold."""

    sam2_checkpoint: Path = _DEFAULT_SAM2_CHECKPOINT
    """SAM2.1 checkpoint used by the SAM2 backend."""

    sam2_model_cfg: str = _DEFAULT_SAM2_MODEL_CFG
    """SAM2.1 model configuration."""

    sam2_autocast_dtype: Sam2AutocastDtype = "bfloat16"
    """SAM2 CUDA autocast dtype."""

    sam3_checkpoint: Path | None = None
    """Optional local SAM3 checkpoint."""

    sam3_load_from_hf: bool = False
    """Use Hugging Face instead of the default ModelScope checkpoint provider."""

    sam3_model_revision: str = DEFAULT_SAM3_MODEL_REVISION
    """Exact SAM3 Hugging Face revision used by the optional provider."""

    sam3_modelscope_revision: str = DEFAULT_SAM3_MODELSCOPE_REVISION
    """Exact SAM3 ModelScope revision used by the default provider."""

    sam3_local_files_only: bool = False
    """Require the selected SAM3 checkpoint to already be cached."""

    sam3_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """SAM3 instance confidence threshold."""

    sam3_resolution: Literal[1008] = 1008
    """SAM3 square inference resolution."""

    sam3_compile: bool = False
    """Compile SAM3 image components."""

    sam3_autocast_dtype: AutocastDtype = "bfloat16"
    """SAM3 CUDA autocast dtype."""

    sam3_deduplicate_mask_iou: float | None = Field(default=0.9, gt=0.0, le=1.0)
    """Cross-prompt SAM3 duplicate suppression threshold."""


def _parse_prompts(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;,\n]+", value) if item.strip()]


def main(cli_args: list[str] | None = None) -> None:
    args = CliApp.run(GroundedSegmentationArgs, cli_args=cli_args)
    if args.hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    prompts = _parse_prompts(args.prompts)
    if not prompts:
        raise ValueError("--prompts must contain at least one concept")
    config = GroundedSegmentationConfig(
        backend=args.backend,
        grounded_sam2=GroundingDinoSam2Config(
            detector=ObjectDetectionConfig(
                model_id=args.det_model_id,
                model_revision=args.det_model_revision,
                device=args.device,
                box_threshold=args.box_th,
                text_threshold=args.text_th,
                hf_local_files_only=args.hf_offline,
            ),
            segmentor=ObjectSegmentationConfig(
                device=args.device,
                sam2_checkpoint=args.sam2_checkpoint,
                sam2_model_cfg=args.sam2_model_cfg,
                autocast_dtype=args.sam2_autocast_dtype,
            ),
        ),
        sam3=Sam3ConceptSegmentationConfig(
            device=args.device,
            checkpoint=args.sam3_checkpoint,
            load_from_hf=args.sam3_load_from_hf,
            model_revision=args.sam3_model_revision,
            modelscope_revision=args.sam3_modelscope_revision,
            local_files_only=args.sam3_local_files_only
            or (args.sam3_load_from_hf and args.hf_offline),
            confidence_threshold=args.sam3_confidence_threshold,
            resolution=args.sam3_resolution,
            compile=args.sam3_compile,
            autocast_dtype=args.sam3_autocast_dtype,
            deduplicate_mask_iou=args.sam3_deduplicate_mask_iou,
        ),
    )
    image = Image.open(args.image).convert("RGB")
    result = build_grounded_segmentor(config).segment(image, prompts)
    output_dir = args.output_dir or args.image.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "detections.json").write_text(
        json.dumps(result.det.to_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _draw_boxes(image, result.det).save(output_dir / "detections_vis.png")
    np.savez_compressed(
        output_dir / "masks.npz",
        masks=result.seg.masks.astype(np.bool_),
        boxes_xyxy=result.seg.boxes_xyxy.astype(np.float32),
        prompt_ids=result.seg.prompt_ids.astype(np.int32),
        scores=result.seg.scores.astype(np.float32),
        prompts=np.asarray(result.seg.prompts, dtype=object),
        image_size=np.asarray(result.seg.image_size, dtype=np.int32),
        **(
            {"prompt_matches": np.asarray(result.seg.prompt_matches, dtype=object)}
            if result.seg.prompt_matches is not None
            else {}
        ),
    )
    (output_dir / "masks_meta.json").write_text(
        json.dumps(result.seg.meta_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    masks_vis = _overlay_masks(image_bgr, result.seg.masks, result.seg.prompt_ids.tolist())
    cv2.imwrite(str(output_dir / "masks_vis.png"), masks_vis)
    print(f"[OK] grounded segmentation saved under: {output_dir}")


if __name__ == "__main__":
    main()
