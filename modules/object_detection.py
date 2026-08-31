"""Object detection (bbox) module for a single image using GroundingDINO.

This file contains:
- a configurable module class (single `config` argument, pydantic BaseModel)
- a CLI entry implemented with pydantic_settings

The detector takes an image and multi-target prompts, and returns boxes.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import CliApp

from gda.datatypes import DetectionResult
from gda.modules.falcon_perception import (
    DEFAULT_FALCON_DETECTION_MODEL_ID,
    DEFAULT_FALCON_DETECTION_MODEL_REVISION,
    FalconPerceptionConfig,
    FalconPerceptionRunner,
)

DEFAULT_GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
DEFAULT_GROUNDING_DINO_MODEL_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"


def _import_grounding_dino_components():
    """Import Transformers lazily so offline flags can be configured first."""

    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("GroundingDINO dependencies are missing. Run `pixi install`.") from exc
    return AutoModelForZeroShotObjectDetection, AutoProcessor


class ObjectDetectionConfig(BaseModel, frozen=True):
    """Configuration for an open-vocabulary box detector."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    backend: Literal["grounding_dino", "falcon"] = "grounding_dino"
    """Detector backend; Falcon uses the 300M detection-only model."""

    model_id: str = DEFAULT_GROUNDING_DINO_MODEL_ID
    """Hugging Face GroundingDINO repository or local model directory."""

    model_revision: str | None = DEFAULT_GROUNDING_DINO_MODEL_REVISION
    """Exact Hugging Face model revision; set to null only for a local directory."""

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    """Torch device used for box detection."""

    box_threshold: float = 0.25
    """Minimum box confidence retained by post-processing."""

    text_threshold: float = 0.3
    """Minimum text-token confidence retained by post-processing."""

    hf_local_files_only: bool = False
    """Require the pinned model files to exist in the local Hugging Face cache."""

    falcon: FalconPerceptionConfig = FalconPerceptionConfig(
        model_id=DEFAULT_FALCON_DETECTION_MODEL_ID,
        model_revision=DEFAULT_FALCON_DETECTION_MODEL_REVISION,
    )
    """Falcon detector configuration used when backend is ``falcon``."""


class GroundingDinoDetector:
    """Detect objects with GroundingDINO given multi-target prompts."""

    def __init__(self, config: ObjectDetectionConfig):
        """Create detector.

        Args:
            config: Model id, thresholds, device.
        """

        self.config = config
        self.device = torch.device(config.device)

        logger = logging.getLogger("gda.det")
        model_type, processor_type = _import_grounding_dino_components()
        logger.info("loading GroundingDINO processor=%s", config.model_id)
        t0 = time.perf_counter()
        self.processor = processor_type.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            local_files_only=bool(config.hf_local_files_only),
        )
        logger.info("processor loaded in %.2fs", time.perf_counter() - t0)

        logger.info("loading GroundingDINO model=%s device=%s", config.model_id, config.device)
        t0 = time.perf_counter()
        self.model = (
            model_type.from_pretrained(
                config.model_id,
                revision=config.model_revision,
                local_files_only=bool(config.hf_local_files_only),
            )
            .to(self.device)
            .eval()
        )
        logger.info("model loaded in %.2fs", time.perf_counter() - t0)

    @torch.no_grad()
    def detect(self, image_rgb: np.ndarray | Image.Image, prompts: list[str]) -> DetectionResult:
        """Run detection on a single image.

        Args:
            image_rgb: Input image as RGB (numpy uint8 array [H,W,3] or PIL Image).
            prompts: List of target prompts (multi-target).

        Returns:
            A JSON-serializable dict containing:
              - image_size: [H, W]
              - prompts: list[str]
              - boxes_xyxy: list[list[float]] (N x 4)
              - scores: list[float] (N)
              - prompt_ids: list[int] (N) maps each box to prompts
              - labels: list[str] (N) raw labels returned by model
        """

        clean_prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
        if not clean_prompts:
            raise ValueError("prompts must be non-empty")

        if isinstance(image_rgb, Image.Image):
            image = image_rgb.convert("RGB")
        else:
            arr = np.asarray(image_rgb)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"image_rgb must have shape [H,W,3], got {arr.shape}")
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            image = Image.fromarray(arr, mode="RGB")
        w, h = image.size

        # Run one prompt at a time. GroundingDINO's post-processing returns a
        # token phrase, not a stable category index; combining prompts and then
        # guessing ownership from substring matches silently mislabels overlaps
        # such as "car"/"car wheel". The extra forwards only affect the explicit
        # SAM2 fallback; native SAM3 remains the default multi-prompt path.
        boxes_parts: list[np.ndarray] = []
        scores_parts: list[np.ndarray] = []
        labels_list: list[str] = []
        prompt_ids_parts: list[np.ndarray] = []
        for prompt_id, prompt in enumerate(clean_prompts):
            text = prompt if prompt.endswith(".") else f"{prompt}."
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                target_sizes=[(h, w)],
            )
            result = results[0]
            boxes = result.get("boxes")
            scores = result.get("scores")
            labels = result.get("labels")
            if boxes is None or scores is None:
                raise RuntimeError("Unexpected GroundingDINO postprocess output")
            boxes_np = boxes.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
            scores_np = scores.detach().cpu().numpy().astype(np.float32).reshape(-1)
            labels_for_prompt = [str(label) for label in labels] if labels is not None else []
            if len(labels_for_prompt) != len(scores_np):
                labels_for_prompt = [prompt] * len(scores_np)
            boxes_parts.append(boxes_np)
            scores_parts.append(scores_np)
            labels_list.extend(labels_for_prompt)
            prompt_ids_parts.append(np.full(len(scores_np), prompt_id, dtype=np.int32))

        boxes_np = (
            np.concatenate(boxes_parts, axis=0)
            if boxes_parts
            else np.zeros((0, 4), dtype=np.float32)
        )
        scores_np = (
            np.concatenate(scores_parts, axis=0)
            if scores_parts
            else np.zeros((0,), dtype=np.float32)
        )
        prompt_ids = (
            np.concatenate(prompt_ids_parts, axis=0)
            if prompt_ids_parts
            else np.zeros((0,), dtype=np.int32)
        )

        return DetectionResult(
            image_size=(int(h), int(w)),
            prompts=clean_prompts,
            boxes_xyxy=boxes_np,
            scores=scores_np,
            prompt_ids=np.asarray(prompt_ids, dtype=np.int32),
            labels=labels_list,
        )

    @torch.no_grad()
    def detect_from_path(self, image_path: str | Path, prompts: list[str]) -> DetectionResult:
        """Backward-compatible helper: load image from path then call detect()."""

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.detect(image, prompts)


class FalconPerceptionDetector:
    """Detect objects with Falcon-Perception-300M text grounding."""

    def __init__(self, config: ObjectDetectionConfig):
        self.config = config
        falcon_config = config.falcon.model_copy(
            update={
                "device": config.device,
                "hf_local_files_only": config.hf_local_files_only,
            }
        )
        self.runner = FalconPerceptionRunner(falcon_config)

    @torch.inference_mode()
    def detect(self, image_rgb: np.ndarray | Image.Image, prompts: list[str]) -> DetectionResult:
        """Run Falcon once per prompt and normalize boxes to GDA's contract."""

        clean_prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
        if not clean_prompts:
            raise ValueError("prompts must be non-empty")
        if isinstance(image_rgb, Image.Image):
            image = image_rgb.convert("RGB")
        else:
            image = np.asarray(image_rgb)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"image_rgb must have shape [H,W,3], got {image.shape}")
        if isinstance(image, Image.Image):
            height, width = image.height, image.width
        else:
            height, width = image.shape[:2]
        image_size = (int(height), int(width))

        boxes_parts: list[np.ndarray] = []
        scores_parts: list[np.ndarray] = []
        prompt_ids_parts: list[np.ndarray] = []
        labels: list[str] = []
        for prompt_id, prompt in enumerate(clean_prompts):
            predictions = self.runner.generate(image_rgb, prompt, task="detection")
            for prediction in predictions:
                boxes_parts.append(self.runner.box(prediction, image_size)[None, :])
                scores_parts.append(np.asarray([self.config.falcon.score], dtype=np.float32))
                prompt_ids_parts.append(np.asarray([prompt_id], dtype=np.int32))
                labels.append(prompt)

        boxes = (
            np.concatenate(boxes_parts, axis=0)
            if boxes_parts
            else np.zeros((0, 4), dtype=np.float32)
        )
        scores = (
            np.concatenate(scores_parts, axis=0)
            if scores_parts
            else np.zeros((0,), dtype=np.float32)
        )
        prompt_ids = (
            np.concatenate(prompt_ids_parts, axis=0)
            if prompt_ids_parts
            else np.zeros((0,), dtype=np.int32)
        )
        return DetectionResult(
            image_size=image_size,
            prompts=clean_prompts,
            boxes_xyxy=boxes,
            scores=scores,
            prompt_ids=prompt_ids,
            labels=labels,
        )

    def detect_from_path(self, image_path: str | Path, prompts: list[str]) -> DetectionResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.detect(Image.open(image_path).convert("RGB"), prompts)


ObjectDetector = GroundingDinoDetector | FalconPerceptionDetector


def build_object_detector(config: ObjectDetectionConfig) -> ObjectDetector:
    """Instantiate the configured open-vocabulary detector."""

    if config.backend == "falcon":
        return FalconPerceptionDetector(config)
    return GroundingDinoDetector(config)


class ObjectDetectionArgs(BaseModel, frozen=True):
    """CLI arguments for object detection."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    backend: Literal["grounding_dino", "falcon"] = "grounding_dino"
    """Detector backend."""

    image: Path
    """Input image path."""

    prompts: str
    """Comma-, semicolon-, or newline-separated text concepts."""

    output_dir: Path | None = None
    """Output directory; defaults to the input image directory."""

    model_id: str = DEFAULT_GROUNDING_DINO_MODEL_ID
    """GroundingDINO model repository or local directory."""

    model_revision: str | None = DEFAULT_GROUNDING_DINO_MODEL_REVISION
    """Exact GroundingDINO Hugging Face revision."""

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    """Torch device used for detection."""

    hf_local_files_only: bool = False
    """Require the selected model files to already be cached locally."""

    box_th: float = 0.25
    """Minimum box confidence retained by post-processing."""

    text_th: float = 0.3
    """Minimum text-token confidence retained by post-processing."""

    falcon_model_id: str = DEFAULT_FALCON_DETECTION_MODEL_ID
    """Falcon-Perception-300M model repository or local export directory."""

    falcon_model_revision: str | None = DEFAULT_FALCON_DETECTION_MODEL_REVISION
    """Pinned Falcon model revision; set to null for a local directory."""

    falcon_dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    """Falcon model dtype."""

    falcon_compile: bool = False
    """Compile Falcon's inference path on first use."""

    falcon_score: float = Field(default=1.0, ge=0.0, le=1.0)
    """Constant score stored for Falcon predictions without confidence output."""


def _parse_prompts(prompts: str) -> list[str]:
    """Parse prompts string into a list.

    Accepts separators: ';' or ',' (and also newlines).

    Note: when passing prompts through a shell runner, unquoted semicolons may be
    interpreted as command separators. Commas are
    safer in that case.
    """

    items = [p.strip() for p in re.split(r"[;,\n]+", prompts)]
    return [p for p in items if p]


def _resolve_output_dir(image_path: Path, output_dir: Path | None) -> Path:
    """Resolve output directory.

    If output_dir is not provided, it defaults to the input image directory.
    """

    return output_dir if output_dir is not None else image_path.parent


def _draw_boxes(image: Image.Image, det: dict[str, Any]) -> Image.Image:
    """Draw detection boxes on image."""

    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if isinstance(det, DetectionResult):
        boxes = det.boxes_xyxy.tolist()
        scores = det.scores.tolist()
        prompt_ids = det.prompt_ids.tolist()
        prompts = det.prompts
    else:
        boxes = det["boxes_xyxy"]
        scores = det["scores"]
        prompt_ids = det["prompt_ids"]
        prompts = det["prompts"]

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

    for box, score, pid in zip(boxes, scores, prompt_ids):
        x1, y1, x2, y2 = [float(v) for v in box]
        color = palette[pid % len(palette)] if pid >= 0 else (255, 255, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = (
            prompts[pid] if (pid is not None and pid >= 0 and pid < len(prompts)) else "unknown"
        )
        text = f"{label} {score:.2f}"
        if font is not None:
            draw.text((x1 + 3, y1 + 3), text, fill=color, font=font)
        else:
            draw.text((x1 + 3, y1 + 3), text, fill=color)

    return img


def main(cli_args: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = CliApp.run(ObjectDetectionArgs, cli_args=cli_args)
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    prompts_list = _parse_prompts(args.prompts)
    if not prompts_list:
        raise ValueError("--prompts must contain at least one item")

    out_dir = _resolve_output_dir(args.image, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = build_object_detector(
        ObjectDetectionConfig(
            backend=args.backend,
            model_id=args.model_id,
            model_revision=args.model_revision,
            device=args.device,
            box_threshold=args.box_th,
            text_threshold=args.text_th,
            falcon=FalconPerceptionConfig(
                model_id=args.falcon_model_id,
                model_revision=args.falcon_model_revision,
                device=args.device,
                dtype=args.falcon_dtype,
                compile=args.falcon_compile,
                score=args.falcon_score,
                hf_local_files_only=args.hf_local_files_only,
            ),
        )
    )

    image = Image.open(args.image).convert("RGB")
    det = detector.detect(image, prompts_list)

    (out_dir / "detections.json").write_text(
        json.dumps(det.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    vis = _draw_boxes(image, det)
    vis.save(out_dir / "detections_vis.png")

    print(f"[OK] detections saved under: {out_dir}")


if __name__ == "__main__":
    main()
