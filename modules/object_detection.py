"""Object detection (bbox) module for a single image using GroundingDINO.

This file contains:
- a configurable module class (single `config` argument, pydantic BaseModel)
- a CLI entry implemented with pydantic_settings

The detector takes an image and multi-target prompts, and returns boxes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class ObjectDetectionConfig(BaseModel):
    """Configuration for GroundingDINO detection."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = "IDEA-Research/grounding-dino-base"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    box_threshold: float = 0.25
    text_threshold: float = 0.3


class GroundingDinoDetector:
    """Detect objects with GroundingDINO given multi-target prompts."""

    def __init__(self, config: ObjectDetectionConfig):
        """Create detector.

        Args:
            config: Model id, thresholds, device.
        """

        self.config = config
        self.device = torch.device(config.device)
        self.processor = AutoProcessor.from_pretrained(config.model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(config.model_id)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def detect(self, image_path: str | Path, prompts: list[str]) -> dict[str, Any]:
        """Run detection on a single image.

        Args:
            image_path: Path to an image.
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

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not prompts:
            raise ValueError("prompts must be non-empty")

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        text = ". ".join([p.strip() for p in prompts if p.strip()])
        if not text.endswith("."):
            text = text + "."

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[(h, w)],
        )
        r0 = results[0]

        boxes = r0.get("boxes")
        scores = r0.get("scores")
        labels = r0.get("labels")

        if boxes is None or scores is None:
            raise RuntimeError("Unexpected GroundingDINO postprocess output")

        boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
        scores_np = scores.detach().cpu().numpy().astype(np.float32)
        labels_list = [str(x) for x in (labels or [])]

        prompts_norm = [p.strip().lower() for p in prompts]
        prompt_ids: list[int] = []
        for lbl in labels_list:
            lbl_n = lbl.strip().lower()
            pid = -1
            if lbl_n in prompts_norm:
                pid = prompts_norm.index(lbl_n)
            else:
                for i, p in enumerate(prompts_norm):
                    if p and (p in lbl_n or lbl_n in p):
                        pid = i
                        break
            prompt_ids.append(pid)

        return {
            "image_size": [h, w],
            "prompts": prompts,
            "boxes_xyxy": boxes_np.tolist(),
            "scores": scores_np.tolist(),
            "prompt_ids": prompt_ids,
            "labels": labels_list,
        }


class ObjectDetectionCLI(BaseSettings):
    """CLI arguments for object detection."""

    model_config = SettingsConfigDict(cli_parse_args=True, extra="ignore")

    image: Path
    prompts: str
    output_dir: Path | None = None

    model_id: str = "IDEA-Research/grounding-dino-base"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    box_th: float = 0.25
    text_th: float = 0.3


def _parse_prompts(prompts: str) -> list[str]:
    """Parse prompts string into a list.

    Accepts separators: ';' or ',' (and also newlines).

    Note: when running via `conda run`, unquoted semicolons may be interpreted
    by the shell because `conda run` generates a temporary script. Commas are
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

        label = prompts[pid] if (pid is not None and pid >= 0 and pid < len(prompts)) else "unknown"
        text = f"{label} {score:.2f}"
        if font is not None:
            draw.text((x1 + 3, y1 + 3), text, fill=color, font=font)
        else:
            draw.text((x1 + 3, y1 + 3), text, fill=color)

    return img


def main() -> None:
    """CLI entrypoint."""

    args = ObjectDetectionCLI()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    prompts_list = _parse_prompts(args.prompts)
    if not prompts_list:
        raise ValueError("--prompts must contain at least one item")

    out_dir = _resolve_output_dir(args.image, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = GroundingDinoDetector(
        ObjectDetectionConfig(
            model_id=args.model_id,
            device=args.device,
            box_threshold=args.box_th,
            text_threshold=args.text_th,
        )
    )

    det = detector.detect(args.image, prompts_list)

    (out_dir / "detections.json").write_text(
        json.dumps(det, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    image = Image.open(args.image).convert("RGB")
    vis = _draw_boxes(image, det)
    vis.save(out_dir / "detections_vis.png")

    print(f"[OK] detections saved under: {out_dir}")


if __name__ == "__main__":
    main()
