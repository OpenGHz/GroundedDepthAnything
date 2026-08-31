"""Shared Falcon Perception model loading and output normalization.

Falcon Perception is loaded from its Hugging Face export with
``trust_remote_code=True``.  The model exposes a small ``generate`` API that
returns normalized center/size boxes and, for the full model, COCO-RLE masks.
This module keeps that vendor-specific API behind a small runner used by both
the detector and grounded-segmentation adapters.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_FALCON_PERCEPTION_MODEL_ID = "tiiuae/Falcon-Perception"
DEFAULT_FALCON_PERCEPTION_MODEL_REVISION = "a3e21ada0abd533393776ffab8908a728f775509"
DEFAULT_FALCON_DETECTION_MODEL_ID = "tiiuae/Falcon-Perception-300M"
DEFAULT_FALCON_DETECTION_MODEL_REVISION = "36993684b3ba5945b22d01e705f8ff2e048bf0b0"

FalconDtype = Literal["float32", "bfloat16", "float16"]

_SAFE_FLEX_ATTENTION_OPTIONS = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "num_stages": 1,
}


class FalconPerceptionConfig(BaseModel, frozen=True):
    """Configuration shared by Falcon detection and segmentation adapters."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    model_id: str = DEFAULT_FALCON_PERCEPTION_MODEL_ID
    """Hugging Face Falcon model repository or local export directory."""

    model_revision: str | None = DEFAULT_FALCON_PERCEPTION_MODEL_REVISION
    """Pinned Hugging Face revision; set to null for a local directory."""

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    """Torch device used by Falcon inference."""

    dtype: FalconDtype = "float32"
    """Model weight and inference dtype."""

    hf_local_files_only: bool = False
    """Require the model export to already exist in the local HF cache."""

    compile: bool = False
    """Allow Falcon's model API to compile its inference path on first use."""

    max_new_tokens: int = Field(default=2048, gt=0)
    """Maximum autoregressive tokens used to enumerate instances."""

    min_image_size: int = Field(default=256, gt=0)
    """Minimum image side passed to Falcon preprocessing."""

    max_image_size: int = Field(default=1024, gt=0)
    """Maximum image side passed to Falcon preprocessing."""

    segmentation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Sigmoid threshold used by Falcon when producing binary masks."""

    score: float = Field(default=1.0, ge=0.0, le=1.0)
    """Fallback score because Falcon's public output has no confidence field."""

    flex_attention_safe: bool = False
    """Use conservative FlexAttention tiles for GPUs with limited shared memory."""


def _import_falcon_model():
    """Import Transformers lazily so SAM-only workflows stay lightweight."""

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional heavyweight runtime
        raise ImportError("Falcon Perception requires Transformers. Run `pixi install`.") from exc
    return AutoModelForCausalLM


class _FalconTokenizer:
    """Small tokenizer adapter required by Falcon's custom model code.

    Falcon exports a Rust ``tokenizers`` tokenizer but currently declares the
    Transformers class name ``TokenizersBackend``.  That class is not shipped
    by the installed Transformers release, so using ``AutoTokenizer`` fails
    before inference starts.  Falcon's processing code only needs this small
    subset of the tokenizer API.
    """

    def __init__(self, export_dir: Path):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(export_dir / "tokenizer.json"))
        config_path = export_dir / "tokenizer_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        )
        special_tokens = dict(config.get("model_specific_special_tokens", {}))
        for name in (
            "eos_token",
            "bos_token",
            "pad_token",
        ):
            if name in config and isinstance(config[name], str):
                special_tokens.setdefault(name, config[name])
        self.special_tokens_map = {
            name: token for name, token in special_tokens.items() if isinstance(token, str)
        }
        for name, token in self.special_tokens_map.items():
            setattr(self, name, token)
            setattr(self, f"{name}_id", self.convert_tokens_to_ids(token))
        self.eos_token_id = self.convert_tokens_to_ids(config.get("eos_token", "<|end_of_text|>"))
        self.bos_token_id = (
            self.convert_tokens_to_ids(config["bos_token"])
            if isinstance(config.get("bos_token"), str)
            else None
        )
        self.pad_token_id = self.convert_tokens_to_ids(config.get("pad_token", "<|pad|>"))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def convert_ids_to_tokens(self, token_id: int) -> str | None:
        return self._tok.id_to_token(int(token_id))


def _resolve_falcon_export(config: FalconPerceptionConfig) -> Path:
    """Resolve a local model export, downloading a pinned HF snapshot if needed."""

    local_path = Path(config.model_id).expanduser()
    if local_path.is_dir():
        return local_path
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency is project-pinned
        raise ImportError(
            "Falcon Perception requires huggingface-hub. Run `pixi install`."
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": config.model_id,
        "repo_type": "model",
        "local_files_only": bool(config.hf_local_files_only),
    }
    if config.model_revision is not None:
        kwargs["revision"] = config.model_revision
    export_dir = Path(snapshot_download(**kwargs))
    tokenizer_path = export_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_kwargs: dict[str, Any] = {
            "repo_id": config.model_id,
            "filename": "tokenizer.json",
            "local_files_only": bool(config.hf_local_files_only),
            "local_dir": str(export_dir),
        }
        if config.model_revision is not None:
            tokenizer_kwargs["revision"] = config.model_revision
        hf_hub_download(**tokenizer_kwargs)
    return export_dir


def _configure_falcon_attention(model: Any, *, safe: bool) -> None:
    """Patch Falcon's prefill FlexAttention with safe Triton tile sizes.

    The upstream remote model currently accepts ``flex_attn_kernel_options`` in
    layer signatures but does not forward it to ``flex_attention``.  Replacing
    its imported prefill callable keeps this workaround local to GDA and avoids
    editing the Hugging Face cache.  The safe tiles are needed on RTX 30/40 and
    similar GPUs whose shared-memory limit is below the upstream default.
    """

    if not safe:
        return
    try:
        modeling_module = importlib.import_module(model.__class__.__module__)
        from torch.nn.attention.flex_attention import flex_attention
    except (ImportError, AttributeError):  # pragma: no cover - remote-code variant
        return

    safe_options = dict(_SAFE_FLEX_ATTENTION_OPTIONS)

    def safe_prefill(*args, **kwargs):
        kwargs["kernel_options"] = safe_options
        return flex_attention(*args, **kwargs)

    modeling_module.compiled_flex_attn_prefill = torch.compile(safe_prefill, dynamic=True)


def _as_rgb_image(image_rgb: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image_rgb, Image.Image):
        return image_rgb.convert("RGB")
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def _decode_rle_mask(rle: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
    """Decode one Falcon COCO-RLE mask and align it to ``(height, width)``."""

    if not isinstance(rle, dict) or "counts" not in rle or "size" not in rle:
        raise ValueError("Falcon prediction has an invalid mask_rle object")
    source = dict(rle)
    if isinstance(source["counts"], str):
        source["counts"] = source["counts"].encode("utf-8")
    mask = np.asarray(mask_utils.decode(source)).astype(bool, copy=False)
    if mask.ndim == 3:
        if mask.shape[-1] != 1:
            raise ValueError(f"Falcon returned multiple masks in one RLE: {mask.shape}")
        mask = mask[..., 0]

    target_h, target_w = image_size
    if mask.shape != (target_h, target_w):
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (target_w, target_h), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ).astype(bool)
    return mask


def _prediction_box(prediction: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
    """Convert Falcon normalized center/size output to pixel ``xyxy``."""

    height, width = image_size
    try:
        xy = prediction["xy"]
        hw = prediction["hw"]
        cx = float(xy["x"])
        cy = float(xy["y"])
        box_h = float(hw["h"])
        box_w = float(hw["w"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Falcon prediction has invalid xy/hw fields: {prediction}") from exc

    cx = float(np.clip(cx, 0.0, 1.0)) * width
    cy = float(np.clip(cy, 0.0, 1.0)) * height
    box_w = max(0.0, float(np.clip(box_w, 0.0, 1.0)) * width)
    box_h = max(0.0, float(np.clip(box_h, 0.0, 1.0)) * height)
    return np.asarray(
        [
            np.clip(cx - box_w / 2.0, 0.0, width),
            np.clip(cy - box_h / 2.0, 0.0, height),
            np.clip(cx + box_w / 2.0, 0.0, width),
            np.clip(cy + box_h / 2.0, 0.0, height),
        ],
        dtype=np.float32,
    )


class FalconPerceptionRunner:
    """Run Falcon's public single-image generation API."""

    def __init__(self, config: FalconPerceptionConfig):
        self.config = config
        self.device = torch.device(config.device)
        model_type = _import_falcon_model()
        export_dir = _resolve_falcon_export(config)
        dtype = getattr(torch, config.dtype)
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": bool(config.hf_local_files_only),
        }
        if dtype is not None:
            kwargs["dtype"] = dtype

        logger = logging.getLogger("gda.falcon")
        logger.info("loading Falcon model=%s device=%s", config.model_id, config.device)
        started = time.perf_counter()
        self.model = model_type.from_pretrained(str(export_dir), **kwargs)
        # Avoid AutoTokenizer's unsupported ``TokenizersBackend`` declaration.
        self.model._tokenizer = _FalconTokenizer(export_dir)
        safe_attention = config.flex_attention_safe
        if not safe_attention and self.device.type == "cuda":
            # Auto-enable the conservative path below Hopper-class GPUs.  The
            # upstream default can exceed Ampere's 99 KiB shared-memory limit.
            try:
                safe_attention = torch.cuda.get_device_capability(self.device)[0] < 9
            except (RuntimeError, AssertionError):
                safe_attention = False
        _configure_falcon_attention(self.model, safe=safe_attention)
        self.model = self.model.to(self.device).eval()
        logger.info("Falcon model ready in %.2fs", time.perf_counter() - started)

    @torch.inference_mode()
    def generate(
        self,
        image_rgb: np.ndarray | Image.Image,
        query: str,
        *,
        task: Literal["detection", "segmentation"],
    ) -> list[dict[str, Any]]:
        """Run one query and return Falcon's raw prediction dictionaries."""

        if not query.strip():
            raise ValueError("query must be non-empty")
        image = _as_rgb_image(image_rgb)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "min_dimension": self.config.min_image_size,
            "max_dimension": self.config.max_image_size,
            "compile": self.config.compile,
            "segm_threshold": self.config.segmentation_threshold,
        }
        # The full Falcon model defaults to segmentation and its current
        # exported ``generate`` signature has no ``task`` parameter.  The
        # 300M export accepts it, but also defaults to detection.  Only pass
        # the keyword when the loaded custom-code model advertises it.
        try:
            if "task" in inspect.signature(self.model.generate).parameters:
                kwargs["task"] = task
        except (TypeError, ValueError):  # pragma: no cover - unusual remote code
            pass
        output = self.model.generate(image, query, **kwargs)
        if not isinstance(output, list) or len(output) != 1:
            raise RuntimeError(f"Unexpected Falcon output shape: {type(output)!r}")
        predictions = output[0]
        if not isinstance(predictions, list):
            raise RuntimeError(f"Unexpected Falcon prediction list: {type(predictions)!r}")
        return [prediction for prediction in predictions if isinstance(prediction, dict)]

    @staticmethod
    def box(prediction: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
        return _prediction_box(prediction, image_size)

    @staticmethod
    def mask(prediction: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
        return _decode_rle_mask(prediction["mask_rle"], image_size)
