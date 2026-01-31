from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    image_size: tuple[int, int]  # (H, W)
    prompts: list[str]
    boxes_xyxy: np.ndarray  # float32, [N,4]
    scores: np.ndarray  # float32, [N]
    prompt_ids: np.ndarray  # int32, [N]
    labels: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "prompts": list(self.prompts),
            "boxes_xyxy": np.asarray(self.boxes_xyxy, dtype=np.float32).tolist(),
            "scores": np.asarray(self.scores, dtype=np.float32).tolist(),
            "prompt_ids": np.asarray(self.prompt_ids, dtype=np.int32).tolist(),
            "labels": list(self.labels),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "DetectionResult":
        h, w = data.get("image_size", [0, 0])
        prompts = [str(x) for x in (data.get("prompts") or [])]
        boxes = np.asarray(data.get("boxes_xyxy") or [], dtype=np.float32)
        scores = np.asarray(data.get("scores") or [], dtype=np.float32)
        prompt_ids = np.asarray(data.get("prompt_ids") or [], dtype=np.int32)
        labels = [str(x) for x in (data.get("labels") or [])]
        if boxes.size == 0:
            boxes = boxes.reshape((0, 4)).astype(np.float32)
        return cls(
            image_size=(int(h), int(w)),
            prompts=prompts,
            boxes_xyxy=boxes,
            scores=scores.reshape((-1,)).astype(np.float32),
            prompt_ids=prompt_ids.reshape((-1,)).astype(np.int32),
            labels=labels,
        )


@dataclass(frozen=True)
class SegmentationResult:
    backend: str
    image_size: tuple[int, int]  # (H, W)
    prompts: list[str]
    boxes_xyxy: np.ndarray  # float32, [N,4]
    prompt_ids: np.ndarray  # int32, [N]
    scores: np.ndarray  # float32, [N]
    masks: np.ndarray  # bool, [N,H,W]

    def meta_json_dict(self) -> dict[str, Any]:
        return {
            "backend": str(self.backend),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "prompts": list(self.prompts),
            "boxes_xyxy": np.asarray(self.boxes_xyxy, dtype=np.float32).tolist(),
            "prompt_ids": np.asarray(self.prompt_ids, dtype=np.int32).tolist(),
            "scores": np.asarray(self.scores, dtype=np.float32).tolist(),
            "_masks_shape": list(np.asarray(self.masks).shape),
        }


@dataclass(frozen=True)
class PositionsResult:
    meta: dict[str, Any]
    rep_uvs: np.ndarray  # int32, [N,2]
    rep_depths: np.ndarray  # float32, [N]
    valids: np.ndarray  # bool, [N]


@dataclass(frozen=True)
class DepthAndSegResult:
    depth: np.ndarray  # float32, [H,W]
    det: DetectionResult
    seg: SegmentationResult


@dataclass(frozen=True)
class ImageToPositionsResult:
    depth: np.ndarray
    det: DetectionResult
    seg: SegmentationResult
    positions: PositionsResult


@dataclass(frozen=True)
class PointCloudResult:
    points_xyz: np.ndarray  # float32, [P,3]
    pixel_uv: np.ndarray  # int32, [P,2]
    K: np.ndarray  # float32, [3,3]
    image_size: np.ndarray  # int32, [2] (H,W)
    mask_ids: np.ndarray | None = None  # int32, [P]
    colors: np.ndarray | None = None  # uint8, [P,3]
    rep_point_indices: np.ndarray | None = None  # int32, [N]

    def to_npz_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "points_xyz": np.asarray(self.points_xyz, dtype=np.float32),
            "pixel_uv": np.asarray(self.pixel_uv, dtype=np.int32),
            "K": np.asarray(self.K, dtype=np.float32),
            "image_size": np.asarray(self.image_size, dtype=np.int32),
        }
        if self.mask_ids is not None:
            out["mask_ids"] = np.asarray(self.mask_ids, dtype=np.int32)
        if self.colors is not None:
            out["colors"] = np.asarray(self.colors, dtype=np.uint8)
        if self.rep_point_indices is not None:
            out["rep_point_indices"] = np.asarray(self.rep_point_indices, dtype=np.int32)
        return out


def _as_list_str(items: Iterable[Any]) -> list[str]:
    return [str(x) for x in items]
