"""Mask position representation module.

Goal:
- Given a depth map and N binary masks, compute a representative pixel per mask.

Method (default):
- For each mask, collect valid depth pixels.
- Compute median depth within the mask.
- Pick the pixel whose depth is closest to the median.

Inputs:
- depth.npy: float32 depth map, shape [H, W]
- masks.npz: produced by gda/modules/object_segmentation or gda/gda
  - expects an array named "masks" with shape [N, Hm, Wm]

Outputs (default names under output_dir):
- positions.npz: rep_uvs [N,2], rep_depths [N], valids [N], plus metadata
- positions.json: human-readable per-mask summary

Notes:
- If depth resolution differs from mask resolution, depth is resized to mask size.
- This module follows the project constraints:
  - module class init takes exactly one `config` argument (pydantic BaseModel)
  - CLI args parsed via pydantic_settings
  - output_dir defaults to the input file directory
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from gda.datatypes import PositionsResult


class PositionRepresentationConfig(BaseModel):
    """Configuration for position representation."""

    model_config = ConfigDict(extra="forbid")

    min_depth: float = 1e-6
    max_depth: float | None = None


class MaskPositionRepresentor:
    """Compute representative (u, v, depth) per mask."""

    def __init__(self, config: PositionRepresentationConfig):
        self.config = config

    def _resize_depth_to_masks(self, depth: np.ndarray, mask_h: int, mask_w: int) -> np.ndarray:
        if depth.shape[0] == mask_h and depth.shape[1] == mask_w:
            return depth
        return cv2.resize(depth, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)

    def compute(
        self,
        depth: np.ndarray,
        masks: np.ndarray,
        prompts: list[str] | None = None,
        prompt_ids: list[int] | None = None,
    ) -> PositionsResult:
        """Compute representative points.

        Args:
            depth: Depth map, shape [Hd, Wd].
            masks: Boolean masks, shape [N, Hm, Wm].
            prompts: Optional list of prompt strings (length N or None).
            prompt_ids: Optional list of prompt ids (length N or None).

        Returns:
            (meta, rep_uvs, rep_depths, valids)
        """

        if masks.ndim != 3:
            raise ValueError(f"masks must have shape [N,H,W], got {masks.shape}")

        masks_bool = masks.astype(bool)
        n, mh, mw = masks_bool.shape
        depth_rs = self._resize_depth_to_masks(np.asarray(depth, dtype=np.float32), mh, mw)

        rep_uvs = np.full((n, 2), -1, dtype=np.int32)
        rep_depths = np.full((n,), np.nan, dtype=np.float32)
        valids = np.zeros((n,), dtype=np.bool_)
        num_valid = np.zeros((n,), dtype=np.int32)

        min_d = float(self.config.min_depth)
        max_d = None if self.config.max_depth is None else float(self.config.max_depth)

        for i in range(n):
            mask = masks_bool[i]
            d = depth_rs
            valid = mask & np.isfinite(d) & (d > min_d)
            if max_d is not None:
                valid &= d <= max_d

            cnt = int(valid.sum())
            num_valid[i] = cnt
            if cnt == 0:
                continue

            vals = d[valid]
            med = float(np.median(vals))

            # Find pixel with depth closest to median.
            vv, uu = np.nonzero(valid)
            diffs = np.abs(d[vv, uu].astype(np.float32) - med)
            j = int(np.argmin(diffs))

            u = int(uu[j])
            v = int(vv[j])
            rep_uvs[i] = (u, v)
            rep_depths[i] = float(d[v, u])
            valids[i] = True

        meta: dict = {
            "method": "median_depth_nearest_pixel",
            "mask_size": [int(mh), int(mw)],
            "depth_size": [int(depth.shape[0]), int(depth.shape[1])],
            "min_depth": min_d,
            "max_depth": max_d,
            "num_masks": int(n),
            "num_valid": num_valid.tolist(),
        }

        # Optional prompt metadata (best-effort)
        if prompts is not None:
            meta["prompts"] = prompts
        if prompt_ids is not None:
            meta["prompt_ids"] = prompt_ids

        return PositionsResult(meta=meta, rep_uvs=rep_uvs, rep_depths=rep_depths, valids=valids)


class PositionRepresentationCLI(BaseSettings):
    """CLI arguments for mask position representation."""

    model_config = SettingsConfigDict(cli_parse_args=True, extra="ignore")

    depth_npy: Path
    masks_npz: Path
    output_dir: Path | None = None

    min_depth: float = 1e-6
    max_depth: float | None = None


def _resolve_output_dir(input_path: Path, output_dir: Path | None) -> Path:
    return output_dir if output_dir is not None else input_path.parent


def main() -> None:
    args = PositionRepresentationCLI()

    if not args.depth_npy.exists():
        raise FileNotFoundError(f"depth_npy not found: {args.depth_npy}")
    if not args.masks_npz.exists():
        raise FileNotFoundError(f"masks_npz not found: {args.masks_npz}")

    out_dir = _resolve_output_dir(args.masks_npz, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    depth = np.load(args.depth_npy).astype(np.float32)
    masks_data = np.load(args.masks_npz, allow_pickle=True)
    if "masks" not in masks_data:
        raise KeyError(f"masks_npz missing 'masks' array: {args.masks_npz}")
    masks = masks_data["masks"]

    prompts: list[str] | None = None
    if "prompts" in masks_data:
        try:
            prompts_arr = masks_data["prompts"]
            prompts = [str(x) for x in prompts_arr.tolist()]
        except Exception:
            prompts = None

    prompt_ids: list[int] | None = None
    if "prompt_ids" in masks_data:
        try:
            prompt_ids_arr = masks_data["prompt_ids"]
            prompt_ids = [int(x) for x in prompt_ids_arr.tolist()]
        except Exception:
            prompt_ids = None

    representor = MaskPositionRepresentor(
        PositionRepresentationConfig(min_depth=args.min_depth, max_depth=args.max_depth)
    )
    result = representor.compute(
        depth=depth,
        masks=masks,
        prompts=prompts,
        prompt_ids=prompt_ids,
    )

    np.savez_compressed(
        out_dir / "positions.npz",
        rep_uvs=result.rep_uvs,
        rep_depths=result.rep_depths,
        valids=result.valids,
        meta=json.dumps(result.meta, ensure_ascii=False),
    )

    per_mask: list[dict] = []
    for i in range(int(result.meta["num_masks"])):
        item = {
            "mask_index": i,
            "valid": bool(result.valids[i]),
            "rep_uv": result.rep_uvs[i].tolist(),
            "rep_depth": float(result.rep_depths[i]) if np.isfinite(result.rep_depths[i]) else None,
            "num_valid": int(result.meta["num_valid"][i]),
        }
        if prompt_ids is not None and i < len(prompt_ids):
            item["prompt_id"] = int(prompt_ids[i])
        if prompts is not None and i < len(prompts):
            item["prompt"] = str(prompts[i])
        per_mask.append(item)

    (out_dir / "positions.json").write_text(
        json.dumps({"meta": result.meta, "items": per_mask}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] positions saved under: {out_dir}")


if __name__ == "__main__":
    main()
