"""Point cloud generation module.

Goal:
- Back-project depth into a 3D point cloud using camera intrinsics K.
- Optionally attach per-point RGB colors (from the input image).
- Optionally attach per-point mask id (from masks.npz).
- Optionally map representative mask points to point indices (from positions.npz).

Inputs:
- depth.npy: float32 depth map, shape [H, W]
- (optional) image: RGB image path (used for colors)
- (optional) masks.npz: contains masks [N,Hm,Wm]
- (optional) positions.npz: contains rep_uvs [N,2] and valids [N]
- intrinsics: either provide fx/fy/cx/cy, or a K file (npy/json)

Outputs (default names under output_dir):
- pointcloud.npz: points_xyz [P,3], colors [P,3] (optional), pixel_uv [P,2], mask_ids [P]
- pointcloud.ply: (optional) Open3D point cloud for easy viewing

Conventions:
- Pixel coordinates: u is x (column), v is y (row).
- Camera coordinates (OpenCV-like):
  - Z forward, X right, Y down.
  - X = (u - cx) / fx * Z
  - Y = (v - cy) / fy * Z

Notes:
- If depth resolution differs from the chosen target size, depth is resized.
- If masks resolution differs from target size, masks are resized with nearest.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from gda.datatypes import PointCloudResult


def load_k_from_json(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "K" in data:
            k = np.asarray(data["K"], dtype=np.float32)
            if k.shape != (3, 3):
                raise ValueError(f"K must be 3x3, got {k.shape}")
            return k
        required = ["fx", "fy", "cx", "cy"]
        if all(k in data for k in required):
            fx, fy, cx, cy = (
                float(data["fx"]),
                float(data["fy"]),
                float(data["cx"]),
                float(data["cy"]),
            )
            return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError("Invalid intrinsics json: expected {K:[[...]]} or {fx,fy,cx,cy}.")


def load_k(k_path: Path) -> np.ndarray:
    if not k_path.exists():
        raise FileNotFoundError(f"K file not found: {k_path}")
    if k_path.suffix.lower() in {".npy"}:
        k = np.load(k_path).astype(np.float32)
        if k.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {k.shape}")
        return k
    if k_path.suffix.lower() in {".json"}:
        return load_k_from_json(k_path)

    # Backward-compatible aliases
    _load_k_from_json = load_k_from_json
    _load_k = load_k
    raise ValueError(f"Unsupported K file extension: {k_path.suffix} (use .npy or .json)")


class PointCloudGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_depth: float = 1e-6
    max_depth: float | None = None

    use_masks_union_only: bool = True
    include_colors: bool = True

    save_ply: bool = True


class PointCloudGenerator:
    def __init__(self, config: PointCloudGenerationConfig):
        self.config = config

    def _resize_depth(self, depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if depth.shape[0] == target_h and depth.shape[1] == target_w:
            return depth
        return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _resize_masks(self, masks: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if masks.shape[1] == target_h and masks.shape[2] == target_w:
            return masks
        out = np.empty((masks.shape[0], target_h, target_w), dtype=np.bool_)
        for i in range(masks.shape[0]):
            m = masks[i].astype(np.uint8) * 255
            m_rs = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            out[i] = m_rs > 0
        return out

    def generate(
        self,
        depth: np.ndarray,
        k: np.ndarray,
        image_rgb: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        rep_uvs: np.ndarray | None = None,
        rep_valids: np.ndarray | None = None,
    ) -> PointCloudResult:
        """Generate point cloud arrays."""

        fx = float(k[0, 0])
        fy = float(k[1, 1])
        cx = float(k[0, 2])
        cy = float(k[1, 2])
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Invalid intrinsics fx/fy: fx={fx}, fy={fy}")

        depth = np.asarray(depth, dtype=np.float32)
        h, w = int(depth.shape[0]), int(depth.shape[1])

        min_d = float(self.config.min_depth)
        max_d = None if self.config.max_depth is None else float(self.config.max_depth)
        valid = np.isfinite(depth) & (depth > min_d)
        if max_d is not None:
            valid &= depth <= max_d

        mask_ids: np.ndarray | None = None
        if masks is not None:
            masks_bool = masks.astype(bool)
            if masks_bool.shape[1] != h or masks_bool.shape[2] != w:
                raise ValueError("masks must already match depth size")
            union = np.any(masks_bool, axis=0)
            if self.config.use_masks_union_only:
                valid &= union

            # Assign each valid pixel a mask id (first mask wins). Background -> -1.
            # If a pixel belongs to multiple masks, pick the smallest mask index.
            mask_ids = np.full((h, w), -1, dtype=np.int32)
            for mi in range(masks_bool.shape[0]):
                m = masks_bool[mi]
                mask_ids[(mask_ids < 0) & m] = int(mi)

        vv, uu = np.nonzero(valid)
        z = depth[vv, uu].astype(np.float32)
        u = uu.astype(np.float32)
        v = vv.astype(np.float32)

        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        points = np.stack([x, y, z], axis=1).astype(np.float32)

        out = PointCloudResult(
            points_xyz=points,
            pixel_uv=np.stack([uu.astype(np.int32), vv.astype(np.int32)], axis=1),
            K=k.astype(np.float32),
            image_size=np.asarray([h, w], dtype=np.int32),
        )

        mask_ids_flat: np.ndarray | None = None
        if mask_ids is not None:
            mask_ids_flat = mask_ids[vv, uu].astype(np.int32)

        colors: np.ndarray | None = None
        if image_rgb is not None and self.config.include_colors:
            if image_rgb.shape[0] != h or image_rgb.shape[1] != w:
                raise ValueError("image must already match depth size")
            colors = image_rgb[vv, uu].astype(np.uint8)

        rep_point_indices: np.ndarray | None = None
        if rep_uvs is not None:
            rep_uvs = np.asarray(rep_uvs)
            rep_point_indices = np.full((rep_uvs.shape[0],), -1, dtype=np.int32)

            # Map pixel -> point index using linear index.
            lin = uu.astype(np.int64) + vv.astype(np.int64) * np.int64(w)
            mapping = {int(lin[i]): int(i) for i in range(lin.shape[0])}

            if rep_valids is None:
                rep_valids = np.ones((rep_uvs.shape[0],), dtype=np.bool_)
            rep_valids = np.asarray(rep_valids).astype(bool)

            for i in range(rep_uvs.shape[0]):
                if not rep_valids[i]:
                    continue
                u0 = int(rep_uvs[i, 0])
                v0 = int(rep_uvs[i, 1])
                if not (0 <= u0 < w and 0 <= v0 < h):
                    continue
                key = u0 + v0 * w
                if key in mapping:
                    rep_point_indices[i] = mapping[key]

        return PointCloudResult(
            points_xyz=out.points_xyz,
            pixel_uv=out.pixel_uv,
            K=out.K,
            image_size=out.image_size,
            mask_ids=mask_ids_flat,
            colors=colors,
            rep_point_indices=rep_point_indices,
        )


class PointCloudGenerationCLI(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, extra="ignore")

    depth_npy: Path
    output_dir: Path | None = None

    # optional inputs
    image: Path | None = None
    masks_npz: Path | None = None
    positions_npz: Path | None = None

    # intrinsics
    k_file: Path | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None

    # filters & behavior
    min_depth: float = 1e-6
    max_depth: float | None = None
    use_masks_union_only: bool = True
    include_colors: bool = True
    save_ply: bool = True


def _resolve_output_dir(input_path: Path, output_dir: Path | None) -> Path:
    return output_dir if output_dir is not None else input_path.parent


def _make_k(args: PointCloudGenerationCLI, target_h: int, target_w: int) -> np.ndarray:
    if args.k_file is not None:
        k = load_k(args.k_file)
        return k
    if None in (args.fx, args.fy, args.cx, args.cy):
        raise ValueError("Provide either --k_file or all of --fx --fy --cx --cy")
    fx, fy, cx, cy = float(args.fx), float(args.fy), float(args.cx), float(args.cy)
    # If user provides normalized intrinsics by mistake, this will be wrong.
    # We assume pixel units.
    _ = (target_h, target_w)
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _maybe_import_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Open3D is required for saving .ply. Install the locked Pixi environment "
            "with `pixi install --locked`."
        ) from e


def save_pointcloud_ply(path: Path, pc: PointCloudResult | dict) -> None:
    """Save a point cloud dict (from PointCloudGenerator.generate) to a .ply file."""

    if isinstance(pc, PointCloudResult):
        points_xyz = pc.points_xyz
        colors_arr = pc.colors
    else:
        if "points_xyz" not in pc:
            raise KeyError("pc missing 'points_xyz'")
        points_xyz = pc["points_xyz"]
        colors_arr = pc.get("colors")

    if colors_arr is not None:
        colors = colors_arr.astype(np.float32) / 255.0
    else:
        colors = None

    o3d = _maybe_import_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


def main() -> None:
    args = PointCloudGenerationCLI()

    if not args.depth_npy.exists():
        raise FileNotFoundError(f"depth_npy not found: {args.depth_npy}")

    depth = np.load(args.depth_npy).astype(np.float32)

    # Decide target size and resize depth/masks/image as needed.
    target_h, target_w = int(depth.shape[0]), int(depth.shape[1])

    image_rgb: np.ndarray | None = None
    if args.image is not None:
        if not args.image.exists():
            raise FileNotFoundError(f"image not found: {args.image}")
        img = Image.open(args.image).convert("RGB")
        image_rgb = np.array(img, dtype=np.uint8)
        target_h, target_w = int(image_rgb.shape[0]), int(image_rgb.shape[1])

    masks: np.ndarray | None = None
    masks_meta: dict | None = None
    if args.masks_npz is not None:
        if not args.masks_npz.exists():
            raise FileNotFoundError(f"masks_npz not found: {args.masks_npz}")
        masks_data = np.load(args.masks_npz, allow_pickle=True)
        if "masks" not in masks_data:
            raise KeyError(f"masks_npz missing 'masks' array: {args.masks_npz}")
        masks = masks_data["masks"].astype(bool)
        masks_meta = {
            "prompts": masks_data["prompts"].tolist() if "prompts" in masks_data else None,
            "prompt_ids": masks_data["prompt_ids"].tolist()
            if "prompt_ids" in masks_data
            else None,
        }
        target_h, target_w = int(masks.shape[1]), int(masks.shape[2])

    # Resize depth to target.
    if depth.shape[0] != target_h or depth.shape[1] != target_w:
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Resize image to target (if provided).
    if image_rgb is not None and (
        image_rgb.shape[0] != target_h or image_rgb.shape[1] != target_w
    ):
        image_rgb = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Resize masks to target (if provided).
    if masks is not None and (masks.shape[1] != target_h or masks.shape[2] != target_w):
        gen_tmp = PointCloudGenerator(PointCloudGenerationConfig())
        masks = gen_tmp._resize_masks(masks, target_h=target_h, target_w=target_w)

    rep_uvs: np.ndarray | None = None
    rep_valids: np.ndarray | None = None
    if args.positions_npz is not None:
        if not args.positions_npz.exists():
            raise FileNotFoundError(f"positions_npz not found: {args.positions_npz}")
        pos = np.load(args.positions_npz, allow_pickle=True)
        rep_uvs = pos["rep_uvs"]
        rep_valids = pos["valids"] if "valids" in pos else None

    k = _make_k(args, target_h=target_h, target_w=target_w)

    out_dir = _resolve_output_dir(args.depth_npy, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = PointCloudGenerator(
        PointCloudGenerationConfig(
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            use_masks_union_only=args.use_masks_union_only,
            include_colors=args.include_colors,
            save_ply=args.save_ply,
        )
    )

    pc = gen.generate(
        depth=depth,
        k=k,
        image_rgb=image_rgb,
        masks=masks,
        rep_uvs=rep_uvs,
        rep_valids=rep_valids,
    )

    # Save NPZ
    npz_kwargs = pc.to_npz_dict()
    if masks_meta is not None:
        npz_kwargs["masks_meta"] = json.dumps(masks_meta, ensure_ascii=False)

    np.savez_compressed(out_dir / "pointcloud.npz", **npz_kwargs)

    # Save PLY (optional)
    if args.save_ply:
        save_pointcloud_ply(out_dir / "pointcloud.ply", pc)

    print(f"[OK] point cloud saved under: {out_dir}")


if __name__ == "__main__":
    main()
