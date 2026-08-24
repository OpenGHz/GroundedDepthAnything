"""Point cloud visualization module (Open3D).

Inputs:
- pointcloud.npz from gda/modules/pointcloud_generation
  - expects points_xyz [P,3]
  - optional colors [P,3] uint8
  - optional rep_point_indices [N]

Behavior:
- Opens an interactive Open3D window.
- Highlights representative points (if provided) by placing small spheres.

Notes:
- This is a GUI tool; on headless machines you may need X11 forwarding.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


def _maybe_import_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Open3D is required for visualization. Install the locked Pixi environment "
            "with `pixi install --locked`."
        ) from e


class PointCloudVisualizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point_size: float = 2.0
    rep_sphere_radius: float = 0.02
    show_axes: bool = True
    timeout_sec: float | None = None


class PointCloudVisualizer:
    def __init__(self, config: PointCloudVisualizationConfig):
        self.config = config

    def visualize(
        self,
        points_xyz: np.ndarray,
        colors: np.ndarray | None,
        rep_point_indices: np.ndarray | None,
    ) -> None:
        o3d = _maybe_import_open3d()

        # Fail fast on common headless setups.
        if os.name != "nt":
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                raise RuntimeError(
                    "Open3D visualization requires a GUI display, but "
                    "DISPLAY/WAYLAND_DISPLAY is not set. "
                    "Run without visualization, or use X11 forwarding / a desktop session."
                )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))

        if colors is not None:
            c = np.asarray(colors, dtype=np.float32)
            if c.max() > 1.0:
                c = c / 255.0
            pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float64))

        geoms: list = [pcd]

        if self.config.show_axes:
            geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

        if rep_point_indices is not None:
            idx = np.asarray(rep_point_indices, dtype=np.int32)
            for i in range(idx.shape[0]):
                pi = int(idx[i])
                if pi < 0 or pi >= points_xyz.shape[0]:
                    continue
                center = np.asarray(points_xyz[pi], dtype=np.float64)
                sph = o3d.geometry.TriangleMesh.create_sphere(
                    radius=float(self.config.rep_sphere_radius)
                )
                sph.translate(center)
                sph.paint_uniform_color([1.0, 0.0, 0.0])
                geoms.append(sph)

        # Use Visualizer to set point size.
        vis = o3d.visualization.Visualizer()
        ok = vis.create_window(window_name="GDA PointCloud", width=1280, height=720)
        if not ok:
            vis.destroy_window()
            raise RuntimeError(
                "Open3D failed to create a window. This is commonly caused by "
                "missing/invalid display setup "
                "(e.g., no X server, bad DISPLAY, or headless environment)."
            )
        for g in geoms:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        if opt is not None:
            opt.point_size = float(self.config.point_size)
        if self.config.timeout_sec is None:
            vis.run()
        else:
            end_t = time.time() + float(self.config.timeout_sec)
            while time.time() < end_t:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01)
        vis.destroy_window()


class PointCloudVisualizationCLI(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, extra="ignore")

    pointcloud_npz: Path

    point_size: float = 2.0
    rep_sphere_radius: float = 0.02
    show_axes: bool = True
    timeout_sec: float | None = None


def main() -> None:
    args = PointCloudVisualizationCLI()
    if not args.pointcloud_npz.exists():
        raise FileNotFoundError(f"pointcloud_npz not found: {args.pointcloud_npz}")

    data = np.load(args.pointcloud_npz, allow_pickle=True)
    if "points_xyz" not in data:
        raise KeyError(f"pointcloud_npz missing 'points_xyz': {args.pointcloud_npz}")

    points = data["points_xyz"]
    colors = data["colors"] if "colors" in data else None
    rep_idx = data["rep_point_indices"] if "rep_point_indices" in data else None

    vis = PointCloudVisualizer(
        PointCloudVisualizationConfig(
            point_size=args.point_size,
            rep_sphere_radius=args.rep_sphere_radius,
            show_axes=args.show_axes,
            timeout_sec=args.timeout_sec,
        )
    )
    vis.visualize(points_xyz=points, colors=colors, rep_point_indices=rep_idx)


if __name__ == "__main__":
    main()
