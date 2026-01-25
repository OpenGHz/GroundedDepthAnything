"""GDA modular building blocks.

Modules:
- depth_estimation: Depth-Anything-3 based single-image depth
- object_detection: GroundingDINO bbox detection
- object_segmentation: SAM2 (default) / SAM3 (optional) box-to-mask segmentation
- position_representation: per-mask representative pixel (u,v,depth)
- pointcloud_generation: back-project depth (+masks/+rgb) into point cloud
- pointcloud_visualization: visualize generated point cloud with Open3D
"""
