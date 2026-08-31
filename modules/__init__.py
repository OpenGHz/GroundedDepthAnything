"""GDA modular building blocks.

Modules:
- depth_estimation: Depth-Anything-3 based single-image depth
- object_detection: GroundingDINO or Falcon-Perception-300M bbox detection
- object_segmentation: SAM2.1 box-to-mask segmentation
- grounded_segmentation: GroundingDINO+SAM2.1, native SAM3, or Falcon text-to-mask segmentation
- position_representation: per-mask representative pixel (u,v,depth)
- pointcloud_generation: back-project depth (+masks/+rgb) into point cloud
- pointcloud_visualization: visualize generated point cloud with Open3D
"""
