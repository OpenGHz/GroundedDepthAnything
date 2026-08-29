# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Open Gaps
# Gap Map

_Field gaps with stable IDs. Updated 2026-08-29._

## Priority Summary

| ID | Status | Short description |
|---|---|---|
| **G1** | unresolved | No unambiguous literal synthetic-only, direct RGB-D, multiclass 2D, target-free real zero-shot paper confirmed. |
| **G2** | unresolved | External image pretraining and learned depth estimators are rarely isolated from geometric learning. |
| **G3** | unresolved | Broad indoor, multiclass, target-free RGB-D evaluation across unseen real sensors is missing. |
| **G4** | unresolved | Native sensor noise, holes, scale bias, and missing-data transfer are not systematically solved for direct RGB-D. |
| **G5** | unresolved | Depth-edge, fixed-layout, and camera shortcuts are rarely separated from reusable geometry. |
| **G6** | unresolved | Matched-capacity random-init RGB-only/depth-only/RGB-D controls are rare. |

## G1 — Literal synthetic-only RGB-D multiclass zero-shot evidence

No unambiguous paper was confirmed that combines direct single-frame RGB-D input, 2D multiclass semantic segmentation, no real-image initialization, no target-domain adaptation/selection, and quantitative real zero-shot evaluation. DSSS is the closest mai
## Key Papers (13 total)
- [paper:back2020_segmenting_unseen_industrial] Segmenting Unseen Industrial Components in a Heavy Clutter Using RGB-D Fusion and Synthetic Data: Synthetic RGB-D with randomized geometry, texture, and depth noise transfers to unseen industrial objects, but the task is category-agnostic instance
- [paper:digumarti2019_approach_semantic_segmentation] An Approach for Semantic Segmentation of Tree-like Vegetation: Late-fused RGB and HHA networks segment tree parts from synthetic single-frame RGB-D and show qualitative real-world transfer, but lack quantitative r
- [paper:handa2015_scenenet_understanding_real] SceneNet: Understanding Real World Indoor Scenes With Synthetic Data: Large-scale synthetic depth rendering can transfer functional indoor scene labels to real NYUv2 and SUN RGB-D, even without RGB at inference.
- [paper:joukovsky2020_multimodal_deep_network] Multi-modal deep network for RGB-D segmentation of clothes: A synthetic RGB-D clothing dataset and multimodal encoder-decoder achieve strong synthetic segmentation and qualitative Kinect-v2 transfer, but no qua
- [paper:li2020_semantic_segmentation_printed] Semantic Segmentation of a Printed Circuit Board for Component Recognition Based on Depth Images: Handcrafted depth-difference features and a random forest transfer synthetic PCB pixel labels to a small real test set, with test-specific parameter c
- [paper:lim2019_hand_object_segmentation] Hand and Object Segmentation from Depth Image using Fully Convolutional Network: Synthetic depth-only training can transfer to real Kinect-v2 pixel segmentation, showing strong but task-narrow geometric robustness.
- [paper:mccormac2017_scenenet_rgbd_photorealistic] SceneNet RGB-D: 5M Photorealistic Images of Synthetic Indoor Trajectories with Ground Truth: SceneNet RGB-D s
## Recent Relationships (23 total)
  paper:rizzoli2024_sourcefree_domain_adaptation --addresses_gap--> gap:G1
  paper:rizzoli2024_sourcefree_domain_adaptation --addresses_gap--> gap:G6
  paper:watanabe2018_multichannel_semantic_segmentation --addresses_gap--> gap:G3
  paper:watanabe2018_multichannel_semantic_segmentation --addresses_gap--> gap:G6
  paper:handa2015_scenenet_understanding_real --addresses_gap--> gap:G4
  paper:mccormac2017_scenenet_rgbd_photorealistic --addresses_gap--> gap:G1
  paper:mccormac2017_scenenet_rgbd_photorealistic --addresses_gap--> gap:G3
  paper:mccormac2017_scenenet_rgbd_photorealistic --extends--> paper:handa2015_scenenet_understanding_real
  paper:planche2021_physicsbased_differentiable_depth --addresses_gap--> gap:G4
  paper:sharma2016_lowcost_scene_modeling --addresses_gap--> gap:G4
  paper:sharma2016_lowcost_scene_modeling --addresses_gap--> gap:G5
  paper:lim2019_hand_object_segmentatio
