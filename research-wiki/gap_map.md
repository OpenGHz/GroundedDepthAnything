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

No unambiguous paper was confirmed that combines direct single-frame RGB-D input, 2D multiclass semantic segmentation, no real-image initialization, no target-domain adaptation/selection, and quantitative real zero-shot evaluation. DSSS is the closest main method but uses ImageNet initialization and estimated GTA5 depth; MISFIT's source-only row has undisclosed initialization.

**Relevant nodes:** `paper:wei2025_depthsensitive_soft_suppression`, `paper:rizzoli2024_sourcefree_domain_adaptation`, `paper:joukovsky2020_multimodal_deep_network`, `paper:digumarti2019_approach_semantic_segmentation`

## G2 — Pretraining and depth-estimator confounds

ImageNet/Pascal pretraining and learned depth estimators can contribute real-image priors. Existing reports rarely isolate random initialization, external pretraining, and renderer-native depth in the same protocol.

**Relevant nodes:** `paper:wei2025_depthsensitive_soft_suppression`, `paper:rizzoli2024_sourcefree_domain_adaptation`, `paper:watanabe2018_multichannel_semantic_segmentation`, `paper:vu2019_dada_depthaware_domain`

## G3 — Indoor target-free RGB-D benchmark

Modern quantitative target-free RGB-D evidence is concentrated in road scenes or narrow objects. A broad indoor multiclass benchmark with fixed, unseen real sensors and no target-domain selection is missing.

**Relevant nodes:** `paper:handa2015_scenenet_understanding_real`, `paper:mccormac2017_scenenet_rgbd_photorealistic`, `paper:watanabe2018_multichannel_semantic_segmentation`

## G4 — Native sensor depth and real-noise transfer

Synthetic depth often lacks structured-light/stereo noise, holes, scale bias, and missing-data patterns. DDS, SceneNet, and depth-only studies show the issue matters, but do not provide a complete direct RGB-D multiclass solution.

**Relevant nodes:** `paper:planche2021_physicsbased_differentiable_depth`, `paper:handa2015_scenenet_understanding_real`, `paper:lim2019_hand_object_segmentation`, `paper:sharma2016_lowcost_scene_modeling`, `paper:back2020_segmenting_unseen_industrial`

## G5 — Geometry versus boundary/layout shortcuts

Depth gains may arise from local discontinuities, fixed camera geometry, class-layout priors, or RGB-depth alignment rather than reusable spatial reasoning. Intervention-based diagnostics are uncommon.

**Relevant nodes:** `paper:wei2025_depthsensitive_soft_suppression`, `paper:li2020_semantic_segmentation_printed`, `paper:sharma2016_lowcost_scene_modeling`, `paper:back2020_segmenting_unseen_industrial`

## G6 — Fair RGB/depth/RGB-D controls

Many papers change architecture, initialization, sensor preprocessing, and data volume together. Matched-capacity random-init RGB-only, depth-only, and RGB-D controls with held-out CAD/layout/sensor splits are still rare.

**Relevant nodes:** `paper:rizzoli2024_sourcefree_domain_adaptation`, `paper:watanabe2018_multichannel_semantic_segmentation`, `paper:handa2015_scenenet_understanding_real`, `paper:mccormac2017_scenenet_rgbd_photorealistic`, `paper:lim2019_hand_object_segmentation`
