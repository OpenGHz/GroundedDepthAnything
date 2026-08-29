---
type: paper
node_id: paper:wei2025_depthsensitive_soft_suppression
title: "Depth-Sensitive Soft Suppression with RGB-D Inter-Modal Stylization Flow for Domain Generalization Semantic Segmentation"
authors: ["Binbin Wei", "Yuhang Zhang", "Shishun Tian", "Muxin Liao", "Wei Li", "Wenbin Zou"]
year: 2025
venue: "arXiv preprint"
external_ids:
  arxiv: "2505.07050"
  doi: null
  s2: null
tags: ["rgb-d", "semantic-segmentation", "domain-generalization", "synthetic-to-real", "geometry"]
added: 2026-08-29T05:58:32Z
---

# Depth-Sensitive Soft Suppression with RGB-D Inter-Modal Stylization Flow for Domain Generalization Semantic Segmentation

## One-line thesis
A depth-sensitive RGB-D domain-generalization framework improves synthetic-to-real multiclass segmentation without seeing target images.

## Problem / Gap
Synthetic-to-real semantic segmentation suffers from appearance and depth-distribution shift. The paper targets domain generalization, where no target images are available during training, and focuses on depth noise, holes, and sensor-dependent sensitivity.

## Method
DeepLabV3+ is extended with RGB-D fusion. RGB-D inter-modal stylization generates stylized depth for sensitivity detection; class-wise soft spatial sensitivity suppression downweights sensitive depth regions; an RGB-D soft alignment loss preserves depth-specific information while aligning selected features. Experiments use ResNet-50, ShuffleNet-V2, and MobileNet-V2 variants.

## Key Results
For GTA5 source training and 19-class evaluation, the ResNet-50 RGB baseline reaches 34.39 mIoU on Cityscapes and 35.20 on InfraParis. Plain RGB-D reaches 37.99 and 37.39; full DSSS reaches **42.07** and **42.05**, respectively (Tables 1 and 3). The paper also reports GTA5→SYNTHIA 31.80 and GTA5→SELMA 38.28. This is target-free evaluation, not UDA.

## Assumptions
RGB and depth are paired at train and test time; source labels use the Cityscapes 19-class vocabulary; unseen domains have a usable depth map. The implementation initializes all backbones with ImageNet weights. GTA5 depth is generated with Monodepth2, whereas SYNTHIA/SELMA depth is renderer-derived.

## Limitations / Failure Modes
It is not literal from-scratch synthetic-only evidence because of ImageNet initialization. Monodepth2 introduces an external depth-estimation prior and GTA5 depth is not native rendered sensor depth. The benchmark is road-centric, so the result does not establish indoor or cross-sensor generalization. The work is an arXiv preprint and does not isolate all texture/layout/camera shortcuts.

## Reusable Ingredients
The ablation gives a useful RGB versus plain RGB-D versus full-method ladder. Depth sensitivity maps, depth stylization, and modality-specific perturbation are candidate tools for a geometry-focused benchmark. Report both target-free and adapted settings separately.

## Open Questions
Would the gain survive random initialization, held-out depth sensors, native physics-based depth, and unseen CAD/layouts? Does depth help through metric structure or mainly through boundary alignment? How does performance change after depth shuffling, scale distortion, holes, and RGB-depth misregistration?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Closest main-method precedent for synthetic-source RGB-D domain generalization ([arXiv:2505.07050](https://arxiv.org/abs/2505.07050)). It motivates gap G1 (strict synthetic-only RGB-D zero-shot evidence) and G2 (pretraining/depth-estimator confounds), while supplying a strong numerical baseline for future comparisons.
