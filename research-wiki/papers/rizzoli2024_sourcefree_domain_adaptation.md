---
type: paper
node_id: paper:rizzoli2024_sourcefree_domain_adaptation
title: "Source-Free Domain Adaptation for RGB-D Semantic Segmentation with Vision Transformers"
authors: ["Giulia Rizzoli", "Donald Shenaj", "Pietro Zanuttigh"]
year: 2024
venue: "WACVW 2024"
external_ids:
  arxiv: "2305.14269"
  doi: "10.1109/WACVW60836.2024.00070"
  s2: null
tags: ["rgb-d", "semantic-segmentation", "source-only", "source-free-adaptation", "synthetic-to-real"]
added: 2026-08-29T05:58:32Z
---

# Source-Free Domain Adaptation for RGB-D Semantic Segmentation with Vision Transformers

## One-line thesis
MISFIT fuses RGB and depth in a SegFormer and adapts on unlabeled target data; its paper also provides a useful source-only RGB-D baseline.

## Problem / Gap
RGB-D segmentation has both color and depth domain shifts. The main paper asks how to adapt a source-trained multimodal model without retaining source data, but its ablations also expose the source-only synthetic-to-real baseline needed for a target-free comparison.

## Method
MISFIT uses a SegFormer/MiT-B5 encoder with depth injected at input, feature, and output levels. Color/depth style transfer is used before self-training, and a depth-validity/entropy mask weights target pseudo-labels. The source-only ablation disables adaptation modules and trains only on SYNTHIA.

## Key Results
On SYNTHIA→Cityscapes, the paper's Table 3/4 labels the all-modules-disabled row as source-only: RGB is **36.93 mIoU16** and RGB-D with Key Swap is **39.79 mIoU16**. The full target-adapted method reaches 54.5 mIoU, but that number uses unlabeled real Cityscapes and must not be called zero-shot.

## Assumptions
The source-only row uses paired synthetic RGB-depth and evaluates 16 common classes on real Cityscapes with stereo depth. The paper says the architecture is pretrained on source data for 40 epochs/160k iterations, but does not disclose whether MiT-B5 starts from ImageNet or another checkpoint.

## Limitations / Failure Modes
The principal contribution is source-free adaptation, so its headline result violates a no-target-data protocol. Initialization is not verifiable from the paper, supplement, or available author code, making strict synthetic-only status indeterminate. The benchmark is road-scene-specific and has only 16 shared classes in the SYNTHIA setting.

## Reusable Ingredients
The source-only RGB/RGB-D rows provide a clean within-paper control. Key Swap cross-modal attention, modality-specific style transfer, and depth-validity masking are useful design points, provided adapted and non-adapted results remain separated.

## Open Questions
Can the 39.79 mIoU16 result be reproduced from random initialization? What is the contribution of depth when the same SegFormer encoder is trained from scratch? Does the source-only gain persist on indoor scenes and unseen depth sensors?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Important baseline for G1 and G6 ([arXiv:2305.14269](https://arxiv.org/abs/2305.14269); [CVF full text](https://openaccess.thecvf.com/content/WACV2024W/Pretrain/html/Rizzoli_Source-Free_Domain_Adaptation_for_RGB-D_Semantic_Segmentation_With_Vision_Transformers_WACVW_2024_paper.html)): it supplies a paper-contained source-only RGB-D comparison, while its undisclosed initialization marks the exact reproducibility gap that a strict study should close.
