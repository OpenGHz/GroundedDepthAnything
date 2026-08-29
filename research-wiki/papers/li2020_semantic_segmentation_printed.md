---
type: paper
node_id: paper:li2020_semantic_segmentation_printed
title: "Semantic Segmentation of a Printed Circuit Board for Component Recognition Based on Depth Images"
authors: ["Dongnian Li", "Changming Li", "Chengjun Chen", "Zhengxu Zhao"]
year: 2020
venue: "Sensors"
external_ids:
  arxiv: null
  doi: "10.3390/s20185318"
  s2: null
tags: ["depth-only", "semantic-segmentation", "industrial", "pcb", "synthetic-to-real", "caveat"]
added: 2026-08-29T05:59:37Z
---

# Semantic Segmentation of a Printed Circuit Board for Component Recognition Based on Depth Images

## One-line thesis
Handcrafted depth-difference features and a random forest transfer synthetic PCB pixel labels to a small real test set, with test-specific parameter choices.

## Problem / Gap
PCB inspection needs pixel-level component labels while remaining robust to illumination and texture. The paper asks whether synthetic depth geometry alone can label components on real PCBs.

## Method
OpenSceneGraph renders synthetic PCB depth/label pairs. Concentric-circle depth-difference features feed a random-forest pixel classifier; predicted labels are evaluated on synthetic and manually labeled real depth images.

## Key Results
Using 200 synthetic training images, the reported pixel accuracy is **98.96%** on 40 synthetic test images and **83.64%** on 10 real test images. The method runs at about 0.9 s/image on the reported hardware.

## Assumptions
The PCB geometry and camera family are tightly constrained. For the real test, the authors narrow the sampled yaw/pitch range and use a different feature modulus (`m=7` versus `m=2` for synthetic tests).

## Limitations / Failure Modes
The real test set has only ten images, and test-specific sampling/feature parameters mean this is not a clean no-tuning zero-shot protocol. It is depth-only and industrially narrow.

## Reusable Ingredients
Depth-neighborhood features and explicit reporting of synthetic-versus-real accuracy can motivate geometry-only controls, provided all parameters are fixed before target evaluation.

## Open Questions
Would a learned RGB-D model transfer across PCB designs and camera poses with one fixed parameter set? How much performance is due to local depth edges versus component shape/context?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Useful cautionary comparator for G4/G5 ([DOI:10.3390/s20185318](https://doi.org/10.3390/s20185318)). It shows why target-blind parameter locking and cross-object/camera tests are necessary.
