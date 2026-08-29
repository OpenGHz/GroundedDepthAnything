---
type: paper
node_id: paper:back2020_segmenting_unseen_industrial
title: "Segmenting Unseen Industrial Components in a Heavy Clutter Using RGB-D Fusion and Synthetic Data"
authors: ["Seunghyeok Back", "Jongwon Kim", "Raeyoung Kang", "Seungjun Choi", "Kyoobin Lee"]
year: 2020
venue: "ICIP 2020"
external_ids:
  arxiv: "2002.03501"
  doi: "10.1109/ICIP40778.2020.9190804"
  s2: null
tags: ["rgb-d", "instance-segmentation", "synthetic-to-real", "industrial", "geometry"]
added: 2026-08-29T05:58:32Z
---

# Segmenting Unseen Industrial Components in a Heavy Clutter Using RGB-D Fusion and Synthetic Data

## One-line thesis
Synthetic RGB-D with randomized geometry, texture, and depth noise transfers to unseen industrial objects, but the task is category-agnostic instance segmentation.

## Problem / Gap
Textureless reflective industrial parts in heavy clutter are hard to segment from RGB alone. The paper tests whether randomized synthetic RGB-D and confidence-aware fusion transfer to unseen real objects.

## Method
V-REP renders 35,000 RGB-D training/validation images with randomized CAD objects, textures, poses, clutter, and simulated depth noise/holes. A two-branch ResNet-50 Mask R-CNN fuses RGB and raw/filled depth with a learned confidence map.

## Key Results
On 100 manually labeled RealSense D415 image pairs, the proposed RGB-D instance model reaches **69.0 AP50, 57.7 AP, and 66.1 AR**. Training uses synthetic data only; the RGB branch is ImageNet-pretrained.

## Assumptions
The task is category-agnostic instance segmentation of unseen industrial components, with objects captured in a bin-picking configuration. Real images share the randomized camera/object-layout range used in simulation.

## Limitations / Failure Modes
This is instance, not multiclass semantic, segmentation. The benchmark is narrow and uses ImageNet initialization; real objects/cameras are deliberately similar to the simulation range. It therefore cannot establish broad indoor geometric understanding.

## Reusable Ingredients
Randomized textures, explicit depth corruption, confidence-weighted fusion, and object/CAD split controls are useful ingredients for a geometry-focused RGB-D transfer benchmark.

## Open Questions
Would the confidence module still help with unseen sensors, layouts, and semantic class taxonomies? Can category-agnostic instance transfer be converted into a fixed multiclass semantic protocol without relying on object-set overlap?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Instance-segmentation near-neighbor only ([arXiv:2002.03501](https://arxiv.org/abs/2002.03501); [DOI:10.1109/ICIP40778.2020.9190804](https://doi.org/10.1109/ICIP40778.2020.9190804)). Keep it as supporting evidence for G4/G5 (depth noise and confidence), not as a direct semantic-segmentation precedent.
