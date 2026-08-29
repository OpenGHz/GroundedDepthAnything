---
type: paper
node_id: paper:handa2015_scenenet_understanding_real
title: "SceneNet: Understanding Real World Indoor Scenes With Synthetic Data"
authors: ["Ankur Handa", "Viorica Patraucean", "Vijay Badrinarayanan", "Simon Stent", "Roberto Cipolla"]
year: 2015
venue: "arXiv preprint"
external_ids:
  arxiv: "1511.07041"
  doi: null
  s2: null
tags: ["depth-only", "semantic-segmentation", "indoor", "synthetic-data", "geometry"]
added: 2026-08-29T05:58:32Z
---

# SceneNet: Understanding Real World Indoor Scenes With Synthetic Data

## One-line thesis
Large-scale synthetic depth rendering can transfer functional indoor scene labels to real NYUv2 and SUN RGB-D, even without RGB at inference.

## Problem / Gap
Dense indoor labels are expensive, while depth provides strong functional and geometric cues. The paper tests whether large-scale synthetic depth can replace much of the real annotation burden.

## Method
SceneNet procedurally renders varied indoor scenes and trains a CNN on depth-derived channels (DHA), with sensor-like noise and randomized layouts. The study evaluates synthetic-only depth training and variants that fine-tune on real NYUv2/SUN RGB-D.

## Key Results
On 13-class NYUv2, SceneNet-DHA reports **54.4 global accuracy / 37.1 class accuracy**; on SUN RGB-D it reports **56.9 / 30.2**. These are global/class accuracies, not mIoU. Other rows include real-data fine-tuning and should not be treated as zero-shot.

## Assumptions
Depth-only inference is assumed sufficient for many functional categories. The architecture uses an ImageNet/VGG-era initialization in the reported setup, and the real benchmarks have fixed indoor sensor and class conventions.

## Limitations / Failure Modes
It is not a direct RGB-D semantic-segmentation model because RGB is absent at inference. Metric definitions differ from the mIoU used by newer work, and some strong rows use real fine-tuning. Texture-defined categories such as books, paintings, TVs, and windows remain difficult from depth alone.

## Reusable Ingredients
Synthetic scene randomization, explicit depth-noise modeling, and per-class analysis are directly reusable. The depth-only control is valuable for measuring how much geometry contributes independently of appearance.

## Open Questions
Can the same scene diversity support a randomly initialized RGB-D model without real fine-tuning? How much do sensor noise, metric scale, and object-shape coverage contribute separately?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Foundational geometry-transfer near-neighbor for G4 and G6 ([arXiv:1511.07041](https://arxiv.org/abs/1511.07041)). It provides a strong depth-only reference and a caution to keep accuracy metrics and real fine-tuning variants separated.
