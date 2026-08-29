---
type: paper
node_id: paper:mccormac2017_scenenet_rgbd_photorealistic
title: "SceneNet RGB-D: 5M Photorealistic Images of Synthetic Indoor Trajectories with Ground Truth"
authors: ["John McCormac", "Ankur Handa", "Stefan Leutenegger", "Andrew J. Davison"]
year: 2017
venue: "ICCV 2017"
external_ids:
  arxiv: "1612.05079"
  doi: "10.1109/ICCV.2017.292"
  s2: null
tags: ["rgb-d", "semantic-segmentation", "indoor", "synthetic-data", "dataset", "geometry"]
added: 2026-08-29T05:59:07Z
---

# SceneNet RGB-D: 5M Photorealistic Images of Synthetic Indoor Trajectories with Ground Truth

## One-line thesis
SceneNet RGB-D supplies 5M synthetic RGB-D trajectories for training from scratch, but its reported indoor segmentation transfer relies on real-data fine-tuning rather than target-free zero-shot evaluation.

## Problem / Gap
Pixel-perfect RGB-D labels and trajectories are scarce in indoor vision. SceneNet RGB-D was created to make large-scale multimodal pretraining and geometric scene understanding feasible.

## Method
The renderer produces 5M RGB-D frames from more than 15K synthetic trajectories, with randomized layouts, lighting, textures, camera motion, and physically simulated object poses. It supplies semantic, instance, depth, pose, flow, and reconstruction ground truth.

## Key Results
The main contribution is the dataset and its scale, not a strict synthetic-to-real zero-shot benchmark. The indoor segmentation experiments reported in the paper use synthetic pretraining followed by real-data fine-tuning in the strongest comparisons; therefore they do not establish target-free transfer.

## Assumptions
Synthetic trajectories and ShapeNet-derived objects are assumed to cover enough indoor variation for pretraining. The dataset is intended to support training from scratch with RGB-D, but downstream evaluation still depends on the chosen real benchmark and protocol.

## Limitations / Failure Modes
No unambiguous target-free real semantic-segmentation result is established by the paper. Rendered textures/materials, object-scale fidelity, and sensor artifacts may differ from real RGB-D. Dataset pretraining should not be conflated with zero-shot deployment.

## Reusable Ingredients
Five-million-frame scale, randomized trajectories, and synchronized semantic/depth ground truth are useful ingredients for a controlled geometry benchmark. The dataset can support matched RGB-only, depth-only, and RGB-D pretraining studies.

## Open Questions
What is the zero-shot performance of a fixed architecture trained only on SceneNet RGB-D, with no real fine-tuning and no external image pretraining? How do held-out CAD models, layouts, and sensors affect transfer?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Dataset-level foundation for G3/G4/G6 ([arXiv:1612.05079](https://arxiv.org/abs/1612.05079)). It exposes the exact missing experiment: direct, target-free evaluation after synthetic-only training rather than synthetic pretraining plus real fine-tuning.
