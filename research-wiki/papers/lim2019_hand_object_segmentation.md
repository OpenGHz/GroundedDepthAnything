---
type: paper
node_id: paper:lim2019_hand_object_segmentation
title: "Hand and Object Segmentation from Depth Image using Fully Convolutional Network"
authors: ["Guan Ming Lim", "Prayook Jatesiktat", "Christopher Wee Keong Kuah", "Wei Tech Ang"]
year: 2019
venue: "EMBC 2019"
external_ids:
  arxiv: null
  doi: "10.1109/EMBC.2019.8857700"
  s2: null
tags: ["depth-only", "semantic-segmentation", "synthetic-to-real", "human-object", "geometry"]
added: 2026-08-29T05:59:37Z
---

# Hand and Object Segmentation from Depth Image using Fully Convolutional Network

## One-line thesis
Synthetic depth-only training can transfer to real Kinect-v2 pixel segmentation, showing strong but task-narrow geometric robustness.

## Problem / Gap
Upper-limb rehabilitation needs dense hand/object labels, but collecting pixel-level depth annotations is expensive. The paper focuses on whether a synthetic depth corpus can transfer to real Kinect-v2 frames.

## Method
A fully convolutional network is trained on synthetic Kinect-like depth images with automatically generated labels. The official release provides synthetic train/test sets and a separate real Kinect-v2 test set; labels distinguish foreground, left/right hands and arms, object, table, and background.

## Key Results
The official dataset release lists 10,000 synthetic training images, 1,000 synthetic test images, and 1,000 real Kinect-v2 test images. The IEEE abstract reports **70.4% mIoU** on real depth images and about **6 ms** GPU inference time.

## Assumptions
The depth sensor and capture geometry are Kinect-v2-like, the scene is hand-object interaction, and the label vocabulary is fixed to upper-limb rehabilitation. No RGB channel is used by the segmentation network.

## Limitations / Failure Modes
Despite the RGB-D camera used for acquisition, this is depth-only; it is not evidence for RGB-D fusion or broad multiclass indoor segmentation. The domain and object vocabulary are narrow, and the abstract does not provide per-class or cross-sensor quantitative analysis.

## Reusable Ingredients
Automatic pixel labels, a large synthetic depth corpus, and a separately held-out real sensor test are a useful minimal protocol for geometry-transfer experiments.

## Open Questions
Would adding RGB help or hurt under texture shift? Does the result persist on unseen Kinect generations, different camera poses, and non-human object categories?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Strongest quantitative depth-only comparator for G4/G6 ([DOI:10.1109/EMBC.2019.8857700](https://doi.org/10.1109/EMBC.2019.8857700)). It provides a useful upper-bound-style reference for what a carefully matched sensor/task can achieve without RGB.
