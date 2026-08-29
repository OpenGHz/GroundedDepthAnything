---
type: paper
node_id: paper:joukovsky2020_multimodal_deep_network
title: "Multi-modal deep network for RGB-D segmentation of clothes"
authors: ["B. Joukovsky", "P. Hu", "A. Munteanu"]
year: 2020
venue: "Electronics Letters"
external_ids:
  arxiv: null
  doi: "10.1049/el.2019.4150"
  s2: null
tags: ["rgb-d", "semantic-segmentation", "clothing", "synthetic-to-real", "qualitative-only"]
added: 2026-08-29T05:59:37Z
---

# Multi-modal deep network for RGB-D segmentation of clothes

## One-line thesis
A synthetic RGB-D clothing dataset and multimodal encoder-decoder achieve strong synthetic segmentation and qualitative Kinect-v2 transfer, but no quantitative real-domain score is reported.

## Problem / Gap
RGB clothing datasets do not provide aligned depth, while appearance and darkness make clothing boundaries difficult. The paper introduces a synthetic RGB-D clothing benchmark and a multimodal segmentation network.

## Method
Blender/Cycles renders 53,354 RGB-D frames of posed people with randomized clothing, materials, lighting, and viewpoints. A two-encoder Multi-modal DeepLabv3+ uses Xception RGB/depth branches and multiscale fusion; training uses Pascal VOC-pretrained Xception weights.

## Key Results
The nine-class synthetic test mIoU is **92.05%** for the proposed RGB-D model (versus 89.75% for the RGB variant in the reported table). Kinect-v2 examples show qualitative transfer to multiple subjects and poses, but no quantitative real-domain metric is reported.

## Assumptions
The target camera is Kinect-v2-like and depth is preprocessed with inpainting. The clothing taxonomy is fixed and human-centric; source and target share the same broad object category.

## Limitations / Failure Modes
Real transfer is qualitative only, so zero-shot quantitative strength cannot be assessed. Pascal VOC pretraining violates a strict no-real-image-pretraining rule, and the setting is narrow nine-class clothing parsing.

## Reusable Ingredients
Randomized materials/poses/viewpoints, paired RGB-D rendering, and a modality ablation are useful for controlled tests of texture versus geometry.

## Open Questions
Would the multimodal gain remain with random initialization and quantified real masks? Can the same pipeline transfer across clothing styles, sensors, and backgrounds without inpainting tuned to the target camera?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Near-neighbor for G1 ([DOI:10.1049/el.2019.4150](https://doi.org/10.1049/el.2019.4150)): it is direct RGB-D and multiclass, but its missing quantitative real evaluation and external pretraining leave the strict claim unresolved.
