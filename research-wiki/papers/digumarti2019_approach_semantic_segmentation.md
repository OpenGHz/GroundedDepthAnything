---
type: paper
node_id: paper:digumarti2019_approach_semantic_segmentation
title: "An Approach for Semantic Segmentation of Tree-like Vegetation"
authors: ["S. Tejaswi Digumarti", "Lukas Maximilian Schmid", "Giuseppe Maria Rizzi", "Juan Nieto", "Roland Siegwart", "Paul Beardsley"]
year: 2019
venue: "ICRA 2019"
external_ids:
  arxiv: null
  doi: "10.1109/ICRA.2019.8793576"
  s2: null
tags: ["rgb-d", "semantic-segmentation", "vegetation", "synthetic-to-real", "qualitative-only"]
added: 2026-08-29T05:59:37Z
---

# An Approach for Semantic Segmentation of Tree-like Vegetation

## One-line thesis
Late-fused RGB and HHA networks segment tree parts from synthetic single-frame RGB-D and show qualitative real-world transfer, but lack quantitative real evaluation.

## Problem / Gap
Agricultural robots need tree-part labels, but pixel annotation across species is expensive. The work studies whether synthetic single-frame RGB-D data can support tree-component parsing.

## Method
Separate RGB and HHA-depth CNNs are trained on rendered trees from six broadleaf species and combined with asynchronous late fusion and different learning rates. Pixels are assigned trunk, branch, twig, leaf, and background-style labels.

## Key Results
The IEEE abstract reports synthetic cross-species accuracy up to **92.5%**. Real-world RGB-D examples are evaluated qualitatively; no quantitative real-domain accuracy/mIoU is reported.

## Assumptions
The target consists of tree-like vegetation viewed in a compatible sensor setup, and HHA is used as the depth representation. The synthetic species and rendering choices are assumed to cover enough shape variation.

## Limitations / Failure Modes
No quantitative real test result means the claimed sim-to-real generalization cannot be compared or audited. The task is a specialized vegetation taxonomy, and HHA/late-fusion design may depend on the capture setup.

## Reusable Ingredients
Asynchronous modality training and late fusion are useful ablations for separating RGB texture cues from depth geometry cues.

## Open Questions
Can real masks be collected for a held-out species/sensor to quantify transfer? Does raw depth or metric-normalized depth outperform HHA under cross-camera changes?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Near-neighbor for G1 ([DOI:10.1109/ICRA.2019.8793576](https://doi.org/10.1109/ICRA.2019.8793576)): direct RGB-D multiclass segmentation with a clear geometry motivation, but real quantitative evaluation is the missing piece.
