---
type: paper
node_id: paper:watanabe2018_multichannel_semantic_segmentation
title: "Multichannel Semantic Segmentation with Unsupervised Domain Adaptation"
authors: ["Kohei Watanabe", "Kuniaki Saito", "Yoshitaka Ushiku", "Tatsuya Harada"]
year: 2018
venue: "AUTONUE Workshop of ECCV 2018"
external_ids:
  arxiv: "1812.04351"
  doi: null
  s2: null
tags: ["rgb-d", "semantic-segmentation", "indoor", "negative-result", "domain-adaptation"]
added: 2026-08-29T05:58:32Z
---

# Multichannel Semantic Segmentation with Unsupervised Domain Adaptation

## One-line thesis
A source-only RGB+HHA baseline transfers poorly from SUNCG to NYUv2, motivating explicit domain adaptation and stronger controls.

## Problem / Gap
The work studies the large synthetic-to-real gap in indoor semantic segmentation and asks whether RGB, HHA depth, and boundary channels can be combined with unsupervised adaptation.

## Method
A dilated residual network receives RGB/HHA through early, late, or score fusion. The paper also predicts depth and instance boundaries as auxiliary tasks and applies MCD-style unsupervised domain adaptation using unlabeled NYUv2 images.

## Key Results
For the source-only SUNCG→NYUv2 protocol over 34 common classes, RGB obtains 3.2 mIoU, HHA 3.8, and RGB+HHA EarlyFusion **4.2 mIoU** (17.9 pixel accuracy). Adapted variants reach up to 13.2 mIoU, but use 795 unlabeled real NYUv2 training images; the 4.2 row is the relevant target-free control.

## Assumptions
SUNCG supplies 568,793 synthetic RGB+HHA+labels; NYUv2 supplies 654 real test images. The DRN-D-38 backbone is explicitly pretrained on ImageNet. HHA is a hand-designed encoding of depth, not raw metric depth.

## Limitations / Failure Modes
The source-only transfer is very weak, and the proposed improvements rely on target-domain UDA. ImageNet initialization violates a no-real-imagery-anywhere rule. SUNCG/NYUv2 class overlap and indoor sensor differences limit direct comparison with modern RGB-D DG.

## Reusable Ingredients
Use this as a negative control and as a warning to report source-only baselines separately from UDA. The 34-class protocol, HHA ablation, and auxiliary boundary/depth tasks are useful stress-test components.

## Open Questions
Would renderer-native depth, sensor-noise simulation, or a stronger RGB-D fusion architecture close the gap without target images? Which classes benefit from geometry after removing ImageNet initialization?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Negative reference for G3 and G6 ([arXiv:1812.04351](https://arxiv.org/abs/1812.04351)). It establishes a low source-only floor and prevents a literature review from conflating UDA gains with evidence for target-free geometric understanding.
