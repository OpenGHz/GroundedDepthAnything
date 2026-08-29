---
type: paper
node_id: paper:vu2019_dada_depthaware_domain
title: "DADA: Depth-Aware Domain Adaptation in Semantic Segmentation"
authors: ["Tuan-Hung Vu", "Himalaya Jain", "Maxime Bucher", "Matthieu Cord", "Patrick Perez"]
year: 2019
venue: "arXiv preprint"
external_ids:
  arxiv: "1904.01886"
  doi: "10.1109/ICCV.2019.00746"
  s2: null
tags: ["depth-privileged", "semantic-segmentation", "domain-adaptation", "synthetic-to-real"]
added: 2026-08-29T05:58:32Z
---

# DADA: Depth-Aware Domain Adaptation in Semantic Segmentation

## One-line thesis
Depth as source-side privileged information can improve synthetic-to-real semantic segmentation, but DADA relies on unsupervised target-domain adaptation.

## Problem / Gap
Synthetic urban segmentation has a large appearance gap to real Cityscapes/Vistas. DADA asks whether source-side dense depth can act as privileged information during unsupervised domain adaptation.

## Method
The network predicts segmentation and auxiliary depth, and uses depth-aware adversarial alignment/late fusion. Real target images are available without labels during UDA training; depth is not required as a direct inference input in the privileged-information formulation.

## Key Results
On SYNTHIA→Cityscapes, the reported 16-class DADA result is **49.8 mIoU** with a positive depth-driven gain over its RGB counterpart. The result is an adapted model and must not be counted as target-free synthetic-only RGB-D segmentation.

## Assumptions
Annotated SYNTHIA source RGB/depth and unannotated real Cityscapes or Mapillary Vistas target images are available during training. The method uses ResNet/VGG-era pretrained segmentation backbones in the published protocol.

## Limitations / Failure Modes
Target images are explicitly used, so the protocol fails the no-adaptation criterion. Depth functions primarily as privileged training information rather than a direct paired RGB-D input at deployment. It is therefore context, not a qualifying paper.

## Reusable Ingredients
Depth-aware adversarial objectives and a depth-driven gain metric are useful for distinguishing privileged supervision from direct RGB-D inference.

## Open Questions
How much of DADA's gain would remain if target images were removed and depth were required at inference? Can the depth-aware objective be converted into a domain-generalization method with no real data?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Negative/context node for protocol hygiene ([arXiv:1904.01886](https://arxiv.org/abs/1904.01886)). It helps prevent conflating depth-privileged UDA with direct RGB-D zero-shot transfer.
