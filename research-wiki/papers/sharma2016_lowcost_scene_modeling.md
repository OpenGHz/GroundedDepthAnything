---
type: paper
node_id: paper:sharma2016_lowcost_scene_modeling
title: "Low-Cost Scene Modeling using a Density Function Improves Segmentation Performance"
authors: ["Vivek Sharma", "Sule Yildirim-Yayilgan", "Luc Van Gool"]
year: 2016
venue: "arXiv preprint"
external_ids:
  arxiv: "1605.08464"
  doi: null
  s2: null
tags: ["depth-only", "semantic-segmentation", "robotics", "synthetic-to-real", "geometry"]
added: 2026-08-29T05:58:32Z
---

# Low-Cost Scene Modeling using a Density Function Improves Segmentation Performance

## One-line thesis
Randomized 3D human-object interaction scenes and noisy synthetic depth improve real Kinect pixel labeling for a small industrial vocabulary.

## Problem / Gap
Safe human-robot collaboration needs pixel labels for humans and nearby objects, but real RGB-D annotation is costly. The work asks whether realistic 3D interaction layouts can make synthetic depth useful in a real workspace.

## Method
V-REP scenes are sampled from a density function over human-object and object-object relationships. A Kinect-like renderer adds Gaussian depth noise; a random-decision forest classifies depth patches and a pairwise CRF smooths pixel labels.

## Key Results
The model is trained on **20,000 synthetic depth frames** and evaluated on **65 real Kinect depth maps**. For ten labels (human parts plus chair, plant, storage, and table), modeled interaction scenes improve average F1 from 0.76 to **0.84**; all reported evaluation data are real test maps.

## Assumptions
The camera is a fixed ceiling-mounted Kinect-like setup and the object vocabulary is small. Human skeletons recorded with a real Kinect are used to construct the synthetic human model/poses, so the data pipeline is not independent of real capture.

## Limitations / Failure Modes
Depth-only and classical random-forest/CRF rather than a modern RGB-D semantic network. Real skeleton recordings contribute to scene generation; the fixed viewpoint and narrow industrial setting limit claims about general geometry understanding.

## Reusable Ingredients
Interaction-aware scene randomization, sensor-noise injection, and explicit modeled-versus-non-modeled controls can be reused for robotics-oriented sim-to-real evaluation.

## Open Questions
Would the interaction prior transfer across camera viewpoints, sensors, and unseen object shapes? How much of the gain comes from relational geometry versus the fixed camera and skeleton prior?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
A geometry-transfer near-neighbor for G4/G5 ([arXiv:1605.08464](https://arxiv.org/abs/1605.08464)). It motivates testing scene-layout randomization while explicitly varying viewpoint and sensor rather than relying on a fixed camera.
