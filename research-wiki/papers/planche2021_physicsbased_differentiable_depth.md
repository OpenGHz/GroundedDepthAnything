---
type: paper
node_id: paper:planche2021_physicsbased_differentiable_depth
title: "Physics-based Differentiable Depth Sensor Simulation"
authors: ["Benjamin Planche", "Rajat Vikram Singh"]
year: 2021
venue: "arXiv preprint"
external_ids:
  arxiv: "2103.16563"
  doi: null
  s2: null
tags: ["depth-only", "semantic-segmentation", "sensor-simulation", "synthetic-to-real", "geometry"]
added: 2026-08-29T05:58:32Z
---

# Physics-based Differentiable Depth Sensor Simulation

## One-line thesis
Physics-based differentiable simulation makes synthetic depth scans more sensor-realistic and improves depth-based semantic segmentation on real scans.

## Problem / Gap
Rendered depth maps are unrealistically clean compared with structured-light and stereo sensors. The paper targets the sensor-realism gap for depth-based recognition, including semantic segmentation.

## Method
Physics-based Differentiable Depth Sensor Simulation (DDS) models structured-light projection, shadows, stereo block matching, statistical noise, and post-processing in a differentiable pipeline. A CNN is trained on synthetic 2.5D scans rendered from reconstructed indoor scenes.

## Key Results
On 2D-3D-S, using eight depth-discernible classes, real-test pixel accuracy is 35.3% for clean depth, 65.3% for DepthSynth, 62.9% for off-the-shelf DDS, and 73.5% for real-data training (Table S1). The `DDS (train.)` value of 69.8% uses optimization against real scans and is not target-free.

## Assumptions
The target sensor type and intrinsic parameters are known; reconstructed 3D scenes and semantic masks are available for rendering. Depth-only CNN segmentation is used, with class imbalance handled by Dice loss.

## Limitations / Failure Modes
It is depth-only, not direct RGB-D segmentation. The optimized DDS setting uses real scans and therefore violates strict synthetic-only training. Reconstructed target scenes can introduce scene overlap, and the benchmark covers only classes discernible in 2.5D.

## Reusable Ingredients
Sensor-specific noise and hole simulation, differentiable calibration, and an explicit clean-versus-realistic depth ablation are valuable for a geometry-transfer study.

## Open Questions
Can DDS-style sensor simulation improve a direct RGB-D model across unseen sensors without any real calibration scans? Which noise components matter for semantic boundaries versus object interiors?

## Claims
_No claims tracked yet — populate via `/proof-checker`._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Key reference for G4 (native sensor realism, noise, holes, scale; [arXiv:2103.16563](https://arxiv.org/abs/2103.16563)). It suggests that a strict RGB-D benchmark should report both ideal-rendered and sensor-simulated depth.
