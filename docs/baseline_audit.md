# Phase 0 baseline and code audit

Audited: 2026-09-25. This is a code-integration audit, not a reproduced accuracy
or speed comparison. Runtime values vary with hardware and protocol; no local
GPU run has been completed.

## Decision

Use **FoundationPose model-based RGB-D** as the first complete baseline, behind
two project-owned adapters:

- registration as `GlobalPoseEstimator` for initialization/relocalization;
- tracking/refinement as `LocalPoseTracker` for frame-to-frame updates.

It is the only primary candidate here whose official release directly exposes
both roles and an explicit YCB-Video script. That minimizes cross-project pose,
renderer, checkpoint, and environment mismatch during Phase 1. It also creates a
clean experiment: keep the pose engine fixed and later compare no recovery,
heuristic recovery, and learned recoverability.

This recommendation is **conditional** on license acceptance and a successful
GPU/container smoke test. The NVIDIA Source Code License is not equivalent to a
permissive OSI license; institutional/legal review may be required before
redistribution or some uses.

## Candidate comparison

| Candidate | Official repository / license | Inputs and object prior | Segmentation / initialization | Tracking and YCB-V | Weights, cost, integration |
|---|---|---|---|---|---|
| FoundationPose (CVPR 2024) | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose), NVIDIA Source Code License | RGB-D for released model-based pipeline; CAD mesh, intrinsics, depth | Requires object mask for registration; registration does not require a pose; tracking takes prior pose | Native registration **and** tracking; official `run_ycb_video.py` | Released refiner/scorer weights. CUDA renderer and compiled extensions; upstream recommends Docker. Registration is much heavier than tracking; benchmark locally. **Medium-high** setup risk, **low** conceptual adapter risk. |
| GigaPose (CVPR 2024) | [nv-nguyen/gigaPose](https://github.com/nv-nguyen/gigaPose), MIT except inherited components | RGB, CAD-rendered templates; optional downstream refiner | Requires detection/segmentation (official pipeline uses CNOS detections); no prior pose | Global per-image estimation, not a temporal tracker; BOP-format evaluation includes YCB-V-era datasets/config lineage but current README paths focus BOP challenge sets—verify exact YCB-V recipe at pinned commit | Checkpoint download scripts supplied. Template onboarding plus coarse/refinement stack; paper emphasizes speed but measure total detection+render+refine cost. **Medium-high** integration, requiring a separate tracker. |
| MegaPose (CoRL 2022) | [megapose6d/megapose6d](https://github.com/megapose6d/megapose6d), Apache-2.0 unless noted | RGB; depth optional; CAD mesh, intrinsics, bounding box | Requires labeled 2D detection/box; coarse hypotheses plus render-and-compare refinement | Primarily per-image global estimation/refinement, not temporal tracking; official YCB-V example and BOP assets | Pretrained models and download tooling supplied. Mature but large rendering/data stack; typically slower multi-hypothesis pipeline. **High** integration if paired with a separate local tracker. |
| BundleSDF (CVPR 2023) / BundleTrack (CVPR 2021) | [FreeArtGS/BundleSDF](https://github.com/FreeArtGS/BundleSDF) and [wenbowen123/BundleTrack](https://github.com/wenbowen123/BundleTrack); verify all nested licenses before use | RGB-D video; designed for unknown objects; BundleSDF jointly reconstructs geometry | First-frame object mask; no CAD required; not autonomous category/object discovery | Strong temporal tracking/reconstruction, but official evaluations center on HO3D, YCBInEOAT, and BEHAVE rather than YCB-Video tracking protocol | Multiple native/CUDA components and pretrained feature/segmentation dependencies; concurrent reconstruction makes controlled local-basin experiments expensive. **Very high** integration. Useful later as a tracker comparison, not Phase 1. |

## Detailed fit analysis

### FoundationPose

The official repository describes a unified novel-object pose estimator and
tracker, released in model-based and model-free variants. The model-based path
accepts a CAD mesh and uses RGB, depth, camera intrinsics, and an object mask.
The demo registers on the first frame and switches to tracking; official scripts
cover LINEMOD and YCB-Video. Released weights are available, although upstream
notes that the public weights omit diffusion-augmented training data and may be
slightly below the paper variant.

Environment risk is real: the upstream setup uses PyTorch plus source-built
PyTorch3D/NVDiffRast and native extensions, and currently documents Python 3.11
for a local conda route while also recommending Docker. Therefore the first task
is an unchanged upstream smoke test, not immediate adapter development.

Research fit is excellent because registration and tracking share representations
while remaining callable modes. Raw scorer/refiner outputs should be preserved as
matching-confidence candidates, but must not be treated as calibrated reliability
without Phase 2 evidence.

### GigaPose

GigaPose is an RGB, CAD-template, novel-object global estimator. Its official
code provides training/testing, checkpoints, BOP-format tooling, CNOS detections,
template rendering, and optional MegaPose/GenFlow refinement. It has no native
temporal tracking state, so choosing it first would immediately force a second
major baseline and confound the initial reproduction. It remains a strong later
candidate for independent-per-frame and relocalization ablations.

### MegaPose

MegaPose consumes an RGB image (optional depth), intrinsics, object mesh, and a
labeled bounding box, and provides pretrained coarse/refinement models. It has
an official YCB-V example and a permissive top-level license. It is credible for
global initialization, but does not itself supply the desired temporal tracker;
its broad renderer/data stack increases adapter effort. Retain it as a Phase 5
global-estimator alternative after the first baseline is stable.

### BundleSDF / BundleTrack

These are genuine tracking systems rather than per-frame estimators. BundleSDF
tracks and reconstructs unknown objects from RGB-D video and only assumes a
first-frame mask. This is scientifically relevant to occlusion and long-term
tracking, but its online reconstruction, multithreaded/native stack, and different
benchmark focus make it a poor first controlled YCB-Video baseline. Audit nested
licenses and evaluation compatibility before any integration.

## Fair-comparison policy

For every estimator/tracker combination, report separately:

- oracle GT first-pose tracking;
- global initialization plus tracking without recovery;
- global per-frame estimation;
- autonomous recovery variants.

Keep masks/detections fixed within a comparison and report whether they are GT,
precomputed, or predicted. Account separately for detector/mask, template
onboarding, registration, tracking, and relocalization time. Never compare a
method using GT masks with one using predicted masks without an explicit label.

## Primary sources

- FoundationPose [official repository](https://github.com/NVlabs/FoundationPose)
  and [paper](https://arxiv.org/abs/2312.08344).
- GigaPose [official repository](https://github.com/nv-nguyen/gigaPose) and
  [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Nguyen_GigaPose_Fast_and_Robust_Novel_Object_Pose_Estimation_via_One_CVPR_2024_paper.pdf).
- MegaPose [official repository](https://github.com/megapose6d/megapose6d) and
  [paper](https://arxiv.org/abs/2212.06870).
- BundleSDF [official implementation](https://github.com/FreeArtGS/BundleSDF)
  and [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.pdf).
- BundleTrack [official repository](https://github.com/wenbowen123/BundleTrack)
  and [paper](https://arxiv.org/abs/2108.00516).
- [BOP Toolkit](https://github.com/thodan/bop_toolkit) for standard pose metric
  cross-checking.

