# Dynamic-object motion proposal benchmark v1

Main configuration: pretrained torchvision RAFT-Small plus CUDA dense
processing, `configs/motion_proposal/raft_cuda.yaml`. Farneback is retained as
an ablation/fallback.

## YCBInEOAT

- Official `cracker_box_reorient` sequence, fixed camera.
- A contiguous 150-frame window starts at index 117; it maximizes summed GT
  bbox-center displacement among possible 150-frame windows and contains 80
  adjacent pairs with center displacement greater than 2 px. GT selects the
  evaluation window only and is not passed to the proposer.
- Evaluated pairs / annotated instances: 149 / 149.
- Center recall: 0.5705.
- Recall@10, IoU >= 0.3: 0.1074.
- Recall@10, IoU >= 0.5: 0.0671.
- Mean maximum IoU: 0.1138.
- Mean end-to-end runtime: 91.83 ms/pair (RAFT 68.05 ms) on RTX 4060.

## HOT3D-Clips (RAFT)

- Official train Aria clip: `clip-001852`, RGB stream `214-1`, 150 frames.
- Evaluated pairs / annotated instances: 149 / 149.
- RGB-visible target: BOP id 29; world-translation net displacement 0.171 m,
  axis-aligned trajectory span 0.575 m.
- Center recall: 0.9732.
- Recall@10, IoU >= 0.3: 0.0268.
- Recall@10, IoU >= 0.5: 0.0067.
- Mean maximum IoU: 0.0482.
- Mean end-to-end runtime: 262.89 ms/pair (RAFT 77.74 ms) on RTX 4060.

The full generated video and frame-level tables remain under
`outputs/ycbineoat_cracker_raft_cuda_v2/` and
`outputs/hot3d_clip_001852_raft_cuda_v1/` and are intentionally gitignored.
This report records the compact result only. GT boxes were used after proposal
generation for evaluation/debug overlays.

Interpretation: RAFT greatly improves HOT3D center coverage over Farneback
(97.32% versus 15.44%), but IoU remains poor because the object, hand, arm, and
fisheye residual often merge into an oversized component. Better optical flow
alone does not solve instance separation; the next improvement belongs in
motion segmentation/component splitting or reference-conditioned verification.
