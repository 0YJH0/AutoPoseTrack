# Full-Frame Motion Proposal Module

## Research boundary

This module answers **“where might a moving target be?”** It does not answer
which proposal is the reference object and does not estimate pose. It consumes
only two RGB frames; camera intrinsics and tracker state are not required by the
phase-1 implementation. GT boxes are accepted only by the separate evaluator.

```text
Full RGB Pair
  → Replaceable Optical Flow
  → Dominant Background Model
  → Residual Flow
  → Adaptive Saliency
  → Components / Merge / Expansion
  → Short-term Tracklets
  → Top-K Candidate Regions
```

Reference verification and global pose estimation deliberately remain outside
`autoposetrack.motion`.

## Implemented components

- `flow/base.py`: `OpticalFlowEstimator` interface.
- `flow/farneback.py`: CPU Farneback baseline. RAFT/GMFlow can implement the
  same interface without changing downstream code.
- `background/models.py`: sampled dense-flow correspondences, affine or
  homography RANSAC, quality checks, explicit affine/previous/no-compensation
  fallback, and dense background-flow generation.
- `residual.py`: observed minus predicted-background flow.
- `dense/`: interchangeable NumPy CPU and `TorchCudaMotionBackend`
  implementations for dense homography flow, residual, three magnitude maps,
  adaptive threshold, and binary threshold mask. CUDA timing includes H2D/D2H
  transfers and synchronizes before measurement.
- `saliency.py`: MAD, percentile, or fixed threshold plus configurable median,
  opening, closing, and hole filling.
- `components.py`: connected components, direction consistency, interpretable
  scores, small-strong-component retention, expansion, merging, and Top-K.
- `temporal.py`: short-lived IoU/centroid tracklets. A first observation is
  always emitted; persistence changes score rather than acting as a hard gate.
- `motion_proposer.py`: orchestration, timings, quality diagnostics, and the
  record-only `motion_alignment_score` reliability cue.
- `evaluation/proposal_metrics.py`: Recall@K, maximum IoU, center recall, and
  average proposal count. This is the only layer that consumes GT boxes.
- `visualization/motion_visualizer.py`: six-panel debug image and proposal
  overlay. GT is optional and labeled evaluator-only.

## Background quality and fallback

The requested background model is valid only if it passes all configured gates:

- number of sampled correspondences;
- RANSAC inlier ratio;
- convex-hull spatial coverage of inliers;
- median inlier reprojection error.

The result records requested and used model types, validity, fallback, quality,
and timing. A bad homography never silently becomes trusted background motion.
If affine fallback also fails, the code uses zero compensation and keeps
`background_model_valid=false`.

Homography is only a dominant 2D approximation. Strong translation with depth
variation, parallax, rolling shutter, dynamic backgrounds, and reflections are
known failure modes. Phase 1 intentionally does not add SLAM, depth, VIO, or 3D
scene flow.

## Configuration

Default parameters are in `configs/motion_proposal/default.yaml`. They are code
defaults, not paper-selected values. Final values must be chosen on validation
sequences and frozen before test.

Install the isolated dependency:

```bash
python -m pip install -e ".[motion,dev]"
```

For the CUDA dense backend, install a CUDA-compatible PyTorch build for the
host, then select `configs/motion_proposal/torch_cuda.yaml`. The implementation
raises an error when CUDA is unavailable; it never silently executes on CPU.

or use Docker:

```bash
export MOTION_INPUT_ROOT="$PWD/data"
docker compose --profile motion build motion
docker compose --profile motion run --rm motion bash
```

Render an ordered image directory or video:

```bash
python -m scripts.render_motion_proposals \
  --input data/videos/example.mp4 \
  --config configs/motion_proposal/default.yaml \
  --output outputs/motion_example_v1 \
  --max-frames 300 \
  --debug-every 30
```

Output:

```text
outputs/motion_example_v1/
├── motion_proposals.mp4
├── frames.csv
├── proposals.csv
├── summary.json
└── debug/frame_*.jpg
```

`frames.csv` records background validity/quality, raw/background/residual flow
means, threshold, proposal count, and stage runtimes. `proposals.csv` records
each box, tracklet, flow statistics, score component, and final score.

## Proposal-level evaluation

Prepare a target-specific CSV with columns:

```text
frame_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2
```

Then run the GT-isolated evaluator:

```bash
python -m scripts.evaluate_motion_proposals \
  --proposals outputs/motion_example_v1/proposals.csv \
  --frames outputs/motion_example_v1/frames.csv \
  --gt-bboxes data/evaluation/example_gt_bbox.csv \
  --iou-threshold 0.5 \
  --output outputs/motion_example_eval_v1
```

The evaluator reports Recall@1/3/5/10, maximum proposal IoU, center recall, and
average proposals/frame. The IoU threshold is an evaluation protocol parameter,
not an input to proposal generation.

## CPU/CUDA consistency and benchmark

```bash
python -m scripts.benchmark_dense_motion_backends \
  --height 576 --width 768 \
  --warmup 10 --iterations 50 \
  --require-cuda --check-consistency \
  --output outputs/dense_motion_backend_benchmark_rtx4060.json
```

On the local RTX 4060 Laptop GPU, including transfers and synchronization:

| Backend | Mean | Median |
|---|---:|---:|
| NumPy CPU | 85.11 ms | 83.74 ms |
| Torch CUDA | 6.13 ms | 5.77 ms |

Mean speedup was 13.87× for the isolated 576×768 dense stage. CPU/CUDA mask
agreement was 100%; maximum background/residual discrepancy was
`1.54e-4` pixels and threshold difference was `1.74e-5`. This does not imply a
13.87× end-to-end speedup: Farneback, RANSAC, morphology, connected components,
merge, and temporal association remain on CPU.

## Controlled motion scenarios

```bash
python -m scripts.test_motion_scenarios \
  --dense-backend numpy_cpu \
  --output outputs/motion_scenarios_cpu_v1

python -m scripts.test_motion_scenarios \
  --dense-backend torch_cuda \
  --output outputs/motion_scenarios_torch_cuda_v1
```

The controlled renderer tests (a) camera translation with the target fixed in
the scene and (b) camera translation plus independent target motion. The first
case produced target residual `0.010 px` and no target center recall under CUDA.
The second produced target residual `11.87 px` and center recall, but only
`0.085` maximum IoU because Farneback motion spread into a large component.

## Dynamic-object public datasets

Use `scripts.run_dynamic_motion_proposals` for continuous sequences where the
object itself moves. Two adapters are intentionally separate from proposal
generation:

- `ycbineoat`: reads `rgb/` for inference and `gt_mask/` (falling back to
  `masks/`) only for evaluator boxes. This is the clean fixed-camera,
  moving-object control.
- `hot3d`: reads the Aria `214-1` RGB stream from an official 150-frame clip
  and uses `objects.json/boxes_amodal` only after inference. This is the hard
  moving-camera plus moving-object case.

```bash
python -m scripts.run_dynamic_motion_proposals \
  --dataset hot3d \
  --input data/HOT3D-Clips/train_aria/clip-001852.tar \
  --stream-id 214-1 \
  --config configs/motion_proposal/raft_cuda.yaml \
  --output outputs/hot3d_clip_001852_raft_cuda_v1 \
  --debug-every 10
```

The selected HOT3D RGB-visible object (BOP id 29) has a 0.575 m axis-aligned
world-trajectory span in the clip. With pretrained RAFT-Small (640-pixel
inference cap), 149 frame pairs / 149 annotated instances give center recall
97.32%, Recall@10 at IoU 0.3 of 2.68%, and Recall@10 at IoU 0.5 of 0.67%.
RAFT strongly improves motion coverage, but the selected components often join
the manipulated object to the hand/arm and remain too large for good IoU. Do
not tune on the test clip; create a separate validation clip/config first.
Both results and six-panel images are kept; the poor IoU is not hidden.

## Public pose-dataset test

The downloaded BOP-YCB-V test subset is sparse, so the runner accepts only
strictly adjacent frame IDs (`Δframe=1`):

```bash
python -m scripts.run_ycbv_motion_proposals \
  --dataset-root data/bop/ycbv \
  --config configs/motion_proposal/torch_cuda.yaml \
  --output outputs/ycbv_motion_adjacent_torch_cuda_v1 \
  --debug-every 5
```

Across all 93 legal adjacent pairs (433 GT object instances), Recall@10 was
0.46% at IoU 0.3 and 0% at IoU 0.5; center recall was 0.92%. This is expected
and important: YCB-V objects are largely static in the scene while the camera
moves, so correct background compensation removes their shared motion. Motion
proposal alone therefore cannot autonomously initialize a static reference
object. A complementary full-frame reference/appearance search is mandatory.

## Required comparison and ablations

`configs/motion_proposal/ablations.yaml` defines the experiment matrix:

- raw flow magnitude;
- affine compensation;
- homography compensation;
- full homography + adaptive/spatial/temporal pipeline;
- no adaptive threshold;
- no direction score;
- no temporal aggregation;
- no merging;
- no expansion.

Each run must report Recall@K, average proposal count, flow/background/proposal
runtime, and background validity rate. Natural-attribute analysis joins results
with `per_frame_attributes.csv` after inference; attributes never enter the
proposal generator.

## Integration contract

Future autonomous initialization and relocalization should call:

```python
result = motion_proposer.propose(previous_rgb, current_rgb, frame_id)
target = reference_verifier.verify(current_rgb, result.proposals, references)
pose = global_estimator.estimate(target)
```

The module may return zero proposals. It may also return several unrelated
moving objects. “No motion” does not mean “target absent,” and motion mismatch
does not by itself mean tracking failure.
