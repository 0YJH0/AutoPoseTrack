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

