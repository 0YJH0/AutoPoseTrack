# AutoPoseTrack

AutoPoseTrack is research infrastructure for **RGB-only, CAD-free,
reference-based autonomous 6D object pose tracking**, targeting a CVPR 2027
submission. The proposed contribution is not a new single-frame pose estimator.
It is a recoverability-aware framework that decides when to continue local
tracking and when to find the object again:

```text
Posed RGB References → Find → Follow → Judge Recoverability → Find Again
```

Ground-truth pose is restricted to dataset analysis, supervision, label
construction, and offline evaluation. It must not enter inference features.

## Method roles

The repository distinguishes three roles:

- **Development baseline — Gen6D:** the primary implementation route because it
  is RGB-only, reference-based, and does not require precise CAD at inference.
- **Base model:** the frozen global estimator and local tracker used in one
  experiment. These components are not the proposed contribution.
- **Comparison methods:** Gen6D, NOPE, DVMNet, OrienPose, RGBTrack, MegaPose,
  and optionally FoundationPose, each reported with its actual input assumptions.

MegaPose is an implemented **CAD-enabled auxiliary baseline**. It validates GPU
inference, rollout, feature generation, and recovery-analysis infrastructure,
but it does not define the paper's primary setting. See
[method roles](docs/method_roles.md) and
[the Gen6D reproduction gate](docs/gen6d_reproduction.md).

## Current status

Implemented and tested:

- RGB observation and posed-reference contracts separated from GT annotations.
- Unified global estimator/local tracker/recoverability interfaces.
- A pinned Gen6D source tree and lazy `Gen6DAdapter` for its real build/predict
  boundary. Official weights and benchmark reproduction are still pending.
- A working MegaPose RGB + CAD adapter and headless CUDA Docker path.
- YCB-V/BOP single-object indexing, ADD/ADD-S, rotation and translation errors.
- Temporal recoverability features, sequence-level splitting, logistic-regression
  training, structured logs, and immutable output directories.
- Recoverability-aware state/routing logic and controlled SE(3) recovery-basin
  trials.
- Evidence-first BOP dataset audit and threshold-free alignment of tracker errors
  with natural dataset attributes.
- A GT-free full-frame motion proposal module with Farneback flow,
  homography/affine RANSAC compensation, adaptive residual saliency, component
  proposals, temporal tracklets, Recall@K evaluation, and debug video.

The checked-in YCB-V audit covers the BOP19 test target subset: 12 scenes,
900 sparse keyframes, 4,125 object instances, and all 21 YCB-V objects. These
frames support standard pose analysis, but **not a valid continuous tracking
protocol**. The audit and current limitations are documented in
[dataset audit](docs/dataset_audit.md) and
[failure analysis](docs/failure_analysis.md).

## Quick start

### Lightweight local environment

```bash
conda env create -f environment.yml
conda activate autoposetrack
python -m pip install -e ".[dev,viz]"
python -m pytest -q
```

Or use the reproducible core image:

```bash
docker compose build core
docker compose run --rm core python -m pytest -q
```

Conda is not required inside Docker. Use the `dev` profile when source files
must be bind-mounted for live editing, the `training` service for reliability
training, and the `megapose` service for the isolated CUDA stack. See
[Docker setup](docs/docker.md).

### Audit the downloaded BOP-YCB-V split

```bash
python -m scripts.analyze_dataset \
  --dataset-root data/bop/ycbv \
  --split test \
  --output analysis/ycbv_bop19_audit_v2 \
  --seed 2027 \
  --visual-samples 24
```

This writes per-instance attributes, object/sequence summaries, distributions,
and a deterministic visual audit. It does not fabricate occlusion, truncation,
velocity, or reference-view gaps when the required annotations are unavailable.
Output directories must be new. The checked-in first audit is under
[`analysis/ycbv_bop19_audit_v1`](analysis/ycbv_bop19_audit_v1).

### Run the implemented MegaPose auxiliary smoke test

```bash
export YCBV_ROOT="$PWD/data/bop/ycbv"
export MEGAPOSE_DATA_ROOT="$PWD/data/megapose"

docker compose --profile gpu build megapose
docker compose --profile gpu run --rm megapose nvidia-smi
docker compose --profile gpu run --rm megapose \
  bash scripts/run_megapose_headless.sh \
  python -m scripts.run_ycbv_megapose \
  --dataset-root /data/ycbv \
  --object-id 5 --sequence 50 --max-frames 1 \
  --model-name megapose-1.0-RGB \
  --renderer-workers 1 --batch-size 64 \
  --output outputs/megapose_object5_smoke_v2
```

This experiment uses an oracle GT bounding box and precise CAD. GT pose is
loaded only after inference for evaluation. It is an engineering smoke test,
not a CAD-free baseline result.

## Evidence-first experiment workflow

Experiments must follow this order:

```text
Raw Dataset
  → Attribute Extraction
  → Distribution Analysis
  → Natural Failure Analysis
  → Validation-derived Stratification
  → Comparison / Ablation
  → Paper Tables
```

Two protocols are intentionally separate:

- [`bop_pose.yaml`](configs/protocols/bop_pose.yaml): sparse BOP frames for
  single-frame pose/global-estimator evaluation.
- [`continuous_tracking.yaml`](configs/protocols/continuous_tracking.yaml):
  consecutive RGB sequences with FPS/timestamps and posed references for
  initialization, failure awareness, relocalization, and long-term metrics.

Difficulty thresholds must be derived from an independent validation split,
saved before test evaluation, and never tuned against test performance. A final
`dataset_stratification.yaml` is therefore intentionally absent: the current
download contains only a sparse test subset.

## Diagnostic sparse-keyframe pipeline

The following pipeline is retained to exercise the implemented components. The
word “rollout” here refers to software execution over ordered BOP targets; it is
not evidence for continuous-video tracking because frame gaps are irregular.

```bash
docker compose --profile gpu run --rm megapose \
  bash scripts/run_megapose_headless.sh \
  python -m scripts.run_ycbv_rollout \
  --dataset-root /data/ycbv \
  --object-id 5 --sequence 50 --sequence 52 \
  --output outputs/ycbv_object5_sparse_rollout_v2

python -m scripts.build_recoverability_features \
  --rollout outputs/ycbv_object5_sparse_rollout_v2/rollout.csv \
  --diameter-m 0.196463 \
  --history-frames 5 --future-frames 10 \
  --train-sequence 000050 \
  --validation-sequence 000052 \
  --output outputs/ycbv_object5_sparse_features_v2
```

The generated labels use GT pose error offline; model inputs remain observable
at RGB inference time. Because this pilot has only two scenes, strong class
imbalance, and no independent test split, it validates code paths only.

Align continuous pose error with audited natural attributes without inventing a
failure threshold:

```bash
python -m scripts.analyze_tracking_failures \
  --predictions outputs/ycbv_object5_sparse_rollout_v2/rollout.csv \
  --attributes analysis/ycbv_bop19_audit_v1/per_frame_attributes.csv \
  --output analysis/object5_sparse_failure_alignment_v2 \
  --error-field pose_error_m
```

## Reliability training and logs

After generating a feature dataset:

```bash
export FEATURE_DATA_ROOT="$PWD/outputs/ycbv_object5_sparse_features_v2"
docker compose build training
docker compose run --rm training \
  python -m scripts.train_reliability \
  configs/experiments/train_reliability_logreg_pilot.yaml
```

The included logistic regression is an auditable training/interface baseline,
not a paper claim. A run stores its resolved config, environment manifest,
weights, metrics, history, JSONL events, and human-readable logs. Export a small
report for remote diagnosis with:

```bash
python -m scripts.export_diagnostics outputs/<run-name>
```

Raw datasets, checkpoints, full output directories, and credentials must not be
committed. Only compact reports and explicitly selected audit artifacts belong
in Git. See [cloud training](docs/cloud_training.md).

## Full-frame motion proposals

The independent phase-1 motion branch proposes candidate regions without GT
boxes, masks, poses, CAD, or reference identity:

```bash
python -m pip install -e ".[motion]"
python -m scripts.render_motion_proposals \
  --input data/videos/example.mp4 \
  --config configs/motion_proposal/default.yaml \
  --output outputs/motion_example_v1
```

It is a high-recall scene-search module, not a detector. Reference verification
will later decide which, if any, proposal is the target. See
[motion proposal design](docs/motion_proposal_design.md) and
[failure-analysis protocol](docs/motion_proposal_failure_analysis.md).

Dynamic-object evaluation is supported on continuous YCBInEOAT sequences and
official 150-frame HOT3D-Clips. Ground-truth masks/boxes are read only after
proposal generation:

```bash
python -m scripts.run_dynamic_motion_proposals \
  --dataset ycbineoat --input data/YCBInEOAT/cracker_box_reorient \
  --start-frame-index 117 --max-pairs 149 \
  --config configs/motion_proposal/raft_cuda.yaml \
  --output outputs/ycbineoat_cracker_raft_cuda_v2

python -m scripts.run_dynamic_motion_proposals \
  --dataset hot3d \
  --input data/HOT3D-Clips/train_aria/clip-001852.tar \
  --stream-id 214-1 \
  --config configs/motion_proposal/raft_cuda.yaml \
  --output outputs/hot3d_clip_001852_raft_cuda_v1
```

Each run writes `motion_proposals.mp4`, six-panel debug images, per-frame
timings, proposals, per-instance metrics, and `summary.json`.

The optional `TorchCudaMotionBackend` accelerates dense background-flow,
residual, magnitude, and threshold computation. On the local RTX 4060 it reduced
that isolated 576×768 stage from 85.11 ms to 6.13 ms (13.87× mean speedup) with
100% threshold-mask agreement. A public BOP-YCB-V adjacent-pair test also
exposed the method boundary: static scene objects are correctly absorbed by
camera-motion compensation, so motion cannot be the only initialization path.

## Data requirements

The current 1.6 GB BOP19 subset is enough for code tests, dataset audit, and
single-frame pose diagnostics. The primary paper protocol additionally needs:

- leakage-free posed RGB reference images with known poses;
- disjoint sequence-level train/validation/test splits;
- truly consecutive RGB query frames;
- FPS or timestamps for physical motion rates;
- GT poses for offline evaluation only.

The BOP test subset must not be reused as its own reference bank. For the Gen6D
route, first reproduce the unchanged upstream GenMOP/LINEMOD evaluation, then
prepare disjoint YCB-V onboarding data and metric alignment.

## Repository layout

```text
analysis/                Versioned dataset/failure audits and figures
autoposetrack/           Project-owned Python package
  analysis/              Dataset-attribute extraction
  datasets/              Dataset and annotation boundaries
  pose/                  Global-estimator and tracker adapters
  reliability/           Features, models, and routing
  rollout/               Diagnostic and controlled rollouts
  state_manager/         Explicit tracking state machine
  evaluation/            Pose metrics and model I/O
configs/
  protocols/             Standard-pose vs continuous-tracking contracts
  experiments/           Reproducible run configuration
docs/                    Decisions, audits, setup, and experiment guides
scripts/                 Thin command-line entry points
tests/                   Unit and integration tests
third_party/             Pinned external repositories/submodules
outputs/                 Generated runs; ignored except documentation
reports/                 Compact diagnostic exports
```

## Conventions

- Pose: object-to-camera `T_camera_object`, homogeneous `4 × 4` matrix.
- Point application: `p_camera = R @ p_object + t`.
- Composition: `compose(A, B) == A @ B`.
- Translation, model points, ADD, and ADD-S: metres.
- Rotation error: degrees.
- RGB: `uint8`, shape `(H, W, 3)`; depth is forbidden in the primary setting.
- Detection boxes: `(xmin, ymin, xmax, ymax)` in image pixels.
- GT boxes/masks must be explicitly marked as oracle inputs.

完整的数据准备、审计、训练、测试和日志命令见
[中文训练与测试指南](docs/中文训练与测试指南.md)。
