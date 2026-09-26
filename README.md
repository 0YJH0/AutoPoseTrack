# AutoPoseTrack

Research infrastructure for **autonomous, failure-aware, and recoverable
monocular-RGB 6D object pose tracking**, targeting a CVPR 2027 submission.

The intended system autonomously initializes an object pose, tracks it locally,
estimates whether the current hypothesis remains recoverable, and relocalizes
after failure. Ground-truth pose is restricted to training supervision,
benchmark construction, and offline evaluation.

## Status

**Phase 1 smoke path operational (2026-09-26):** the repository now includes a
BOP YCB-V single-object reader/index, a pinned MegaPose RGB adapter, strict
inference/evaluation separation, headless GPU Docker execution, pose metrics,
and structured run logs. The initial one-frame result is only an environment
smoke test, not a baseline performance claim.

The first recommended baseline is **MegaPose RGB** behind an adapter: its coarse
estimator provides initialization/relocalization and its RGB refiner uses the
previous pose for local tracking. Depth is prohibited at inference. See
[the baseline audit](docs/baseline_audit.md) for the decision and caveats.

## What is implemented

- Strict RGB-only `FrameObservation` contract with no ground-truth or depth field.
- Separate annotation interface used only by offline evaluation.
- Global estimator, local tracker, and reliability estimator interfaces.
- Object-to-camera SE(3) utilities using metres and column-vector composition.
- Reference ADD, ADD-S, rotation error, translation error, and threshold AUC.
- Configurable `UNINITIALIZED → RELOCALIZING → TRACKING → UNCERTAIN → LOST`
  state transitions with multi-frame hysteresis.
- Strict YAML validation and reproducibility manifest collection.
- Standard run directory creation without overwriting an existing experiment.
- Versioned feature-dataset schema, deterministic sequence-level splits, and
  explicit temporal leakage checks for reliability training.
- Trainable reliability-model protocol, NumPy logistic-regression baseline,
  checkpoint metadata, and training CLI ready for future MLP/PyTorch adapters.

MegaPose is pinned as a submodule at commit `f3b8e124`; its RGB coarse estimator
has passed a one-frame YCB-V GPU smoke run on the local RTX 4060 host.

## Single-object YCB-V MegaPose smoke run

The checked path uses object 5 (`006_mustard_bottle`) and oracle BOP
`bbox_obj` boxes. Ground-truth pose is loaded only after inference for ADD
evaluation. On a headless Docker/WSL host:

```bash
export YCBV_ROOT="$PWD/data/bop/ycbv"
export MEGAPOSE_DATA_ROOT="$PWD/data/megapose"
docker compose --profile gpu run --rm megapose \
  bash scripts/run_megapose_headless.sh \
  python -m scripts.run_ycbv_megapose \
  --dataset-root /data/ycbv --object-id 5 --sequence 50 \
  --max-frames 1 --model-name megapose-1.0-RGB \
  --renderer-workers 1 --batch-size 64 \
  --output outputs/megapose_rgb_ycbv_object5
```

Increase `--max-frames` only after the one-frame check succeeds. The output
contains `frame_index.json`, `per_frame.csv`, `events.csv`, `metrics.json`, the
resolved configuration/manifest, and structured logs.

## Installation

The lightweight core does not install MegaPose or download datasets:

```bash
conda env create -f environment.yml
conda activate autoposetrack
python -m pytest
```

For Docker-based development, Conda is not required inside the container. Build
the reproducible core environment and run its tests with:

```bash
docker compose build core
docker compose run --rm core
```

Build the separate training image with optional analysis dependencies:

```bash
docker compose build training
docker compose run --rm training
```

Use the `dev` profile for live source edits through a bind mount, and use the
separate `megapose` profile for CUDA dependencies. Full WSL/GPU setup,
image versioning, volume mounts, and troubleshooting are documented in
[`docs/docker.md`](docs/docker.md).

Alternatively, in an existing Python 3.9+ environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Third-party pose systems must live under `third_party/` as pinned submodules or
external checkouts. Their dependencies belong in isolated environments; do not
merge an unpinned CUDA stack into the lightweight core environment.

## Configuration and dry run

The checked-in smoke configuration contains explicit placeholder paths so an
invalid machine cannot silently pick up a dataset from an unrelated location:

```bash
python -m scripts.validate_config configs/experiments/megapose_rgb_ycbv_smoke.yaml
```

After replacing all `/REPLACE/WITH/...` values and selecting a documented mask
source, create a traceable run directory with:

```bash
python -m scripts.create_run configs/experiments/megapose_rgb_ycbv_smoke.yaml
```

This command only records the resolved configuration and host manifest. It does
not run inference and does not create fake metrics.

Every actual experiment must produce:

```text
outputs/<experiment_name>/
├── config.yaml
├── manifest.json
├── metrics.json
├── per_frame.csv
├── events.csv
├── figures/
└── logs/
```

Run names are immutable: the artifact writer refuses to overwrite an existing
directory.

## Reliability training interface

Tracker rollouts will write a `FeatureDataset` NPZ containing observable
features, binary offline recoverability labels, sequence/frame/object IDs,
feature names, and label-generation metadata. It intentionally excludes GT pose
matrices. Sequence IDs are mandatory so training, validation, and test frames
from one video cannot leak across splits.

After generating that dataset, edit
`configs/experiments/train_reliability_logreg.yaml` and run:

```bash
docker compose run --rm training \
  python -m scripts.train_reliability \
  configs/experiments/train_reliability_logreg.yaml
```

The command refuses to overwrite an output directory and stores model weights,
feature schema, dataset metadata, resolved configuration, environment manifest,
metrics, and training history. The included logistic regression is an auditable
baseline, not a claimed research result.

### Cloud logging and remote diagnosis

Training writes both human- and machine-readable records:

```text
outputs/<run>/
├── config.yaml
├── manifest.json
├── model.npz
├── metrics.json
├── history.json
└── logs/
    ├── train.log
    ├── metrics.jsonl
    ├── events.jsonl
    └── failure.json       # only when an exception occurs
```

JSONL records are flushed on every write so useful evidence survives cloud
preemption. To publish a small diagnostic report without checkpoints or raw
data:

```bash
python -m scripts.export_diagnostics outputs/<run-name>
git add reports/<run-name>
git commit -m "Add diagnostics for <run-name>"
git push
```

The trainer never pushes automatically and never receives a GitHub token. See
[`docs/cloud_training.md`](docs/cloud_training.md) for the end-to-end workflow.

## Dataset storage

For continuous YCB-Video tracking, the sparse 900-frame BOP test subset is not
sufficient. Start with `ycbv_base.zip`, `ycbv_models.zip`, and
`ycbv_test_all.zip`; reserve roughly 100–150 GB for extracted data, the pose
environment, caches, videos, and experiment outputs. Dataset paths are local
configuration values and must never be committed.

## Repository layout

```text
configs/                 Dataset, model, experiment, and ablation configuration
autoposetrack/           Project-owned Python package
  datasets/              Dataset and annotation contracts
  pose/                  Third-party global-estimator/tracker adapters
  reliability/           Reliability and recoverability models
  state_manager/         Explicit tracking state machine
  corruption/            Deterministic, non-destructive corruptions
  evaluation/            Pose and tracking metrics
  geometry/              SE(3), convention, and symmetry utilities
  visualization/         Paper figures and videos
  utils/                 Logging, configuration, reproducibility
scripts/                 Reproducible command-line entry points
tests/                   Unit and integration tests
docs/                    Research decisions and audits
third_party/             External repositories/submodules (not vendored edits)
outputs/                 Generated runs (ignored except documentation)
```

## Conventions

- Pose: `T_camera_object` (object-to-camera), 4x4 homogeneous matrix.
- Point application: `p_camera = R @ p_object + t` (column-vector semantics).
- Composition: `compose(A, B) == A @ B`.
- Translation, model points, ADD, and ADD-S: metres.
- Rotation error: degrees.
- RGB: `uint8`, shape `(H, W, 3)`; depth is forbidden at inference.
- Detection boxes: `(xmin, ymin, xmax, ymax)` in image pixels.
- Masks, when used, must be derived from RGB or explicitly labeled as oracle GT.
- Quaternion serialization, if introduced: scalar-last `(x, y, z, w)`.

Every third-party adapter must convert to these conventions and validate its
output at the boundary.

## Verification

```bash
find configs autoposetrack scripts tests docs third_party outputs -maxdepth 2 -type d | sort
python3 --version
nvidia-smi
python -m pytest
python -m scripts.validate_config configs/experiments/megapose_rgb_ycbv_smoke.yaml
```

The current host and Docker GPU passthrough have been verified with an RTX 4060.

## Next minimal executable task

Run a clean MegaPose RGB demo in its own pinned environment, then
save the exact upstream commit, container/environment digest, GPU information,
runtime, and output artifact. Do not integrate YCB-Video until this smoke test
passes. Proposed commands and acceptance criteria are in
[`docs/research_plan.md`](docs/research_plan.md).

## Documentation

- [Research plan](docs/research_plan.md)
- [Baseline and code audit](docs/baseline_audit.md)
- [Environment audit](docs/environment_audit.md)
- [Open questions and decision gates](docs/open_questions.md)
- [Docker development and GPU setup](docs/docker.md)
- [Cloud training, logging, and diagnosis](docs/cloud_training.md)

## License

Project-owned code is released under the [MIT License](LICENSE). Datasets,
MegaPose, model weights, renderers, and all other third-party components
retain their own licenses and are not redistributed by this repository.

## Research principles

- Correctness and reproducibility before throughput.
- Config-driven protocols; no dataset paths or thresholds in source code.
- Explicit pose frames, multiplication order, quaternion order, and units.
- Third-party code is pinned and wrapped, not silently modified.
- Missing dependencies and missing observations fail explicitly.
- No fabricated results, test-GT inference, leakage, or protocol changes.
