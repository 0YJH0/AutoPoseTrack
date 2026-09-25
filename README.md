# AutoPoseTrack

Research infrastructure for **autonomous, failure-aware, and recoverable 6D
object pose tracking**, targeting a CVPR 2027 submission.

The intended system autonomously initializes an object pose, tracks it locally,
estimates whether the current hypothesis remains recoverable, and relocalizes
after failure. Ground-truth pose is restricted to training supervision,
benchmark construction, and offline evaluation.

## Status

**Phase 0 complete (2026-09-25):** repository and environment audit, strict
data/model contracts, SE(3) and reference pose metrics, configurable state
manager, experiment artifact writer, example configurations, and unit tests.
No baseline performance or experimental result is claimed yet.

The first recommended baseline is **FoundationPose (model-based RGB-D)** behind
an adapter, using its registration path for initialization/relocalization and
its tracking path for local updates. See [the baseline audit](docs/baseline_audit.md)
for the decision and caveats.

## What is implemented

- Inference-only `FrameObservation` contract with no ground-truth field.
- Separate annotation interface used only by offline evaluation.
- Global estimator, local tracker, and reliability estimator interfaces.
- Object-to-camera SE(3) utilities using metres and column-vector composition.
- Reference ADD, ADD-S, rotation error, translation error, and threshold AUC.
- Configurable `UNINITIALIZED → RELOCALIZING → TRACKING → UNCERTAIN → LOST`
  state transitions with multi-frame hysteresis.
- Strict YAML validation and reproducibility manifest collection.
- Standard run directory creation without overwriting an existing experiment.

FoundationPose and YCB-Video adapters are deliberately not presented as
implemented until the official upstream demo has passed on a GPU-capable host.

## Installation

The lightweight core does not install FoundationPose or download datasets:

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

Use the `dev` profile for live source edits through a bind mount, and use the
separate `foundationpose` profile for CUDA dependencies. Full WSL/GPU setup,
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
python -m scripts.validate_config configs/experiments/foundationpose_ycbv_smoke.yaml
```

After replacing all `/REPLACE/WITH/...` values and selecting a documented mask
source, create a traceable run directory with:

```bash
python -m scripts.create_run configs/experiments/foundationpose_ycbv_smoke.yaml
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
- Translation, depth, model points, ADD, and ADD-S: metres.
- Rotation error: degrees.
- RGB: `uint8`, shape `(H, W, 3)`; depth: floating-point metres.
- Quaternion serialization, if introduced: scalar-last `(x, y, z, w)`.

Every third-party adapter must convert to these conventions and validate its
output at the boundary.

## Verification

```bash
find configs autoposetrack scripts tests docs third_party outputs -maxdepth 2 -type d | sort
python3 --version
nvidia-smi
python -m pytest
python -m scripts.validate_config configs/experiments/foundationpose_ycbv_smoke.yaml
```

The current host does **not** expose an NVIDIA GPU to WSL2. Resolve the GPU
preflight in `docs/open_questions.md` before installing or running FoundationPose.

## Next minimal executable task

Run a clean FoundationPose model-based demo in its own pinned environment, then
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

## License

Project-owned code is released under the [MIT License](LICENSE). Datasets,
FoundationPose, model weights, renderers, and all other third-party components
retain their own licenses and are not redistributed by this repository.

## Research principles

- Correctness and reproducibility before throughput.
- Config-driven protocols; no dataset paths or thresholds in source code.
- Explicit pose frames, multiplication order, quaternion order, and units.
- Third-party code is pinned and wrapped, not silently modified.
- Missing dependencies and missing observations fail explicitly.
- No fabricated results, test-GT inference, leakage, or protocol changes.
