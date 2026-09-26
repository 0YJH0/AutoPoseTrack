# AutoPoseTrack research plan

Last updated: 2026-09-26

## Research claim and falsifiable question

The project tests whether a tracker that estimates *recoverability* from
observable signals can detect unrecoverable hypotheses and trigger global
relocalization more effectively than score thresholds, pose-jump heuristics,
and constant-motion filtering—without ground truth at inference time.

The central prediction target is

\[
c_t=P(T_t \in \mathcal B_{recoverable}\mid I_t,T_t,H_t).
\]

Success requires improvements in failure-detection quality and long-term
recovery behavior under an unchanged public evaluation protocol, not merely a
better per-frame pose score.

## Non-negotiable experiment boundaries

- YCB-Video is the first dataset; DexYCB and UAV data are out of Phase 1 scope.
- Ground truth is used only for training labels, controlled benchmark creation,
  and offline metrics.
- Inference modality is monocular RGB. Depth images, depth-derived features,
  ICP, and RGB-D checkpoints are prohibited in the primary protocol.
- The primary protocol uses posed RGB reference images and does not expose a
  precise CAD model to the estimator. CAD-enabled runs are auxiliary studies.
- All third-party revisions, data splits, seeds, configurations, checkpoints,
  and hardware are recorded.
- Every reported aggregate is reproducible from saved per-frame predictions.
- Pose convention at the project boundary will be an explicit object-to-camera
  4x4 transform, translation in metres. Quaternion serialization, if used, is
  scalar-last `(x, y, z, w)`. Adapters must convert and validate conventions.
- Symmetry-aware evaluation is metadata-driven; symmetric objects are never
  inferred from object names in source code.

## Phase roadmap and gates

### Phase 0 — repository and code audit (complete)

Delivered:

- research-oriented repository skeleton;
- verified WSL/Docker GPU environment;
- audited reference-based and CAD-enabled estimator families;
- selected Gen6D as the development baseline and MegaPose as a CAD auxiliary;
- recorded unresolved choices in `docs/open_questions.md`.

Gate: documentation is internally consistent and makes no performance claim.

### Phase 1 — baseline first (next)

1. **GPU/environment preflight.** Record driver, CUDA runtime/toolkit, GPU,
   VRAM, image digest, and upstream commit (host/container GPU already verified).
2. **Upstream smoke test.** Run the official Gen6D GenMOP/LINEMOD evaluation
   unchanged. Archive its command, logs, output visualization, and runtime.
3. **Pinned external dependency.** Add Gen6D under `third_party/` as a
   pinned submodule or external checkout; record patches separately. Never copy
   its implementation into `autoposetrack/`.
4. **Data contract.** Implement a YCB-Video sequence loader returning RGB,
   intrinsics, detection box, optional RGB-derived mask, object id, and
   timestamp/frame id. GT pose remains evaluation-only.
5. **Adapters.** Define reference/CAD-neutral `GlobalPoseEstimator` and
   `LocalPoseTracker` protocols and wrap Gen6D first. Preserve
   upstream scores and timing as raw outputs.
6. **Prediction schema.** Persist one row per frame/object plus event records;
   never hide failed frames. Include validity/error fields rather than NaNs with
   undocumented meaning.
7. **Metrics and tests.** Implement ADD, ADD-S, rotation/translation error and
   AUC with geometry unit tests and a tiny synthetic fixture. Cross-check against
   BOP Toolkit where definitions overlap.
8. **Reproduction run.** First reproduce the official protocol with GT first
   pose and upstream masks, then add global initialization as a separate run.
9. **Figures.** Generate pose-error timelines and ADD(-S) curves solely from
   the saved per-frame table.

Gate: one pinned YCB-Video sequence runs end-to-end and its metrics are checked
against hand-computed/synthetic cases before scaling up. Only after full
evaluation may performance be described as reproduced or not reproduced.

### Phase 2 — failure characterization

Freeze the Phase 1 baseline and log pose error, inter-frame motion, raw model
scores, mask IoU, visibility proxies, and runtime. Define failure labels only in
the evaluation layer. Analyze observable/label association and produce
`docs/failure_analysis.md`; do not select reliability features beforehand.

Gate: failure definition, temporal tolerance, and class imbalance treatment are
predeclared; figures are generated from saved tables.

### Phase 3 — recoverable-basin analysis

Apply seeded SE(3) perturbations to GT pose over translation and rotation grids,
including coupled perturbations. Run a fixed number of local updates and estimate
`P(recover | delta_R, delta_t)` with confidence intervals, broken down by object,
symmetry, visibility, and motion. Store every trial and perturbation seed.

Gate: convergence definition and rollout horizon are fixed before comparisons;
the empirical basin is reproducible from trial records.

### Phase 4 — reliability estimator

Start with heuristics and logistic regression, then MLP, and only then richer
fusion. Compare pose-error labels with future-recoverability labels. Use
sequence-level splits to avoid adjacent-frame leakage. Report AUROC, AUPRC,
precision/recall/F1 at declared operating points, calibration error, false alarm
rate, and missed-failure rate.

Gate: learned methods outperform declared heuristics on held-out sequences with
uncertainty estimates, or the negative result is reported.

### Phase 5 — autonomous tracking

Connect global registration, local tracking, reliability, an explicit state
machine, and verified relocalization. Thresholds, hysteresis, and retry policy
remain config values. Evaluate autonomous initialization, recovery latency,
long-term accuracy, jitter/drift, and compute cost.

Gate: all requested baselines and ablations use identical inputs, split, masks,
corruptions, and metrics.

## Experiment artifact contract

Each `outputs/<experiment>/` run must contain:

```text
config.yaml              resolved immutable config
manifest.json            git/upstream revisions, environment, hardware, data IDs
metrics.json             aggregate metrics and definitions/version
per_frame.csv            predictions, GT-for-evaluation, errors, score, state, mode
events.csv               initialization/failure/relocalization events
figures/                 derived only from persisted records
logs/                    stdout/stderr and timing
```

GT columns in `per_frame.csv` belong to the offline evaluation artifact and must
not enter the inference API. Prefer separate prediction and annotation tables
internally, joining them only in evaluation, to make leakage structurally hard.

## Next minimal executable task

From a pinned external checkout:

```bash
git submodule add https://github.com/megapose6d/megapose6d.git third_party/MegaPose
git -C third_party/MegaPose submodule update --init
git -C third_party/MegaPose rev-parse HEAD
docker pull ylabbe/megapose6d
docker compose --profile gpu build megapose
docker compose --profile gpu run --rm megapose nvidia-smi
```

These commands are a proposed procedure, not yet executed. Before cloning,
record the selected upstream commit in an ADR or manifest. Acceptance criteria:
GPU visible in container, extensions build without local source edits, demo
completes, output pose/visualization exists, and command/log/runtime/hardware are
archived. If the current upstream instructions differ, use and pin those exact
instructions rather than silently repairing them.
