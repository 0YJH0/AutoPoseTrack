# Method roles and estimator-agnostic protocol

Last updated: 2026-09-26.

## Three roles are not interchangeable

1. **Development baseline — Gen6D.** The first end-to-end implementation target.
   It matches the primary setting: RGB-only, no precise CAD at inference, and
   posed reference images for novel-object initialization. Reproducibility takes
   precedence over recency during system development.
2. **Base model — the frozen pose components used in one experiment.** A run's
   base model is the chosen global estimator plus local tracker. It is not itself
   the proposed contribution. The same recoverability estimator and routing
   protocol must be evaluated with multiple compatible base models.
3. **Comparison methods — external methods reported in the paper.** Gen6D,
   NOPE, DVMNet, OrienPose, RGBTrack, MegaPose, and possibly FoundationPose are
   compared under explicitly reported, non-identical input assumptions.

MegaPose is a verified **CAD-enabled auxiliary baseline**, not the primary
development baseline. Its existing integration remains useful for testing the
estimator-agnostic claim and the recovery infrastructure.

## Primary system contract

```text
Posed RGB reference images
        ↓
GlobalPoseEstimator (Find)
        ↓
LocalPoseTracker (Follow)
        ↓
RecoverabilityEstimator (Judge)
        ↓ failure
GlobalPoseEstimator (Find Again)
```

All global adapters return a metric object-to-camera `4 x 4` SE(3) transform,
runtime, optional scalar score, and method-specific diagnostics. The tracking
pipeline receives only the neutral `ReferenceObject`; CAD-aware adapters must
explicitly require `ModelReference` and may not silently become the default.

```text
GlobalPoseEstimator
├── Gen6DAdapter       development baseline; implementation started
├── NOPEAdapter        planned external comparison
├── DVMNetAdapter      planned stronger estimator
├── OrienPoseAdapter   planned recent strong comparison
└── MegaPoseRGBAdapter implemented CAD auxiliary baseline
```

Adapters listed as planned are not fake placeholders and are not advertised as
working before their unchanged upstream reproductions succeed.

## Input-condition matrix

| Method | RGB-only | Precise CAD at inference | Reference input | GT initial pose | Native recovery | Project role |
|---|---:|---:|---|---:|---:|---|
| Gen6D | Yes | No | Multiple posed RGB views / SfM onboarding | No | No | Development baseline |
| NOPE | Yes | No | Single RGB reference | No | No | Comparison/global estimator |
| DVMNet | Yes | No | RGB reference/query pair | No | No | Comparison/global estimator |
| OrienPose | Yes | No | Single RGB reference | No | No | Recent strong comparison |
| RGBTrack | Yes | Yes | No | Protocol-dependent | Yes | CAD-enabled tracking comparison |
| MegaPose | Yes | Yes | No | No | No | CAD-enabled auxiliary baseline |
| FoundationPose | Setting-dependent | CAD or reference views; released paths differ | Optional few-shot references | No | Tracking supported | Conditional comparison |
| Ours | Yes | No in primary protocol | Posed or single RGB reference | No | Yes | Proposed framework |

Metrics must always be accompanied by this input-condition disclosure. A method
using precise CAD, depth, GT masks, or GT initialization must not be declared
universally stronger from a single aggregate number.

## Required development order

1. Reproduce official Gen6D on its released GenMOP/LINEMOD protocol.
2. Freeze the upstream revision and wrap Gen6D as `GlobalPoseEstimator`.
3. Prepare leakage-free posed reference images for the selected tracking data.
4. Attach a CAD-free-compatible local tracker.
5. Generate failures and empirical recoverable-basin trials.
6. Train the reliability/recoverability estimator.
7. Close autonomous initialization, failure awareness, and relocalization.
8. Add NOPE and DVMNet.
9. Add OrienPose without making the project depend on successful reproduction.
10. Re-run the same framework with MegaPose as a CAD-enabled auxiliary study.

The intended claim is not that our single-frame estimator is stronger. It is:

> Recoverability-aware global/local switching improves autonomous long-term
> tracking across more than one global estimator.

## Current reproducibility status

- MegaPose auxiliary adapter: real GPU smoke and rollout complete.
- Gen6D official source: pinned at `0884c7713fee14b471d587fda80e35105f6718e8`.
- Gen6D neutral adapter: implemented against upstream `build`/`predict` APIs.
- Gen6D pretrained weights and official evaluation: not yet reproduced.
- Gen6D YCB-V posed-reference onboarding: not yet prepared.
- NOPE/DVMNet/OrienPose adapters: intentionally not implemented yet.

Therefore MegaPose results remain engineering validation only, and no Gen6D
accuracy or speed claim is currently made.
