# Open questions and decision gates

Last updated: 2026-09-25

Questions are recorded here instead of being silently resolved in code.

## Blocking before Phase 1

1. **GPU visibility:** Why does NVML report that GPU access is blocked in WSL2?
   Confirm Windows driver/WSL GPU passthrough, GPU model and VRAM, `nvidia-smi`
   inside WSL, and GPU access inside Docker.
2. **Repository ownership:** This directory contains a `.git` directory but Git
   reports it is not a repository. Should it be initialized as a new repository,
   or is missing metadata expected to be restored? Do not initialize or overwrite
   it without resolving this.
3. **License acceptance:** Is FoundationPose's NVIDIA Source Code License
   acceptable for the institution, artifact release plan, and intended use?
   Review dependencies and model/data licenses separately.
4. **Dataset availability:** Which YCB-Video distribution is locally available
   (original layout, BOP YCB-V, or both), and where? Paths must enter only through
   configuration/environment, never source code.

## Protocol decisions before full YCB-Video runs

5. Which masks define the first controlled baseline: GT visible masks, official
   upstream masks, or a fixed predicted segmenter? Recommendation: reproduce the
   official FoundationPose protocol first, label it precisely, then add one fixed
   predicted-mask protocol for autonomous experiments.
6. Is the unit of evaluation a single annotated object track or all visible
   object instances per sequence? Define target selection and re-entry behavior.
7. How are symmetric objects declared, and which exact ADD/ADD-S/AUC convention
   and maximum threshold will be primary? Cross-check BOP and common YCB-V
   conventions before freezing tables.
8. What constitutes a missing/invalid depth observation, and may the global or
   local module use RGB-only fallback? Recommendation: no silent fallback;
   declare separate RGB-D and RGB protocols.
9. What upstream commit/container is the reproducibility anchor? Pin it only
   after the unchanged demo succeeds, then archive its digest.

## Decisions deferred until Phase 2/3 evidence

10. Failure threshold and temporal tolerance used as offline labels.
11. Recoverability rollout horizon, success threshold, perturbation composition
    order, and sampling distribution.
12. Observable inputs retained for the reliability model; do not assume visual,
    geometry, temporal, or matching features are useful before analysis.
13. State thresholds, hysteresis length, relocalization retry/backoff, and how
    unsuccessful global hypotheses are verified without GT.
14. Calibration method and operating-point selection. Thresholds must be chosen
    on validation sequences, never on test sequences.

