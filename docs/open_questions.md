# Open questions and decision gates

Last updated: 2026-09-25

Questions are recorded here instead of being silently resolved in code.

## Blocking before Phase 1

1. **Pinned baseline:** Which reviewed MegaPose commit and immutable Docker image
   digest will anchor Phase 1? Record any adapter patches separately.
2. **License review:** Review MegaPose's nested renderer, weights, asset, and
   dataset licenses even though project-owned code is Apache-2.0/MIT compatible.
3. **Dataset availability:** Which YCB-Video distribution is locally available
   (original layout, BOP YCB-V, or both), and where? Paths must enter only through
   configuration/environment, never source code.

## Protocol decisions before full YCB-Video runs

4. Which RGB detections define the first controlled baseline: GT boxes, official
   detections, or a fixed RGB detector/segmenter? Recommendation: reproduce with
   GT boxes first and label it precisely, then add one fixed predicted protocol.
5. Is the unit of evaluation a single annotated object track or all visible
   object instances per sequence? Define target selection and re-entry behavior.
6. How are symmetric objects declared, and which exact ADD/ADD-S/AUC convention
   and maximum threshold will be primary? Cross-check BOP and common YCB-V
   conventions before freezing tables.
7. Which mask protocol is used for silhouette reliability: no mask, GT mask as
   an oracle ablation, or a fixed RGB-derived mask? Never mix them silently.
8. How is metric object scale converted when MegaPose meshes use millimetres but
   the project boundary requires metres?

## Decisions deferred until Phase 2/3 evidence

9. Failure threshold and temporal tolerance used as offline labels.
10. Recoverability rollout horizon, success threshold, perturbation composition
    order, and sampling distribution.
11. Observable RGB inputs retained for the reliability model; do not assume visual,
    geometry, temporal, or matching features are useful before analysis.
12. State thresholds, hysteresis length, relocalization retry/backoff, and how
    unsuccessful global hypotheses are verified without GT.
13. Calibration method and operating-point selection. Thresholds must be chosen
    on validation sequences, never on test sequences.
