# Open questions and decision gates

Last updated: 2026-09-26

Questions are recorded here instead of being silently resolved in code.

## Blocking before Phase 1

1. **Gen6D reproduction assets:** Obtain the official pretrained detector,
   selector and refiner weights plus an official GenMOP/LINEMOD evaluation set.
2. **License review:** Review Gen6D code, nested dependencies, weights and data;
   separately retain the existing MegaPose auxiliary-baseline review.
3. **Reference onboarding:** Define a leakage-free source of posed YCB-V RGB
   references and a metric SfM scale. Test frames may never serve as references.

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
8. How is Gen6D's SfM reconstruction aligned and scaled to the metric object
   frame? Record the transform; do not infer scale from test GT.

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
