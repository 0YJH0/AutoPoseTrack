# Reference matching failure analysis

Every query is assigned exactly one first-stage outcome:

| Label | Meaning | Primary evidence |
|---|---|---|
| `proposal_failure` | No SAM proposal reaches the fixed GT mask-IoU threshold | best proposal IoU |
| `matching_failure` | A correct proposal exists but DINO ranks another first | complete score ranking and oracle set |
| `acceptance_rejection` | DINO ranks a correct proposal first but score/margin verification rejects it | best/second score and margin |
| `success` | Correct top-1 proposal passes acceptance | selected IoU and acceptance |

Viewpoint, occlusion, and distractor are analysis attributes, not inferred from
an arbitrary score threshold. Add them only when the dataset provides a
reproducible annotation or evaluator-only measurement.

## Required inspection order

1. Check oracle proposal recall. If low, inspect SAM masks and proposal filters;
   do not tune DINO.
2. Condition on oracle-positive frames and check matching accuracy. If low,
   compare masked crop, bbox crop, multiple references, and local patch matches.
3. Condition on correct top-1 and inspect acceptance rejection. Set acceptance
   thresholds on a separate validation sequence, never the reported test clip.
4. Report runtime separately for proposal generation and reference matching.

Do not silently replace a failed full-frame proposal with GT bbox/mask. Such a
run is a separate oracle ablation, not autonomous initialization.
