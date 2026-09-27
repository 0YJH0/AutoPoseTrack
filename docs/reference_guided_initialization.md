# Reference-guided autonomous initialization

This stage answers one bounded question: can sparse reference images identify
an unknown rigid target among class-agnostic full-frame masks? It does not use
query-frame GT for inference and does not yet claim temporal mask or 6D pose
tracking.

## Pipeline and boundaries

1. A fixed reference RGB and its mask build `ReferenceMemory`.
2. The official SAM automatic-mask generator processes the complete query RGB.
3. The same masked-crop preprocessing is applied to references and proposals.
4. DINOv2 ViT-S/14 produces L2-normalized CLS descriptors.
5. Cosine matching ranks every proposal and logs the full ranking, best score,
   second score, margin, and matched reference id.
6. Only after ranking is complete does the evaluator load the query GT mask.

Query GT bbox, mask, and pose are never arguments of `ProposalGenerator`,
`ReferenceEncoder`, or `GlobalReferenceMatcher`.

## Dataset protocols

### YCBInEOAT

- Sequence: `cracker_box_reorient`.
- Reference: fixed index supplied by `--reference-index`; default is the first
  available frame and its GT mask.
- Queries: deterministic index range/stride, excluding reference indices.
- Recommended pilot uses a reference in the motion window and queries the same
  fixed window; test-set inspection must not be used to choose a closer view.

### HOT3D-Clips

- Clip: official Aria `clip-001852`, RGB stream `214-1`.
- Target: BOP object id 29, the RGB-visible independently moving target.
- Reference: first target-visible frame by default.
- Modal masks in `objects.json` are decoded only for the reference and
  post-inference evaluator.

## Run

Download the official SAM ViT-B checkpoint to
`data/models/sam/sam_vit_b_01ec64.pth`. DINOv2 weights are downloaded by the
official Torch Hub loader into `TORCH_HOME` on first use.

```bash
docker compose --profile motion build motion
docker compose --profile motion run --rm motion bash

python -m scripts.run_reference_guided_initialization \
  --dataset ycbineoat \
  --input data/YCBInEOAT/cracker_box_reorient \
  --reference-index 117 --query-start 127 --query-stride 10 \
  --max-queries 10 \
  --output outputs/ycbineoat_reference_init_v1

python -m scripts.run_reference_guided_initialization \
  --dataset hot3d \
  --input outputs/hot3d_clip_001852_motion_cuda_v1/extracted_clip \
  --object-id 29 --stream-id 214-1 \
  --reference-index 0 --query-start 10 --query-stride 10 \
  --max-queries 10 \
  --output outputs/hot3d_reference_init_v1
```

## Outputs

- `summary.json`: aggregate oracle proposal recall, conditional matching
  accuracy, Recall@1/3/5, mask IoU, and failure counts.
- `per_frame.csv`: scores, margins, timings, metrics, and failure class.
- `frame_*_matches.json`: complete ranked proposal scores.
- `frame_*_topk_masks.npz`: actual Top-5 boolean candidate masks, ids, and
  matching scores.
- `debug/*.jpg`: masked reference beside full-frame Top-5 proposals; green GT
  is explicitly evaluator-only.

The critical decomposition is:

- **Oracle proposal recall**: at least one SAM mask overlaps GT above the fixed
  IoU threshold.
- **Matching accuracy given oracle**: DINO selects a correct mask, conditioned
  on SAM having produced one.

This prevents a SAM coverage failure from being misreported as a DINO identity
failure.

## Initial integration result

The first deterministic three-query pilot produced:

| Dataset | Oracle proposal recall | Top-1 / Top-3 | Mean selected mask IoU |
|---|---:|---:|---:|
| YCBInEOAT | 100% | 66.67% / 100% | 0.594 |
| HOT3D | 100% | 100% / 100% | 0.859 |

This is a smoke-scale result only. It proves that both adapters and the
GT-isolated evaluation path run end to end; it is not sufficient for a paper
claim or threshold tuning.
