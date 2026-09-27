# Reference-guided autonomous initialization

This stage answers one bounded question: can sparse reference images identify
an unknown rigid target among class-agnostic full-frame masks? It does not use
query-frame GT for inference and does not yet claim temporal mask or 6D pose
tracking.

## Pipeline and boundaries

1. A fixed reference RGB and its mask build `ReferenceMemory`.
2. FastSAM-s (default) processes the complete query RGB at 640 px; the original
   SAM backend remains available as a quality/speed baseline.
3. The same masked-crop preprocessing is applied to references and proposals.
4. DINOv2 ViT-S/14 encodes proposal crops in FP16 batches of 32 and produces
   L2-normalized CLS descriptors.
5. Cosine matching ranks every proposal and logs the full ranking, best score,
   second score, margin, and matched reference id.
6. Only after ranking is complete does the evaluator load the query GT mask.

Query GT bbox, mask, and pose are never arguments of `ProposalGenerator`,
`ReferenceEncoder`, or `GlobalReferenceMatcher`.

The command accepts only `--mode initialization` and `--mode relocalization`.
There is intentionally no tracking mode: normal frames must use the local
tracker, so the full-frame proposal/matching cost is paid only while finding
the object or recovering from failure.

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

Download the FastSAM-s checkpoint to `data/models/fastsam/FastSAM-s.pt`.
DINOv2 weights are downloaded by the official Torch Hub loader into
`TORCH_HOME` on first use. To reproduce the slower original SAM baseline,
download SAM ViT-B to `data/models/sam/sam_vit_b_01ec64.pth` and select
`configs/autoposetrack/reference_initialization.yaml`.

The expected FastSAM-s SHA-256 is
`c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844`.

```bash
docker compose --profile motion build motion
docker compose --profile motion run --rm motion bash

python -m scripts.run_reference_guided_initialization \
  --dataset ycbineoat \
  --input data/YCBInEOAT/cracker_box_reorient \
  --config configs/autoposetrack/reference_initialization_fastsam.yaml \
  --proposal-size 640 --max-proposals 50 \
  --mode initialization \
  --reference-index 117 --query-start 127 --query-stride 10 \
  --max-queries 10 \
  --output outputs/ycbineoat_reference_init_v1

python -m scripts.run_reference_guided_initialization \
  --dataset hot3d \
  --input outputs/hot3d_clip_001852_motion_cuda_v1/extracted_clip \
  --config configs/autoposetrack/reference_initialization_fastsam.yaml \
  --proposal-size 768 --max-proposals 30 \
  --mode relocalization \
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

## FastSAM smoke result and boundary

On the local RTX 4060, one YCBInEOAT query at 640/Top-50 was recovered with
mask IoU 0.837. FastSAM took 1.42 s and the batched FP16 DINOv2-S match took
0.16 s (1.59 s total, including first-inference warm-up). On the tested HOT3D
query, neither 640 nor 768 reached the 0.5 proposal-IoU criterion; increasing
resolution improved best proposal IoU only from 0.186 to 0.200. Lowering the
FastSAM confidence to 0.10 did not recover the small hand-held target.

Therefore 640/Top-50 is the default fast path, but FastSAM proposal coverage
must be logged separately and must not be described as universally replacing
SAM. HOT3D needs either the retained high-recall SAM fallback, an
EfficientViT-SAM adapter, or a reference-conditioned ROI proposal stage.
`--proposal-confidence` is exposed for diagnostics, not as a demonstrated fix.
