# Gen6D development-baseline reproduction

## Status

- Official source pinned as `third_party/Gen6D` at commit
  `0884c7713fee14b471d587fda80e35105f6718e8`.
- Project adapter implemented against the upstream `estimator.build()` and
  `estimator.predict()` boundary.
- Pretrained weights and official GenMOP/LINEMOD metrics are **not yet
  reproduced**.

The upstream repository is GPL-3.0 and states that its old OneDrive links are
broken. Its README points to a replacement Google Drive folder for pretrained
detector/selector/refiner weights and processed evaluation data. Review the
weights/data terms separately before redistribution. Do not commit downloaded
assets into this repository.

## Expected upstream data layout

```text
third_party/Gen6D/data/
├── model/
│   ├── detector_pretrain/model_best.pth
│   ├── selector_pretrain/model_best.pth
│   └── refiner_pretrain/model_best.pth
├── GenMOP/
└── LINEMOD/
```

## Reproduction gate

Run the unchanged upstream evaluations from the Gen6D environment:

```bash
cd third_party/Gen6D
python eval.py --cfg configs/gen6d_pretrain.yaml --object_name genmop/tformer
python eval.py --cfg configs/gen6d_pretrain.yaml --object_name linemod/cat
```

Archive stdout, package/CUDA versions, GPU, runtime, produced poses, and
visualizations. Compare ADD-0.1d and Prj-5 with the paper/repository before
running the AutoPoseTrack adapter.

## Project adapter contract

`Gen6DAdapter` requires a validated `ReferenceObject` and these metadata keys:

```python
backend_metadata={
    "gen6d_database": "custom/my_object",
    "translation_scale_to_m": 0.001,
}
```

The scale must come from onboarding/alignment, never from query/test GT. Gen6D
returns a 3x4 pose in its database coordinate scale; the adapter converts it to
the project's metric 4x4 object-to-camera SE(3) convention.

The adapter preserves selector score, selector margin, detected relative scale,
selected in-plane angle, refinement-step count, and runtime for the later
recoverability model.

## YCB-V onboarding gate

The current BOP test subset alone is insufficient for a clean Gen6D primary
experiment because test images must not be reused as posed references. Prepare
references from a disjoint real training/onboarding sequence, recover their
camera poses and sparse object geometry, align the object coordinate frame and
metric scale, then freeze the reference manifest before testing.
