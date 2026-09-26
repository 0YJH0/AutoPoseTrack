# Motion proposal CUDA v1 report

This compact report records the reproducible metrics from the RTX 4060 dense
backend benchmark, two controlled motion scenarios, and all strictly adjacent
frame pairs in the downloaded public BOP-YCB-V subset.

Large generated artifacts are intentionally kept out of Git:

- `outputs/dense_motion_backend_benchmark_rtx4060.json`
- `outputs/motion_scenarios_torch_cuda_v1/`
- `outputs/ycbv_motion_adjacent_torch_cuda_v1/`

The BOP-YCB-V result demonstrates a method boundary rather than success: static
objects share camera-induced background motion and are suppressed by correct
compensation. Motion proposals must be complemented by full-frame reference
appearance search for autonomous initialization and relocalization.
