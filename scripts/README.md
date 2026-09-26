# Experiment entry points

Phase 0 intentionally contains no executable pipeline. Future scripts must be
thin, configuration-driven entry points into the `autoposetrack` package and
must fail explicitly when data, checkpoints, or dependencies are unavailable.

Available infrastructure entry points:

- `python -m scripts.check_environment`
- `python -m scripts.validate_config <config>`
- `python -m scripts.create_run <config>`
- `python -m scripts.train_reliability <training-config>`
- `python -m scripts.export_diagnostics outputs/<run-name>`
- `python -m scripts.run_ycbv_megapose ...`
- `bash scripts/run_megapose_headless.sh <command>` for Panda3D on headless hosts
- `python -m scripts.render_motion_proposals ...` for GT-free full-frame motion
  proposals and debug video
- `python -m scripts.evaluate_motion_proposals ...` for evaluator-only Recall@K
- `python -m scripts.benchmark_dense_motion_backends ...` for synchronized
  NumPy/Torch-CUDA consistency and latency
- `python -m scripts.test_motion_scenarios ...` for controlled camera/target motion
- `python -m scripts.run_ycbv_motion_proposals ...` for strictly adjacent public
  BOP-YCB-V frame pairs
