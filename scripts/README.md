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
