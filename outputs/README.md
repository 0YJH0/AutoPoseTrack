# Generated outputs

Each experiment must follow the artifact contract in `docs/research_plan.md`.
Generated contents are ignored by Git; this file documents the expected layout:

```text
outputs/<run-name>/
├── config.yaml
├── manifest.json
├── model.npz
├── metrics.json
├── history.json
└── logs/
    ├── train.log
    ├── metrics.jsonl
    ├── events.jsonl
    └── failure.json       # only when the run fails
```

Use `python -m scripts.export_diagnostics outputs/<run-name>` to create a
small, reviewable report under `reports/`; do not commit this output directory.
