# Cloud training, logging, and GitHub diagnosis

## Logging contract

Every training run uses a unique output directory. Console output is mirrored to
`logs/train.log`; per-step metrics and lifecycle events are written as JSONL and
flushed immediately. If training raises an exception, `logs/failure.json`
contains its type, message, UTC timestamp, and traceback.

The logger is local-only. It does not transmit data or credentials and does not
hide failures. Cloud launchers should preserve `outputs/` on persistent storage.

## Start a cloud run

```bash
git clone git@github.com:0YJH0/AutoPoseTrack.git
cd AutoPoseTrack
export FEATURE_DATA_ROOT=/persistent/features
docker compose build training
docker compose run --rm training \
  python -m scripts.train_reliability \
  configs/experiments/train_reliability_logreg.yaml \
  2>&1 | tee cloud-launch.log
```

The feature dataset path and output directory in the resolved config must point
to mounted paths. Use a new run name/directory for every launch; resuming and
checkpoint rotation are not implemented yet.

## Monitor

```bash
tail -F outputs/<run>/logs/train.log
tail -F outputs/<run>/logs/metrics.jsonl
nvidia-smi
docker stats
```

For longer jobs, a cloud scheduler should capture container exit code and retain
the entire run directory even when the process fails.

## Export only diagnostic evidence

Generated runs are intentionally ignored by Git. Export a small report:

```bash
python -m scripts.export_diagnostics outputs/<run>
```

The report contains available configuration, environment manifest, aggregate
metrics, structured metrics/events, failure traceback, and the last 500 log
lines. It excludes checkpoint weights, full histories, images, videos, and large
per-frame tables. Review it for private paths or secrets before publishing.

Then explicitly push it:

```bash
git add reports/<run>
git commit -m "Add diagnostics for <run>"
git push origin main
```

After the report is on GitHub, this coding environment can pull/read the report
and diagnose convergence, class imbalance, leakage, numerical errors, crashes,
configuration drift, and host/GPU mismatches. Large artifacts should remain in
object storage and be referenced by checksum/URI in the report.

## What not to upload to GitHub

- datasets or RGB frames;
- checkpoints and optimizer states;
- credentials, tokens, `.env`, or cloud metadata;
- full videos or large per-frame tables;
- third-party weights with redistribution restrictions.

GitHub is suitable for source, resolved configs, small metrics, failure traces,
and diagnostic summaries—not as the primary experiment artifact store.

