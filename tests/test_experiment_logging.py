import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

from autoposetrack.training import FeatureDataset
from autoposetrack.utils.experiment_logging import ExperimentLogger


def test_logger_writes_metrics_events_and_failure(tmp_path):
    run_dir = tmp_path / "run"
    logger = ExperimentLogger(run_dir)
    logger.log_event("run_started", seed=2027)
    logger.log_metrics(3, "train", {"loss": 0.25})
    try:
        raise RuntimeError("expected failure")
    except RuntimeError as error:
        logger.record_failure(error)
    finally:
        logger.close()

    event = json.loads((run_dir / "logs/events.jsonl").read_text().splitlines()[0])
    metric = json.loads((run_dir / "logs/metrics.jsonl").read_text().splitlines()[0])
    failure = json.loads((run_dir / "logs/failure.json").read_text())
    assert event["event"] == "run_started"
    assert metric["metrics"]["loss"] == pytest.approx(0.25)
    assert failure["type"] == "RuntimeError"
    assert "expected failure" in failure["traceback"]


def test_diagnostic_export_excludes_checkpoint(tmp_path):
    run_dir = tmp_path / "cloud-run"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("experiment: test\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text('{"loss": 0.1}\n', encoding="utf-8")
    (run_dir / "model.npz").write_bytes(b"large checkpoint placeholder")
    (logs / "train.log").write_text("line one\nline two\n", encoding="utf-8")
    reports = tmp_path / "reports"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.export_diagnostics",
            str(run_dir),
            "--reports-root",
            str(reports),
            "--log-tail-lines",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    exported = reports / "cloud-run"
    assert (exported / "metrics.json").is_file()
    assert (exported / "logs/train.tail.log").read_text() == "line two\n"
    assert not (exported / "model.npz").exists()


def test_training_cli_writes_complete_log_contract(tmp_path):
    dataset_path = tmp_path / "features.npz"
    FeatureDataset(
        features=np.asarray([[0.0], [1.0], [0.2], [0.8]], dtype=np.float64),
        labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
        sequence_ids=np.asarray(["s0", "s0", "s1", "s1"]),
        frame_ids=np.arange(4, dtype=np.int64),
        object_ids=np.ones(4, dtype=np.int64),
        feature_names=("rgb_confidence",),
        metadata={"modality": "rgb"},
    ).save(dataset_path)
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config = {
        "experiment": {"name": "logger_smoke", "seed": 2027},
        "data": {"feature_dataset": str(dataset_path)},
        "split": {"validation_fraction": 0.0, "test_fraction": 0.0},
        "model": {
            "name": "logistic_regression_gd",
            "learning_rate": 0.1,
            "epochs": 3,
            "l2": 0.0,
        },
        "logging": {"console_every_epochs": 1},
        "output": {"directory": str(run_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "scripts.train_reliability", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (run_dir / "model.npz").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert len((run_dir / "logs/metrics.jsonl").read_text().splitlines()) == 4
    events = [
        json.loads(line)
        for line in (run_dir / "logs/events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "run_completed"
    assert not (run_dir / "logs/failure.json").exists()
