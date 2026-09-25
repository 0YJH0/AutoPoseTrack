#!/usr/bin/env python3
"""Train the transparent reliability baseline from a generated feature table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from autoposetrack.training import (
    FeatureDataset,
    LogisticRegressionGD,
    ReliabilityTrainer,
    split_by_sequence,
)
from autoposetrack.utils.reproducibility import build_manifest, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    with args.config.expanduser().resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    dataset = FeatureDataset.load(Path(config["data"]["feature_dataset"]))
    split = split_by_sequence(
        dataset.sequence_ids,
        validation_fraction=float(config["split"]["validation_fraction"]),
        test_fraction=float(config["split"]["test_fraction"]),
        seed=seed,
    )
    model_config = config["model"]
    model = LogisticRegressionGD(
        learning_rate=float(model_config["learning_rate"]),
        epochs=int(model_config["epochs"]),
        l2=float(model_config["l2"]),
    )
    trainer = ReliabilityTrainer(model)
    result = trainer.run(dataset, split)
    output = Path(config["output"]["directory"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = build_manifest(Path.cwd())
    trainer.save_checkpoint(
        output / "model.npz",
        dataset.feature_names,
        {"config": config, "dataset_metadata": dict(dataset.metadata), "manifest": manifest},
    )
    metrics = {
        "train": result.train_metrics,
        "validation": result.validation_metrics,
        "test": result.test_metrics,
    }
    with (output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "history.json").open("x", encoding="utf-8") as handle:
        json.dump(result.history, handle)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()

