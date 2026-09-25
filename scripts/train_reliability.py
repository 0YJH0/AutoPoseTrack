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
from autoposetrack.utils.experiment_logging import ExperimentLogger
from autoposetrack.utils.reproducibility import build_manifest, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    with args.config.expanduser().resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output = Path(config["output"]["directory"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logger = ExperimentLogger(output)

    try:
        manifest = build_manifest(Path.cwd())
        with (output / "config.yaml").open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        with (output / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        seed = int(config["experiment"]["seed"])
        seed_everything(seed)
        logger.log_event("run_started", seed=seed, config=str(args.config.resolve()))
        dataset = FeatureDataset.load(Path(config["data"]["feature_dataset"]))
        logger.log_event(
            "dataset_loaded",
            rows=len(dataset.labels),
            features=list(dataset.feature_names),
            positive_rate=float(dataset.labels.mean()),
        )
        split = split_by_sequence(
            dataset.sequence_ids,
            validation_fraction=float(config["split"]["validation_fraction"]),
            test_fraction=float(config["split"]["test_fraction"]),
            seed=seed,
        )
        logger.log_event(
            "split_created",
            train_rows=len(split.train),
            validation_rows=len(split.validation),
            test_rows=len(split.test),
        )
        model_config = config["model"]
        model = LogisticRegressionGD(
            learning_rate=float(model_config["learning_rate"]),
            epochs=int(model_config["epochs"]),
            l2=float(model_config["l2"]),
        )
        trainer = ReliabilityTrainer(model)
        log_every = int(config.get("logging", {}).get("console_every_epochs", 10))
        if log_every <= 0:
            raise ValueError("logging.console_every_epochs must be positive")

        def epoch_callback(epoch: int, values: dict[str, float]) -> None:
            logger.log_metrics(epoch, "train_epoch", values)
            if epoch == 0 or (epoch + 1) % log_every == 0:
                logger.info("epoch=%d metrics=%s", epoch + 1, values)

        result = trainer.run(dataset, split, callback=epoch_callback)
        trainer.save_checkpoint(
            output / "model.npz",
            dataset.feature_names,
            {
                "config": config,
                "dataset_metadata": dict(dataset.metadata),
                "manifest": manifest,
            },
        )
        metrics = {
            "train": result.train_metrics,
            "validation": result.validation_metrics,
            "test": result.test_metrics,
        }
        for split_name, values in metrics.items():
            if values is not None:
                logger.log_metrics(model.epochs, split_name, values)
        with (output / "metrics.json").open("x", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=False)
            handle.write("\n")
        with (output / "history.json").open("x", encoding="utf-8") as handle:
            json.dump(result.history, handle)
            handle.write("\n")
        logger.log_event("run_completed", checkpoint="model.npz")
        logger.info("artifacts=%s", output)
    except BaseException as error:
        logger.record_failure(error)
        raise
    finally:
        logger.close()
    print(output)


if __name__ == "__main__":
    main()
