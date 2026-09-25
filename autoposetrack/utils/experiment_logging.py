"""Cloud-friendly console, file, metric, event, and failure logging."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentLogger:
    """Write human-readable logs and machine-readable JSONL records.

    Every JSONL write is flushed immediately so useful diagnostics survive a
    preempted cloud instance. This class deliberately has no network side effect.
    """

    def __init__(self, run_dir: Path, console_level: int = logging.INFO):
        self.run_dir = run_dir.expanduser().resolve()
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.events_path = self.log_dir / "events.jsonl"
        self.failure_path = self.log_dir / "failure.json"

        name = f"autoposetrack.{self.run_dir.name}.{id(self)}"
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        formatter = logging.Formatter(
            "%(asctime)sZ | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S"
        )
        file_handler = logging.FileHandler(
            self.log_dir / "train.log", mode="x", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self._handlers = (file_handler, console_handler)

    def info(self, message: str, *args: Any) -> None:
        self.logger.info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self.logger.warning(message, *args)

    def log_metrics(
        self, step: int, split: str, metrics: Mapping[str, Optional[float]]
    ) -> None:
        record = {
            "timestamp": _timestamp(),
            "step": int(step),
            "split": split,
            "metrics": dict(metrics),
        }
        self._append_jsonl(self.metrics_path, record)

    def log_event(self, event: str, **payload: Any) -> None:
        record = {"timestamp": _timestamp(), "event": event, **payload}
        self._append_jsonl(self.events_path, record)
        self.logger.info("event=%s %s", event, json.dumps(payload, sort_keys=True))

    def record_failure(self, error: BaseException) -> None:
        payload = {
            "timestamp": _timestamp(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        with self.failure_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        self.logger.exception("training failed")

    def close(self) -> None:
        for handler in self._handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

    @staticmethod
    def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()

