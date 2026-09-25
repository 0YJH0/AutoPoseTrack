"""Determinism and experiment-manifest helpers."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)


def _command_output(command: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def build_manifest(repo_root: Path) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"], repo_root),
        "git_dirty": bool(
            _command_output(["git", "status", "--porcelain"], repo_root)
        ),
        "gpu": _command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

