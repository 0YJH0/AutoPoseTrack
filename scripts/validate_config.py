#!/usr/bin/env python3
"""Validate a resolved experiment configuration without running a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoposetrack.state_manager.manager import StateManagerConfig
from autoposetrack.utils.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    state = config.raw["state_manager"]
    StateManagerConfig(
        track_threshold=float(state["track_threshold"]),
        lost_threshold=float(state["lost_threshold"]),
        consecutive_lost_frames=int(state["consecutive_lost_frames"]),
    ).validate()
    print(f"valid configuration: {config.name} (seed={config.seed})")


if __name__ == "__main__":
    main()

