#!/usr/bin/env python3
"""Create a traceable empty run directory; does not execute inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoposetrack.utils.config import load_experiment_config
from autoposetrack.utils.reproducibility import build_manifest, seed_everything
from autoposetrack.utils.run_artifacts import RunArtifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    seed_everything(config.seed)
    artifacts = RunArtifacts(Path(config.raw["output"]["root"]), config.name)
    artifacts.create(config.raw, build_manifest(args.repo_root.resolve()))
    print(artifacts.path)


if __name__ == "__main__":
    main()

