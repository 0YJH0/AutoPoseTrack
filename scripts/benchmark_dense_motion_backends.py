#!/usr/bin/env python3
"""Benchmark and compare NumPy CPU and Torch CUDA dense motion backends."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from autoposetrack.motion.dense import (
    NumpyDenseMotionBackend,
    TorchCudaMotionBackend,
)
from autoposetrack.motion.types import SaliencyConfig


def _measure(backend, flow, matrix, saliency, warmup, iterations):
    for _ in range(warmup):
        backend.process(flow, matrix, saliency)
    runtimes = []
    output = None
    for _ in range(iterations):
        started = time.perf_counter()
        output = backend.process(flow, matrix, saliency)
        runtimes.append(time.perf_counter() - started)
    return output, runtimes


def _runtime_summary(values):
    return {
        "mean_ms": statistics.mean(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "min_ms": min(values) * 1000.0,
        "max_ms": max(values) * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--threshold-type",
        choices=("mad", "percentile", "fixed"),
        default="mad",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-consistency", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    flow = rng.normal(0.0, 2.0, (args.height, args.width, 2)).astype(np.float32)
    matrix = np.asarray(
        [[1.001, -0.002, 2.5], [0.001, 0.999, -1.5], [1e-6, -2e-6, 1.0]],
        dtype=np.float64,
    )
    saliency = SaliencyConfig(threshold_type=args.threshold_type)
    cpu_output, cpu_times = _measure(
        NumpyDenseMotionBackend(), flow, matrix, saliency, args.warmup, args.iterations
    )
    report = {
        "shape": [args.height, args.width],
        "iterations": args.iterations,
        "warmup": args.warmup,
        "threshold_type": args.threshold_type,
        "cpu": _runtime_summary(cpu_times),
        "cuda_available": False,
    }
    try:
        cuda_backend = TorchCudaMotionBackend(args.device)
    except RuntimeError as error:
        if args.require_cuda:
            raise
        report["cuda_error"] = str(error)
    else:
        cuda_output, cuda_times = _measure(
            cuda_backend, flow, matrix, saliency, args.warmup, args.iterations
        )
        report["cuda_available"] = True
        report["cuda"] = _runtime_summary(cuda_times)
        report["speedup_mean"] = statistics.mean(cpu_times) / statistics.mean(
            cuda_times
        )
        consistency = {
            "background_flow_max_abs": float(
                np.max(np.abs(cpu_output.background_flow - cuda_output.background_flow))
            ),
            "residual_flow_max_abs": float(
                np.max(np.abs(cpu_output.residual_flow - cuda_output.residual_flow))
            ),
            "magnitude_max_abs": float(
                np.max(
                    np.abs(
                        cpu_output.residual_magnitude
                        - cuda_output.residual_magnitude
                    )
                )
            ),
            "threshold_abs": abs(cpu_output.threshold - cuda_output.threshold),
            "mask_agreement": float(
                np.mean(cpu_output.threshold_mask == cuda_output.threshold_mask)
            ),
        }
        report["consistency"] = consistency
        if args.check_consistency:
            if consistency["background_flow_max_abs"] > 1e-3:
                raise AssertionError("CUDA background flow differs from CPU")
            if consistency["residual_flow_max_abs"] > 1e-3:
                raise AssertionError("CUDA residual flow differs from CPU")
            if consistency["magnitude_max_abs"] > 1e-3:
                raise AssertionError("CUDA residual magnitude differs from CPU")
            if consistency["threshold_abs"] > 1e-3:
                raise AssertionError("CUDA threshold differs from CPU")
            if consistency["mask_agreement"] < 0.999:
                raise AssertionError("CUDA threshold mask differs from CPU")
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
