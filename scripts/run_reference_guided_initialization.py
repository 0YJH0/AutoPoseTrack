#!/usr/bin/env python3
"""Run lightweight full-frame proposals + DINOv2 target discovery."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from autoposetrack.datasets import load_hot3d_discovery, load_ycbineoat_discovery
from autoposetrack.evaluation.reference_discovery import evaluate_discovery
from autoposetrack.proposal import (
    FastSamProposalGenerator,
    ProposalFilterConfig,
    ProposalInvocation,
    SamProposalGenerator,
)
from autoposetrack.reference import (
    CropConfig,
    DINOv2ReferenceEncoder,
    GlobalReferenceMatcher,
    MatchingConfig,
    ReferenceFrame,
    ReferenceMemory,
)
from autoposetrack.visualization.reference_discovery import discovery_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ycbineoat", "hot3d"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--object-id")
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/autoposetrack/reference_initialization_fastsam.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-index", action="append", type=int)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-stride", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--proposal-size", type=int, choices=(640, 768))
    parser.add_argument("--max-proposals", type=int, choices=range(30, 51))
    parser.add_argument(
        "--mode",
        choices=("initialization", "relocalization"),
        default="initialization",
        help="Full-frame proposals are deliberately unavailable in tracking mode.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    debug_dir = args.output / "debug"
    debug_dir.mkdir()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.dataset == "hot3d" and args.object_id is None:
        raise ValueError("--object-id is required for HOT3D")
    frames = (
        load_ycbineoat_discovery(args.input, args.object_id)
        if args.dataset == "ycbineoat"
        else load_hot3d_discovery(args.input, args.object_id, args.stream_id)
    )
    if len(frames) < 2:
        raise ValueError("dataset has fewer than two target-visible frames")
    reference_indices = args.reference_index or [0]
    reference_frames = [
        ReferenceFrame(
            f"{args.dataset}:{frames[index].frame_id}",
            frames[index].rgb(),
            frames[index].load_gt_mask(),
        )
        for index in reference_indices
    ]
    ref_cfg = cfg["reference"]
    proposal_cfg = cfg["proposal"]
    match_cfg = cfg["matching"]
    backend = str(proposal_cfg.get("backend", "sam")).lower()
    if args.proposal_size is not None:
        if backend != "fastsam":
            raise ValueError("--proposal-size is supported only by FastSAM")
        proposal_cfg["inference_size"] = args.proposal_size
    if args.max_proposals is not None:
        proposal_cfg["max_proposals"] = args.max_proposals
    crop = CropConfig(
        float(ref_cfg["bbox_expansion"]),
        int(ref_cfg["crop_size"]),
        str(ref_cfg["background"]),
    )
    encoder = DINOv2ReferenceEncoder(
        model_name=str(ref_cfg["encoder"]),
        device=str(ref_cfg["device"]),
        batch_size=int(ref_cfg.get("batch_size", 32)),
        use_fp16=bool(ref_cfg.get("use_fp16", True)),
    )
    memory = ReferenceMemory.build(reference_frames, encoder, crop)
    filter_config = ProposalFilterConfig(
        int(proposal_cfg["min_area"]),
        float(proposal_cfg["max_area_ratio"]),
        float(proposal_cfg.get("stability_threshold", 0.0)),
        float(proposal_cfg["duplicate_iou"]),
        int(proposal_cfg["max_proposals"]),
    )
    if backend == "fastsam":
        generator = FastSamProposalGenerator(
            str(proposal_cfg["checkpoint"]),
            str(proposal_cfg["device"]),
            int(proposal_cfg.get("inference_size", 640)),
            float(proposal_cfg.get("confidence", 0.25)),
            float(proposal_cfg.get("nms_iou", 0.9)),
            bool(proposal_cfg.get("use_fp16", True)),
            filter_config,
        )
    elif backend == "sam":
        generator = SamProposalGenerator(
            str(proposal_cfg["checkpoint"]),
            str(proposal_cfg["model_type"]),
            str(proposal_cfg["device"]),
            int(proposal_cfg["points_per_side"]),
            float(proposal_cfg["pred_iou_threshold"]),
            float(proposal_cfg["stability_threshold"]),
            filter_config,
        )
    else:
        raise ValueError(f"unsupported proposal backend: {backend}")
    matcher = GlobalReferenceMatcher(
        encoder,
        crop,
        MatchingConfig(
            str(match_cfg["aggregation"]),
            int(match_cfg["top_k_average"]),
            float(match_cfg["acceptance_score"]),
            float(match_cfg["acceptance_margin"]),
        ),
    )
    reference_set = set(reference_indices)
    query_indices = [
        index
        for index in range(args.query_start, len(frames), args.query_stride)
        if index not in reference_set
    ][: args.max_queries]
    rows = []
    for query_index in query_indices:
        frame = frames[query_index]
        rgb = frame.rgb()
        started = time.perf_counter()
        proposal_started = time.perf_counter()
        proposals = generator.generate(rgb, ProposalInvocation(args.mode))
        proposal_time = time.perf_counter() - proposal_started
        match_started = time.perf_counter()
        result = matcher.match(rgb, proposals, memory)
        match_time = time.perf_counter() - match_started
        gt_mask = frame.load_gt_mask()  # evaluator boundary: loaded only after inference
        metrics = evaluate_discovery(proposals, result, gt_mask, float(cfg["evaluation"]["mask_iou_threshold"]))
        rows.append({
            "dataset": args.dataset, "frame_id": frame.frame_id, "object_id": frame.object_id,
            "num_proposals": len(proposals), "selected_proposal_id": result.matches[0].proposal_id if result.matches else "",
            "proposal_score": result.best_score, "second_best_score": result.second_best_score, "score_margin": result.score_margin,
            "accepted": result.success, "oracle_proposal_recall": metrics.oracle_proposal_recall,
            "matching_eligible": metrics.matching_eligible, "matching_correct": metrics.matching_correct,
            "recall_at_1": metrics.recall_at_1, "recall_at_3": metrics.recall_at_3, "recall_at_5": metrics.recall_at_5,
            "selected_mask_iou": metrics.selected_mask_iou, "best_proposal_iou": metrics.best_proposal_iou,
            "failure_type": metrics.failure_type, "runtime_proposal_s": proposal_time,
            "runtime_reference_match_s": match_time, "runtime_total_s": time.perf_counter() - started,
        })
        panel = discovery_panel(reference_frames[0].image, reference_frames[0].mask, rgb, proposals, result, gt_mask)
        Image.fromarray(panel).save(debug_dir / f"frame_{frame.frame_id:06d}.jpg")
        with (args.output / f"frame_{frame.frame_id:06d}_matches.json").open("w", encoding="utf-8") as file:
            json.dump([match.__dict__ for match in result.matches], file, indent=2)
        top_matches = result.matches[:5]
        np.savez_compressed(
            args.output / f"frame_{frame.frame_id:06d}_topk_masks.npz",
            proposal_ids=np.asarray([item.proposal_id for item in top_matches], dtype=np.int32),
            scores=np.asarray([item.score for item in top_matches], dtype=np.float32),
            masks=np.stack([proposals[item.proposal_id].mask for item in top_matches])
            if top_matches
            else np.empty((0, *rgb.shape[:2]), dtype=bool),
        )
    if not rows:
        raise ValueError("no query frames selected")
    with (args.output / "per_frame.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    eligible = [row for row in rows if row["matching_eligible"]]
    summary = {
        "dataset": args.dataset, "object_id": frames[0].object_id, "reference_indices": reference_indices,
        "mode": args.mode, "proposal_backend": backend,
        "proposal_max_count": int(proposal_cfg["max_proposals"]),
        "proposal_inference_size": proposal_cfg.get("inference_size"),
        "reference_encoder": str(ref_cfg["encoder"]),
        "reference_batch_size": int(ref_cfg.get("batch_size", 32)),
        "reference_fp16": bool(ref_cfg.get("use_fp16", True)),
        "num_queries": len(rows), "oracle_proposal_recall": float(np.mean([row["oracle_proposal_recall"] for row in rows])),
        "matching_accuracy_given_oracle": float(np.mean([row["matching_correct"] for row in eligible])) if eligible else None,
        **{f"recall_at_{k}": float(np.mean([row[f"recall_at_{k}"] for row in rows])) for k in (1, 3, 5)},
        "mean_selected_mask_iou": float(np.mean([row["selected_mask_iou"] for row in rows])),
        "mean_best_proposal_iou": float(np.mean([row["best_proposal_iou"] for row in rows])),
        "mean_runtime_proposal_s": float(np.mean([row["runtime_proposal_s"] for row in rows])),
        "mean_runtime_reference_match_s": float(np.mean([row["runtime_reference_match_s"] for row in rows])),
        "mean_runtime_total_s": float(np.mean([row["runtime_total_s"] for row in rows])),
        "failure_counts": dict(Counter(row["failure_type"] for row in rows)),
        "gt_usage": "reference_mask_and_post_inference_evaluator_only",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
