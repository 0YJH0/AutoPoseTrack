"""Evidence-first audit of a classic scene-wise BOP dataset split."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


NUMERIC_ATTRIBUTES = (
    "visibility_fraction",
    "invisible_fraction",
    "occlusion_ratio",
    "truncation_ratio",
    "bbox_area_ratio",
    "visible_area_ratio",
    "translation_delta",
    "rotation_delta_deg",
    "translation_velocity",
    "rotation_velocity_deg",
    "frame_id_delta",
    "nearest_reference_rotation_deg",
    "nearest_reference_translation",
    "sharpness_proxy",
    "num_visible_objects",
    "num_instances",
    "bbox_overlap_count",
    "max_bbox_iou",
)


@dataclass(frozen=True)
class AuditConfig:
    dataset_root: Path
    output_dir: Path
    split: str = "test"
    seed: int = 2027
    visual_samples: int = 24
    reference_manifest: Optional[Path] = None


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rgb_path(scene_dir: Path, frame_id: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = scene_dir / "rgb" / f"{frame_id:06d}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No RGB image for frame {frame_id} in {scene_dir}")


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(float(bbox[2]), 0.0) * max(float(bbox[3]), 0.0)


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = map(float, a)
    bx1, by1, bw, bh = map(float, b)
    ax2, ay2 = ax1 + max(aw, 0.0), ay1 + max(ah, 0.0)
    bx2, by2 = bx1 + max(bw, 0.0), by1 + max(bh, 0.0)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return intersection / union if union > 0.0 else 0.0


def _rotation_delta_deg(r_prev: np.ndarray, r_now: np.ndarray) -> float:
    relative = r_prev.T @ r_now
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _sharpness_proxy(image: Image.Image, bbox: Sequence[float]) -> float:
    """Variance of a discrete grayscale Laplacian; not a physical blur value."""
    x, y, width, height = map(int, bbox)
    left, top = max(x, 0), max(y, 0)
    right = min(x + max(width, 1), image.width)
    bottom = min(y + max(height, 1), image.height)
    if right - left < 3 or bottom - top < 3:
        return math.nan
    crop = np.asarray(image.crop((left, top, right, bottom)).convert("L"), dtype=float)
    laplacian = (
        -4.0 * crop[1:-1, 1:-1]
        + crop[:-2, 1:-1]
        + crop[2:, 1:-1]
        + crop[1:-1, :-2]
        + crop[1:-1, 2:]
    )
    return float(np.var(laplacian))


def _load_reference_bank(
    path: Optional[Path],
) -> Dict[int, List[Tuple[str, np.ndarray]]]:
    if path is None:
        return {}
    payload = _json(path)
    bank: Dict[int, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    entries = payload.get("references", payload)
    if not isinstance(entries, list):
        raise ValueError("reference manifest must be a list or contain 'references'")
    for item in entries:
        pose = np.asarray(item["object_to_camera"], dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("reference object_to_camera must be 4x4")
        reference_id = str(item.get("reference_id", len(bank[int(item["object_id"])])))
        bank[int(item["object_id"])].append((reference_id, pose))
    return dict(bank)


def _reference_gap(
    pose: np.ndarray, references: Sequence[Tuple[str, np.ndarray]]
) -> Tuple[float, float, Optional[str], float]:
    if not references:
        return math.nan, math.nan, None, math.nan
    gaps = [
        (
            _rotation_delta_deg(ref[:3, :3], pose[:3, :3]),
            float(np.linalg.norm(ref[:3, 3] - pose[:3, 3])),
            reference_id,
        )
        for reference_id, ref in references
    ]
    nearest = min(gaps, key=lambda item: (item[0], item[1]))
    return nearest[0], nearest[1], nearest[2], float(np.mean([gap[0] for gap in gaps]))


def _pose(gt: Mapping[str, Any]) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(gt["cam_R_m2c"], dtype=float).reshape(3, 3)
    pose[:3, 3] = np.asarray(gt["cam_t_m2c"], dtype=float) / 1000.0
    return pose


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result.append(numeric)
    return np.asarray(result, dtype=float)


def _statistics(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    values = _finite(row.get(field) for row in rows)
    missing = len(rows) - len(values)
    result: Dict[str, Any] = {
        "attribute": field,
        "count": int(len(values)),
        "missing": int(missing),
    }
    for name in ("min", "q25", "median", "q75", "max", "mean", "std"):
        result[name] = math.nan
    if len(values):
        result.update(
            {
                "min": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )
    return result


def _group_statistics(
    rows: Sequence[Mapping[str, Any]], key: str
) -> List[Dict[str, Any]]:
    groups: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    output = []
    for value, group in sorted(groups.items()):
        visibility = _finite(row["visibility_fraction"] for row in group)
        bbox_area = _finite(row["bbox_area_ratio"] for row in group)
        output.append(
            {
                key: value,
                "num_annotations": len(group),
                "num_unique_frames": len({row["frame_uid"] for row in group}),
                "visibility_mean": float(np.mean(visibility)),
                "visibility_median": float(np.median(visibility)),
                "bbox_area_ratio_mean": float(np.mean(bbox_area)),
                "bbox_area_ratio_median": float(np.median(bbox_area)),
            }
        )
    return output


def _draw_visuals(
    rows: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    count: int,
    seed: int,
) -> List[str]:
    count = min(max(count, 0), len(rows))
    if not count:
        return []
    selected = random.Random(seed).sample(list(rows), count)
    visual_dir = output_dir / "figures" / "visual_audit"
    visual_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    thumbs = []
    for row in selected:
        image = Image.open(dataset_root / str(row["rgb_path"])).convert("RGB")
        draw = ImageDraw.Draw(image)
        bbox_fields = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
        x, y, width, height = [float(row[k]) for k in bbox_fields]
        draw.rectangle((x, y, x + width, y + height), outline=(255, 40, 40), width=3)
        label = (
            f"s{int(row['scene_id']):06d} f{int(row['frame_id']):06d} "
            f"obj{row['object_id']} vis={float(row['visibility_fraction']):.2f}"
        )
        draw.rectangle((0, 0, min(image.width, 430), 22), fill=(0, 0, 0))
        draw.text((4, 4), label, fill=(255, 255, 255))
        name = (
            f"s{int(row['scene_id']):06d}_f{int(row['frame_id']):06d}_"
            f"g{int(row['gt_id']):02d}.jpg"
        )
        path = visual_dir / name
        image.save(path, quality=90)
        paths.append(str(path.relative_to(output_dir)))
        thumb = image.copy()
        thumb.thumbnail((320, 240))
        thumbs.append(thumb)
    cols = 4
    rows_count = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 320, rows_count * 240), color=(30, 30, 30))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 320, (index // cols) * 240))
    sheet.save(output_dir / "figures" / "visual_audit_contact_sheet.jpg", quality=90)
    return paths


def _plot_distributions(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    fields = (
        "visibility_fraction",
        "bbox_area_ratio",
        "translation_delta",
        "rotation_delta_deg",
        "sharpness_proxy",
        "max_bbox_iou",
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for field in fields:
        values = _finite(row[field] for row in rows)
        if not len(values):
            continue
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.hist(values, bins=min(40, max(10, int(np.sqrt(len(values))))))
        axis.set_xlabel(field)
        axis.set_ylabel("annotation count")
        axis.set_title(f"Natural distribution: {field}")
        fig.tight_layout()
        path = figure_dir / f"{field}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path.relative_to(output_dir)))
    return written


def audit_bop_dataset(config: AuditConfig) -> Dict[str, Any]:
    """Audit all GT object instances in one BOP split and write immutable artifacts."""
    split_dir = config.dataset_root / config.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"BOP split not found: {split_dir}")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    reference_bank = _load_reference_bank(config.reference_manifest)
    records: List[Dict[str, Any]] = []
    previous_pose: Dict[Tuple[int, int], Tuple[int, np.ndarray]] = {}
    scene_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    for scene_dir in scene_dirs:
        scene_id = int(scene_dir.name)
        scene_gt = _json(scene_dir / "scene_gt.json")
        scene_info = _json(scene_dir / "scene_gt_info.json")
        for frame_key in sorted(scene_gt, key=int):
            frame_id = int(frame_key)
            gt_items = scene_gt[frame_key]
            info_items = scene_info[frame_key]
            if len(gt_items) != len(info_items):
                raise ValueError(f"GT/info length mismatch: {scene_id}/{frame_id}")
            rgb_path = _rgb_path(scene_dir, frame_id)
            with Image.open(rgb_path) as opened:
                image = opened.convert("RGB")
            image_area = float(image.width * image.height)
            num_visible = len(
                {
                    int(gt_item["obj_id"])
                    for gt_item, info_item in zip(gt_items, info_items)
                    if int(info_item["px_count_visib"]) > 0
                }
            )
            visible_boxes = [item["bbox_visib"] for item in info_items]
            for gt_id, (gt, info) in enumerate(zip(gt_items, info_items)):
                object_id = int(gt["obj_id"])
                pose = _pose(gt)
                key = (scene_id, object_id)
                translation_delta = rotation_delta = frame_delta = math.nan
                if key in previous_pose:
                    previous_frame, previous = previous_pose[key]
                    frame_delta = frame_id - previous_frame
                    translation_delta = float(
                        np.linalg.norm(pose[:3, 3] - previous[:3, 3])
                    )
                    rotation_delta = _rotation_delta_deg(previous[:3, :3], pose[:3, :3])
                previous_pose[key] = (frame_id, pose)
                overlaps = [
                    _bbox_iou(info["bbox_visib"], other)
                    for index, other in enumerate(visible_boxes)
                    if index != gt_id
                ]
                reference_gap = _reference_gap(pose, reference_bank.get(object_id, []))
                ref_rotation, ref_translation, ref_id, ref_rotation_mean = reference_gap
                visibility = float(info["visib_fract"])
                bbox = info["bbox_visib"]
                sharpness = _sharpness_proxy(image, bbox)
                source = {
                    "pose": "scene_gt.json",
                    "visibility_scale_bbox": "scene_gt_info.json",
                    "motion": "delta_between_available_BOP_frames",
                    "velocity": "missing_no_timestamp_or_fps",
                    "occlusion_truncation": "missing_not_separable_from_BOP_fields",
                    "reference_gap": (
                        str(config.reference_manifest)
                        if config.reference_manifest is not None
                        else "missing_no_reference_manifest"
                    ),
                    "sharpness_proxy": "RGB_crop_grayscale_laplacian_variance",
                    "clutter": "scene_gt_info_visible_bboxes",
                }
                records.append(
                    {
                        "dataset": config.dataset_root.name,
                        "split": config.split,
                        "scene_id": scene_id,
                        "sequence_id": f"{scene_id:06d}",
                        "frame_id": frame_id,
                        "frame_uid": f"{scene_id:06d}/{frame_id:06d}",
                        "gt_id": gt_id,
                        "object_id": object_id,
                        "rgb_path": str(rgb_path.relative_to(config.dataset_root)),
                        "image_width": image.width,
                        "image_height": image.height,
                        "gt_tx_m": float(pose[0, 3]),
                        "gt_ty_m": float(pose[1, 3]),
                        "gt_tz_m": float(pose[2, 3]),
                        "gt_tx": float(pose[0, 3]),
                        "gt_ty": float(pose[1, 3]),
                        "gt_tz": float(pose[2, 3]),
                        "visibility_fraction": visibility,
                        "invisible_fraction": 1.0 - visibility,
                        "occlusion_ratio": math.nan,
                        "truncation_ratio": math.nan,
                        "px_count_all": int(info["px_count_all"]),
                        "px_count_valid": int(info["px_count_valid"]),
                        "px_count_visib": int(info["px_count_visib"]),
                        "bbox_x": int(bbox[0]),
                        "bbox_y": int(bbox[1]),
                        "bbox_w": int(bbox[2]),
                        "bbox_h": int(bbox[3]),
                        "bbox_area_ratio": _bbox_area(info["bbox_obj"]) / image_area,
                        "visible_area_ratio": (
                            float(info["px_count_visib"]) / image_area
                        ),
                        "translation_delta": translation_delta,
                        "rotation_delta_deg": rotation_delta,
                        "translation_velocity": math.nan,
                        "rotation_velocity_deg": math.nan,
                        "linear_velocity": math.nan,
                        "angular_velocity_deg_s": math.nan,
                        "frame_id_delta": frame_delta,
                        "nearest_reference_rotation_deg": ref_rotation,
                        "nearest_reference_translation": ref_translation,
                        "nearest_reference_angle": ref_rotation,
                        "nearest_reference_id": ref_id,
                        "reference_angle_min": ref_rotation,
                        "reference_angle_mean": ref_rotation_mean,
                        "sharpness_proxy": sharpness,
                        "sharpness_score": sharpness,
                        "blur_proxy": sharpness,
                        "num_visible_objects": num_visible,
                        "num_instances": len(gt_items),
                        "bbox_overlap_count": sum(value > 0.0 for value in overlaps),
                        "max_bbox_iou": max(overlaps, default=0.0),
                        "num_overlapping_objects": sum(
                            value > 0.0 for value in overlaps
                        ),
                        "max_bbox_overlap": max(overlaps, default=0.0),
                        "mean_bbox_overlap": (
                            float(np.mean(overlaps)) if overlaps else 0.0
                        ),
                        "attribute_source": json.dumps(source, sort_keys=True),
                    }
                )
    if not records:
        raise ValueError(f"No BOP annotations found in {split_dir}")
    distributions = [_statistics(records, field) for field in NUMERIC_ATTRIBUTES]
    object_stats = _group_statistics(records, "object_id")
    sequence_stats = _group_statistics(records, "sequence_id")
    _write_csv(config.output_dir / "per_frame_attributes.csv", records)
    _write_csv(config.output_dir / "per_object_statistics.csv", object_stats)
    _write_csv(config.output_dir / "per_sequence_statistics.csv", sequence_stats)
    _write_csv(config.output_dir / "attribute_distribution.csv", distributions)
    visual_paths = _draw_visuals(
        records,
        config.dataset_root,
        config.output_dir,
        config.visual_samples,
        config.seed,
    )
    plots = _plot_distributions(records, config.output_dir)
    frame_ids_by_scene: Dict[int, List[int]] = defaultdict(list)
    for row in records:
        frame_ids_by_scene[int(row["scene_id"])].append(int(row["frame_id"]))
    unique_frames = {
        (int(row["scene_id"]), int(row["frame_id"])) for row in records
    }
    summary = {
        "schema_version": 1,
        "dataset_root": str(config.dataset_root.resolve()),
        "split": config.split,
        "seed": config.seed,
        "num_scenes": len(scene_dirs),
        "num_frames": len(unique_frames),
        "num_annotations": len(records),
        "num_objects": len({int(row["object_id"]) for row in records}),
        "object_ids": sorted({int(row["object_id"]) for row in records}),
        "frames_per_scene": Counter(scene for scene, _ in unique_frames),
        "frame_id_step_distribution": dict(
            sorted(
                Counter(
                    current - previous
                    for ids in frame_ids_by_scene.values()
                    for previous, current in zip(
                        sorted(set(ids))[:-1], sorted(set(ids))[1:]
                    )
                ).items()
            )
        ),
        "is_continuous_tracking_split": False,
        "continuity_reason": (
            "BOP19 test target subset contains sparse sampled keyframes; no "
            "timestamps/FPS are available in scene annotations."
        ),
        "missing_attributes": {
            "occlusion_ratio": (
                "BOP visibility does not separate occlusion from truncation"
            ),
            "truncation_ratio": (
                "BOP visibility does not separate occlusion from truncation"
            ),
            "velocities": "timestamps/FPS unavailable",
            "reference_gap": (
                None if reference_bank else "posed reference manifest unavailable"
            ),
        },
        "visual_audit_samples": visual_paths,
        "distribution_figures": plots,
    }
    (config.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return summary
