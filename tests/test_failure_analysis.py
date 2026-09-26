from scripts.analyze_tracking_failures import analyze


def test_failure_analysis_joins_and_reports_threshold_free_trends() -> None:
    predictions = [
        {
            "sequence_id": "1",
            "frame_id": str(i),
            "object_id": "5",
            "pose_error_m": str(i / 10),
        }
        for i in range(4)
    ]
    attributes = [
        {
            "sequence_id": "1",
            "frame_id": str(i),
            "object_id": "5",
            "visibility_fraction": str(1 - i / 10),
            "bbox_area_ratio": str(i + 1),
            "visible_area_ratio": str(i + 1),
            "translation_delta": str(i),
            "rotation_delta_deg": str(i),
            "nearest_reference_angle": "nan",
            "sharpness_score": str(i),
            "num_instances": "1",
            "max_bbox_overlap": "0",
        }
        for i in range(4)
    ]
    joined, summary = analyze(predictions, attributes, "pose_error_m")
    assert len(joined) == 4
    assert summary["threshold_free"] is True
    assert summary["attribute_trends"]["visibility_fraction"][
        "spearman_error_correlation"
    ] == -1.0
