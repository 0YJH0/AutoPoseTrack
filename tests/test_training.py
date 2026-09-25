import json

import numpy as np
import pytest

from autoposetrack.training import (
    FeatureDataset,
    LogisticRegressionGD,
    ReliabilityTrainer,
    SequenceSplit,
    split_by_sequence,
)


def make_dataset() -> FeatureDataset:
    features = np.array(
        [
            [-2.0, 0.1],
            [-1.0, 0.2],
            [1.0, 0.8],
            [2.0, 0.9],
            [-1.5, 0.1],
            [1.5, 0.9],
        ]
    )
    return FeatureDataset(
        features=features,
        labels=np.array([0, 0, 1, 1, 0, 1]),
        sequence_ids=np.array(["a", "a", "b", "b", "c", "c"]),
        frame_ids=np.arange(6),
        object_ids=np.full(6, 2),
        feature_names=("matching_score", "mask_iou"),
        metadata={"label_definition": "unit-test fixture", "modality": "rgb"},
    )


def test_feature_dataset_round_trip(tmp_path):
    dataset = make_dataset()
    target = tmp_path / "features.npz"
    dataset.save(target)
    loaded = FeatureDataset.load(target, required_features=dataset.feature_names)
    np.testing.assert_allclose(loaded.features, dataset.features)
    np.testing.assert_array_equal(loaded.labels, dataset.labels)
    assert loaded.metadata == dataset.metadata
    with pytest.raises(FileExistsError):
        dataset.save(target)


def test_sequence_split_is_deterministic_and_has_no_leakage():
    ids = np.array([f"sequence-{index // 2}" for index in range(40)])
    first = split_by_sequence(ids, 0.2, 0.2, seed=2027)
    second = split_by_sequence(ids, 0.2, 0.2, seed=2027)
    np.testing.assert_array_equal(first.train, second.train)
    first.validate(ids)


def test_sequence_split_detects_leakage():
    split = SequenceSplit(
        train=np.array([0]), validation=np.array([1]), test=np.array([], dtype=int)
    )
    with pytest.raises(ValueError, match="leakage"):
        split.validate(np.array(["same", "same"]))


def test_logistic_baseline_trains_and_checkpoint_is_safe_npz(tmp_path):
    dataset = make_dataset()
    split = SequenceSplit(
        train=np.array([0, 1, 2, 3]),
        validation=np.array([4, 5]),
        test=np.array([], dtype=int),
    )
    model = LogisticRegressionGD(learning_rate=0.1, epochs=200, l2=0.0)
    trainer = ReliabilityTrainer(model)
    result = trainer.run(dataset, split)
    assert result.train_metrics["accuracy_at_0.5"] == pytest.approx(1.0)
    assert result.validation_metrics is not None
    checkpoint = tmp_path / "model.npz"
    trainer.save_checkpoint(checkpoint, dataset.feature_names, {"seed": 2027})
    with np.load(checkpoint, allow_pickle=False) as payload:
        assert tuple(payload["feature_names"].tolist()) == dataset.feature_names
        assert json.loads(str(payload["metadata_json"]))["seed"] == 2027


def test_training_rejects_one_class_labels():
    model = LogisticRegressionGD()
    with pytest.raises(ValueError, match="both classes"):
        model.fit(np.ones((3, 2)), np.ones(3))


def test_feature_dataset_rejects_depth_features():
    dataset = make_dataset()
    invalid = FeatureDataset(
        features=dataset.features,
        labels=dataset.labels,
        sequence_ids=dataset.sequence_ids,
        frame_ids=dataset.frame_ids,
        object_ids=dataset.object_ids,
        feature_names=("matching_score", "depth_residual"),
        metadata={"modality": "rgb"},
    )
    with pytest.raises(ValueError, match="depth features"):
        invalid.validate()
