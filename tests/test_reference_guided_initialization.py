import numpy as np

from autoposetrack.datasets import decode_hot3d_rle
from autoposetrack.evaluation.reference_discovery import evaluate_discovery
from autoposetrack.proposal import SegmentationProposal
from autoposetrack.reference import (
    CropConfig,
    GlobalReferenceMatcher,
    MatchingConfig,
    ReferenceEncoder,
    ReferenceFrame,
    ReferenceMemory,
)


class MeanColorEncoder(ReferenceEncoder):
    def encode_global(self, images):
        values = np.stack([image.mean(axis=(0, 1)) for image in images]).astype(np.float32)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)


def proposal(mask, source_id):
    ys, xs = np.nonzero(mask)
    return SegmentationProposal(
        mask,
        (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
        int(mask.sum()),
        1.0,
        1.0,
        source_id,
    )


def test_reference_matcher_selects_same_color_target_and_reports_oracle():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    target = np.zeros((32, 48), dtype=bool); target[8:24, 5:18] = True
    distractor = np.zeros_like(target); distractor[8:24, 28:43] = True
    image[target] = (220, 20, 20); image[distractor] = (20, 20, 220)
    reference = ReferenceFrame("red", image, target)
    crop = CropConfig(expansion=1.0, size=32, background="zero")
    encoder = MeanColorEncoder()
    memory = ReferenceMemory.build([reference], encoder, crop)
    proposals = [proposal(distractor, 0), proposal(target, 1)]
    result = GlobalReferenceMatcher(
        encoder, crop, MatchingConfig(acceptance_score=0.0, acceptance_margin=0.01)
    ).match(image, proposals, memory)
    metrics = evaluate_discovery(proposals, result, target, 0.5)
    assert result.matches[0].proposal_id == 1
    assert result.success
    assert metrics.oracle_proposal_recall
    assert metrics.matching_correct
    assert metrics.recall_at_1
    assert metrics.selected_mask_iou == 1.0


def test_metrics_distinguish_proposal_and_matching_failures():
    gt = np.zeros((16, 16), dtype=bool); gt[2:6, 2:6] = True
    wrong = np.zeros_like(gt); wrong[10:14, 10:14] = True
    encoder = MeanColorEncoder(); crop = CropConfig(expansion=1.0, size=16, background="zero")
    image = np.zeros((16, 16, 3), dtype=np.uint8); image[gt] = 100; image[wrong] = 100
    memory = ReferenceMemory.build([ReferenceFrame("target", image, gt)], encoder, crop)
    proposals = [proposal(wrong, 0)]
    result = GlobalReferenceMatcher(encoder, crop, MatchingConfig(acceptance_score=-1.0, acceptance_margin=0.0)).match(image, proposals, memory)
    assert evaluate_discovery(proposals, result, gt).failure_type == "proposal_failure"


def test_hot3d_start_length_rle_decode():
    mask = decode_hot3d_rle({"height": 3, "width": 4, "rle": [2, 3, 9, 2]})
    expected = np.zeros(12, dtype=bool); expected[2:5] = True; expected[9:11] = True
    np.testing.assert_array_equal(mask.ravel(), expected)
