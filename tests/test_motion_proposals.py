from dataclasses import replace

import numpy as np
import pytest

from autoposetrack.evaluation import evaluate_proposals, summarize_proposal_metrics
from autoposetrack.motion import FullFrameMotionProposer, MotionProposalConfig
from autoposetrack.motion.background import (
    BackgroundMotionEstimator,
    background_flow_from_model,
)
from autoposetrack.motion.components import expand_bbox
from autoposetrack.motion.flow import OpticalFlowEstimator
from autoposetrack.motion.types import (
    BackgroundConfig,
    MorphologyConfig,
    ProposalConfig,
    SaliencyConfig,
    TemporalConfig,
)


class FixedFlow(OpticalFlowEstimator):
    def __init__(self, flow):
        self.flow = np.asarray(flow, dtype=np.float32)

    def estimate(self, prev_rgb, curr_rgb):
        return self.flow.copy()


def config(background="homography", temporal=False):
    return MotionProposalConfig(
        background=BackgroundConfig(
            model=background,
            sample_step=4,
            ransac_threshold=0.5,
            min_matches=12,
            min_inlier_ratio=0.5,
            min_coverage_ratio=0.2,
            max_median_reprojection_error=0.5,
            fallback="affine",
        ),
        saliency=SaliencyConfig(
            threshold_type="fixed", fixed_threshold=0.5, minimum_threshold=0.0
        ),
        morphology=MorphologyConfig(enabled=False),
        proposal=ProposalConfig(
            min_area_pixels=4,
            bbox_expand_scale=1.2,
            merge_iou=0.1,
            merge_centroid_distance_ratio=0.03,
            top_k=10,
        ),
        temporal=TemporalConfig(enabled=temporal, history_length=3),
    )


def blank(shape=(64, 80, 3)):
    return np.zeros(shape, dtype=np.uint8)


def test_static_scene_produces_no_proposals():
    flow = np.zeros((64, 80, 2), dtype=np.float32)
    proposer = FullFrameMotionProposer(config(), FixedFlow(flow))
    result = proposer.propose(blank(), blank(), 1)
    assert result.proposals == ()
    assert result.background_motion_model.valid
    assert np.max(result.motion_score_map) < 1e-5


@pytest.mark.parametrize("model_type", ["affine", "homography"])
def test_camera_translation_is_removed_by_global_model(model_type):
    flow = np.zeros((64, 80, 2), dtype=np.float32)
    flow[..., 0] = 3.0
    flow[..., 1] = -2.0
    estimator = BackgroundMotionEstimator(config(background=model_type).background)
    model = estimator.estimate(flow)
    predicted = background_flow_from_model(model, flow.shape[:2])
    assert model.valid
    assert model.used_type == model_type
    assert np.mean(np.linalg.norm(flow - predicted, axis=2)) < 1e-3


@pytest.mark.parametrize("camera_flow", [(0.0, 0.0), (3.0, -2.0)])
def test_moving_object_survives_static_or_moving_camera(camera_flow):
    flow = np.zeros((64, 80, 2), dtype=np.float32)
    flow[..., 0] = camera_flow[0]
    flow[..., 1] = camera_flow[1]
    flow[20:35, 30:48, 0] += 4.0
    proposer = FullFrameMotionProposer(config(), FixedFlow(flow))
    result = proposer.propose(blank(), blank(), 1)
    assert result.proposals
    best = result.proposals[0]
    assert best.bbox_xyxy[0] <= 30 <= best.bbox_xyxy[2]
    assert best.bbox_xyxy[1] <= 20 <= best.bbox_xyxy[3]
    assert np.mean(result.motion_score_map[20:35, 30:48]) > 3.5


def test_border_expansion_is_clipped():
    assert expand_bbox((0.0, 0.0, 8.0, 10.0), 1.5, 80, 64) == (
        0.0,
        0.0,
        10.0,
        12.5,
    )


def test_tiny_strong_component_is_preserved():
    flow = np.zeros((32, 32, 2), dtype=np.float32)
    flow[1:3, 1:3, 0] = 5.0
    cfg = config(background="none")
    cfg = replace(
        cfg,
        proposal=replace(
            cfg.proposal,
            min_area_pixels=20,
            preserve_small_strong=True,
            small_strong_factor=1.5,
        ),
    )
    result = FullFrameMotionProposer(cfg, FixedFlow(flow)).propose(
        blank((32, 32, 3)), blank((32, 32, 3)), 1
    )
    assert len(result.proposals) == 1
    assert result.proposals[0].area == 4


def test_temporal_persistence_increases_score_without_gating_first_frame():
    flow = np.zeros((32, 32, 2), dtype=np.float32)
    flow[10:18, 10:18, 0] = 3.0
    proposer = FullFrameMotionProposer(
        config(background="none", temporal=True), FixedFlow(flow)
    )
    first = proposer.propose(blank((32, 32, 3)), blank((32, 32, 3)), 1)
    second = proposer.propose(blank((32, 32, 3)), blank((32, 32, 3)), 2)
    assert len(first.proposals) == 1
    assert first.proposals[0].temporal_score > 0.0
    assert second.proposals[0].temporal_score > first.proposals[0].temporal_score
    assert second.proposals[0].tracklet_id == first.proposals[0].tracklet_id


def test_proposal_metrics_are_isolated_from_proposer():
    flow = np.zeros((32, 32, 2), dtype=np.float32)
    flow[10:20, 10:20, 0] = 3.0
    result = FullFrameMotionProposer(
        config(background="none"), FixedFlow(flow)
    ).propose(blank((32, 32, 3)), blank((32, 32, 3)), 1)
    metrics = evaluate_proposals(result.proposals, (10.0, 10.0, 20.0, 20.0), 0.5)
    summary = summarize_proposal_metrics([metrics])
    assert metrics.recall_at_k[1]
    assert metrics.center_recalled
    assert summary["recall_at_1"] == 1.0


def test_farneback_static_rgb_integration():
    pytest.importorskip("cv2")
    proposer = FullFrameMotionProposer(config(background="none"))
    image = np.random.default_rng(7).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    result = proposer.propose(image, image.copy(), 1)
    assert result.proposals == ()
    assert float(np.mean(result.motion_score_map)) < 5e-3
