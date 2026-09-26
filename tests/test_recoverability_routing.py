from pathlib import Path

import numpy as np
import pytest

from autoposetrack.contracts import (
    FrameObservation,
    ModelReference,
    PoseEstimate,
    PoseMode,
)
from autoposetrack.reliability import (
    CostSensitiveRouter,
    CostSensitiveRoutingConfig,
    RoutingAction,
    TemporalWindow,
)
from autoposetrack.rollout import (
    RecoverabilityAwareTracker,
    SE3Perturbation,
    make_perturbation_grid,
    perturb_pose_object_frame,
)


def _router(cooldown=0):
    return CostSensitiveRouter(
        CostSensitiveRoutingConfig(0.1, 2.0, 10.0, 0.7, 0.3, 2, cooldown)
    )


def test_cost_router_uses_hysteresis_and_expected_cost():
    router = _router()
    assert router.choose(0.9) is RoutingAction.LOCAL
    assert router.choose(0.2) is RoutingAction.WAIT
    assert router.choose(0.2) is RoutingAction.GLOBAL
    # Moderate risk is withheld rather than causing an expensive global call.
    assert router.choose(0.5) is RoutingAction.WAIT


def test_router_forces_global_without_pose_and_honours_cooldown():
    router = _router(cooldown=3)
    assert router.choose(0.9, has_pose=False) is RoutingAction.GLOBAL
    assert router.choose(0.1) is RoutingAction.WAIT
    assert router.choose(0.1) is RoutingAction.WAIT
    assert router.choose(0.1) is RoutingAction.GLOBAL


def test_temporal_window_is_causal_and_masked():
    window = TemporalWindow(("score", "jump"), history_frames=3)
    first = window.append({"score": 0.8, "jump": 0.1})
    np.testing.assert_allclose(first, [0, 0, 0, 0, 0.8, 0.1, 0, 0, 1])
    second = window.append({"score": 0.6, "jump": 0.2})
    np.testing.assert_allclose(second[-3:], [0, 1, 1])
    assert len(window.output_names) == len(second)


def test_perturbation_grid_is_deterministic_and_pose_is_valid():
    first = make_perturbation_grid([0, 10], [0, 0.01])
    second = make_perturbation_grid([0, 10], [0, 0.01])
    assert first == second
    assert (
        sum(
            item.rotation_deg_xyz == (0, 0, 0) and item.translation_m_xyz == (0, 0, 0)
            for item in first
        )
        == 1
    )
    changed = perturb_pose_object_frame(
        np.eye(4), SE3Perturbation("x", (0, 0, 90), (0.01, 0, 0))
    )
    assert np.linalg.det(changed[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(changed[:3, 3], [0.01, 0, 0])


class _PoseBackend:
    def __init__(self):
        self.global_calls = 0
        self.local_calls = 0

    def estimate(self, observation, model):
        self.global_calls += 1
        return PoseEstimate(np.eye(4), 1.0, PoseMode.GLOBAL, 1.0)

    def track(self, observation, previous, model):
        self.local_calls += 1
        return PoseEstimate(np.eye(4), 0.5, PoseMode.LOCAL, 0.1)

    def reset(self):
        pass


def test_closed_loop_routes_local_then_global():
    backend = _PoseBackend()
    probabilities = iter([0.9, 0.1, 0.1])
    tracker = RecoverabilityAwareTracker(
        backend,
        backend,
        _router(),
        lambda observation, estimate, history: next(probabilities),
        ModelReference(5, Path("mesh.ply"), 0.1),
    )
    observation = FrameObservation(
        "s",
        0,
        5,
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.eye(3),
        np.array([0, 0, 4, 4], dtype=float),
    )
    assert tracker.step(observation).action is RoutingAction.GLOBAL
    assert tracker.step(observation).action is RoutingAction.LOCAL
    assert tracker.step(observation).action is RoutingAction.WAIT
    assert tracker.step(observation).action is RoutingAction.GLOBAL
    assert backend.global_calls == 2
