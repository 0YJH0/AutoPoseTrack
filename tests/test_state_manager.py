import pytest

from autoposetrack.state_manager import StateManager, StateManagerConfig, TrackingState


@pytest.fixture
def manager():
    return StateManager(StateManagerConfig(0.7, 0.3, 2))


def test_initialization_and_relocalization(manager):
    assert manager.state == TrackingState.UNINITIALIZED
    assert manager.request_global() == TrackingState.RELOCALIZING
    assert manager.global_result(True) == TrackingState.TRACKING


def test_hysteresis_requires_consecutive_low_scores(manager):
    manager.request_global()
    manager.global_result(True)
    assert manager.local_score(0.2) == TrackingState.UNCERTAIN
    assert manager.local_score(0.2) == TrackingState.LOST


def test_middle_score_is_uncertain_but_not_accumulated(manager):
    manager.request_global()
    manager.global_result(True)
    manager.local_score(0.2)
    assert manager.local_score(0.5) == TrackingState.UNCERTAIN
    assert manager.local_score(0.2) == TrackingState.UNCERTAIN

