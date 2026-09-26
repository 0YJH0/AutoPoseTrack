import numpy as np

from autoposetrack.pose.global_estimator.gen6d import Gen6DAdapter


def test_gen6d_diagnostics_preserve_selector_information():
    diagnostics = Gen6DAdapter._diagnostics(
        {
            "sel_scores": np.array([0.1, 0.8, 0.3]),
            "det_scale_r2q": np.array(1.2),
            "sel_angle_r2q": 0.25,
            "refine_poses": [np.eye(4), np.eye(4), np.eye(4)],
        }
    )
    assert diagnostics["selector_max_score"] == 0.8
    assert diagnostics["selector_score_margin"] == 0.5
    assert diagnostics["detected_scale"] == 1.2
    assert diagnostics["selected_inplane_angle"] == 0.25
    assert diagnostics["refinement_steps"] == 2
