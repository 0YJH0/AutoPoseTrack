import numpy as np
import pytest

from autoposetrack.motion.dense import (
    NumpyDenseMotionBackend,
    TorchCudaMotionBackend,
)
from autoposetrack.motion.types import SaliencyConfig


@pytest.mark.parametrize("threshold_type", ["mad", "percentile", "fixed"])
def test_numpy_dense_backend_shapes_and_threshold(threshold_type):
    rng = np.random.default_rng(17)
    flow = rng.normal(0, 1, (24, 32, 2)).astype(np.float32)
    matrix = np.asarray(
        [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]
    )
    config = SaliencyConfig(
        threshold_type=threshold_type,
        fixed_threshold=1.25,
        minimum_threshold=0.0,
    )
    output = NumpyDenseMotionBackend().process(flow, matrix, config)
    assert output.background_flow.shape == flow.shape
    assert output.residual_flow.shape == flow.shape
    assert output.residual_magnitude.shape == flow.shape[:2]
    assert output.threshold_mask.dtype == np.bool_
    np.testing.assert_allclose(output.background_flow[..., 0], 2.0, atol=1e-5)
    np.testing.assert_allclose(output.background_flow[..., 1], -1.0, atol=1e-5)
    assert output.threshold >= 0.0


def test_torch_cuda_matches_numpy_when_cuda_is_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(2027)
    flow = rng.normal(0, 2, (72, 96, 2)).astype(np.float32)
    matrix = np.asarray(
        [[1.001, -0.002, 2.5], [0.001, 0.999, -1.5], [1e-6, -2e-6, 1.0]]
    )
    saliency = SaliencyConfig(threshold_type="mad", minimum_threshold=0.0)
    cpu = NumpyDenseMotionBackend().process(flow, matrix, saliency)
    cuda = TorchCudaMotionBackend().process(flow, matrix, saliency)
    np.testing.assert_allclose(cuda.background_flow, cpu.background_flow, atol=2e-4)
    np.testing.assert_allclose(cuda.residual_flow, cpu.residual_flow, atol=2e-4)
    np.testing.assert_allclose(
        cuda.residual_magnitude, cpu.residual_magnitude, atol=2e-4
    )
    assert abs(cuda.threshold - cpu.threshold) < 2e-4
    assert np.mean(cuda.threshold_mask == cpu.threshold_mask) > 0.999
