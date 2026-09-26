"""Dense background-flow, residual, magnitude, and threshold backends."""

from .base import DenseMotionBackend, DenseMotionOutput
from .numpy_backend import NumpyDenseMotionBackend
from .torch_cuda import TorchCudaMotionBackend

__all__ = [
    "DenseMotionBackend",
    "DenseMotionOutput",
    "NumpyDenseMotionBackend",
    "TorchCudaMotionBackend",
]
