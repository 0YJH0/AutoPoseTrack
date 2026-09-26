"""Global pose-estimator adapters."""
from .gen6d import Gen6DAdapter
from .megapose import MegaPoseRGBAdapter, MegaPoseRGBEstimator

__all__ = ["Gen6DAdapter", "MegaPoseRGBAdapter", "MegaPoseRGBEstimator"]
