"""Both task families share the exact same numerical buffer and lifecycle API."""
from causal_features.adapter import FeatureAdapter, open_buffers, close_buffers

__all__ = ['FeatureAdapter', 'open_buffers', 'close_buffers']
