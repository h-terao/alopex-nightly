"""Stores the environ information."""
try:
    import torch  # NOQA

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
