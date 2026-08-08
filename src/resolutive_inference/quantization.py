"""Deterministic scalar quantization helpers."""

import numpy as np


def uniform_quantize(values: np.ndarray, levels: int, lower: float, upper: float) -> np.ndarray:
    """Quantize values to uniformly spaced reconstruction levels."""
    if levels < 2 or not lower < upper:
        raise ValueError("levels must be >= 2 and lower must be below upper")
    grid = (upper - lower) / (levels - 1)
    clipped = np.clip(np.asarray(values, dtype=float), lower, upper)
    return lower + np.rint((clipped - lower) / grid) * grid
