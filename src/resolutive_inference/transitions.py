"""State-transition utilities."""

import numpy as np


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    """Normalize non-negative weights into a probability vector."""
    if values.ndim != 1 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("probability weights must be a finite, non-negative vector")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("probability weights must have positive mass")
    return values / total


def predict_state(posterior: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Apply a row-stochastic transition matrix to a state posterior."""
    return normalize_probabilities(posterior @ transition)
