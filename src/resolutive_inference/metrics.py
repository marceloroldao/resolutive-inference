"""Evaluation metrics with explicit input validation."""

import numpy as np


def mean_negative_log_likelihood(probabilities: np.ndarray) -> float:
    """Mean negative log likelihood of probabilities assigned to observed outcomes."""
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0 or np.any(values <= 0) or np.any(values > 1):
        raise ValueError("probabilities must be non-empty and in (0, 1]")
    return float(-np.mean(np.log(values)))
