"""Observation likelihood functions."""

import numpy as np


def gaussian_log_likelihood(
    observation: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """Return diagonal-Gaussian log likelihood for each state."""
    residual = observation[None, :] - means
    terms = np.log(2.0 * np.pi * variances) + residual**2 / variances
    return -0.5 * np.sum(terms, axis=1)
