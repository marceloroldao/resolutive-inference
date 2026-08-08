"""Evaluation metrics with explicit input validation."""

import numpy as np


def mean_negative_log_likelihood(probabilities: np.ndarray) -> float:
    """Mean negative log likelihood of probabilities assigned to observed outcomes."""
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0 or np.any(values <= 0) or np.any(values > 1):
        raise ValueError("probabilities must be non-empty and in (0, 1]")
    return float(-np.mean(np.log(values)))


def state_accuracy(true_states: np.ndarray, predicted_states: np.ndarray) -> float:
    """Fraction of exactly recovered latent-state labels."""
    truth = np.asarray(true_states)
    predictions = np.asarray(predicted_states)
    if truth.ndim != 1 or truth.size == 0 or truth.shape != predictions.shape:
        raise ValueError("state arrays must be non-empty one-dimensional arrays of equal shape")
    return float(np.mean(truth == predictions))


def mean_brier_score(true_states: np.ndarray, posteriors: np.ndarray) -> float:
    """Mean multiclass Brier score for sequential state posteriors."""
    truth = np.asarray(true_states)
    probabilities = np.asarray(posteriors, dtype=float)
    if truth.ndim != 1 or probabilities.ndim != 2 or probabilities.shape[0] != truth.size:
        raise ValueError("posteriors must have one row per true state")
    if truth.size == 0 or np.any(truth < 0) or np.any(truth >= probabilities.shape[1]):
        raise ValueError("true states must index posterior columns")
    if np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("posteriors must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("posterior rows must sum to one")
    targets = np.eye(probabilities.shape[1])[truth.astype(int)]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def change_detection_delays(
    true_states: np.ndarray, predicted_states: np.ndarray, *, horizon: int | None = None
) -> np.ndarray:
    """Delay until each new true state is first predicted after a change.

    An undetected change receives ``horizon`` (or the remaining sequence length),
    making the censoring rule explicit and deterministic.
    """
    truth = np.asarray(true_states)
    predictions = np.asarray(predicted_states)
    if truth.ndim != 1 or truth.size == 0 or truth.shape != predictions.shape:
        raise ValueError("state arrays must be non-empty one-dimensional arrays of equal shape")
    if horizon is not None and horizon < 0:
        raise ValueError("horizon must be non-negative")
    changes = np.flatnonzero(truth[1:] != truth[:-1]) + 1
    delays: list[int] = []
    for change in changes:
        next_change_candidates = changes[changes > change]
        segment_end = int(next_change_candidates[0]) if next_change_candidates.size else truth.size
        if horizon is not None:
            segment_end = min(segment_end, change + horizon + 1)
        matches = np.flatnonzero(predictions[change:segment_end] == truth[change])
        censoring_delay = horizon if horizon is not None else segment_end - change
        delays.append(int(matches[0]) if matches.size else int(censoring_delay))
    return np.asarray(delays, dtype=int)
