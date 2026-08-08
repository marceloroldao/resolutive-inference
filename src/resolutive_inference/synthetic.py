"""Reproducible synthetic sequences with known latent states."""

from dataclasses import dataclass

import numpy as np

from .transitions import normalize_probabilities


@dataclass(frozen=True)
class SyntheticSequence:
    """Observations and ground-truth states produced by a known model."""

    observations: np.ndarray
    states: np.ndarray
    seed: int


def generate_gaussian_hmm(
    length: int,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    *,
    seed: int,
    initial: np.ndarray | None = None,
) -> SyntheticSequence:
    """Sample a finite-state HMM with diagonal Gaussian emissions."""
    if length < 1:
        raise ValueError("length must be positive")
    transition = np.asarray(transition, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be square")
    n_states = transition.shape[0]
    if means.ndim != 2 or means.shape[0] != n_states:
        raise ValueError("means must have one row per state")
    if variances.shape != means.shape or np.any(variances <= 0):
        raise ValueError("variances must be positive and match means")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(variances)):
        raise ValueError("emission parameters must be finite")
    normalized_transition = np.apply_along_axis(
        normalize_probabilities, 1, transition
    )
    initial_probabilities = normalize_probabilities(
        np.ones(n_states) if initial is None else np.asarray(initial, dtype=float)
    )

    rng = np.random.default_rng(seed)
    states = np.empty(length, dtype=int)
    observations = np.empty((length, means.shape[1]), dtype=float)
    states[0] = rng.choice(n_states, p=initial_probabilities)
    for index in range(length):
        if index:
            states[index] = rng.choice(n_states, p=normalized_transition[states[index - 1]])
        state = states[index]
        observations[index] = rng.normal(means[state], np.sqrt(variances[state]))
    return SyntheticSequence(observations=observations, states=states, seed=seed)
