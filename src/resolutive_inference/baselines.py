"""Independent hidden Markov model filtering baselines.

These known-parameter controls intentionally avoid external ML runtimes so comparisons
remain auditable. They are conventional first-order HMM filters, not learned models.
"""

from dataclasses import dataclass, field
from math import lgamma

import numpy as np

from .transitions import normalize_probabilities


def student_t_log_likelihood(
    observation: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    degrees_of_freedom: float,
) -> np.ndarray:
    """Return independent-dimension Student-t log likelihood by state."""
    if degrees_of_freedom <= 0 or not np.isfinite(degrees_of_freedom):
        raise ValueError("degrees_of_freedom must be finite and positive")
    if scales.shape != locations.shape or np.any(scales <= 0):
        raise ValueError("scales must be positive and match locations")
    standardized = (observation[None, :] - locations) / scales
    constant = (
        lgamma((degrees_of_freedom + 1.0) / 2.0)
        - lgamma(degrees_of_freedom / 2.0)
        - 0.5 * np.log(degrees_of_freedom * np.pi)
    )
    terms = constant - np.log(scales) - (
        (degrees_of_freedom + 1.0) / 2.0
    ) * np.log1p(standardized**2 / degrees_of_freedom)
    return np.sum(terms, axis=1)


@dataclass
class GaussianHMM:
    """Known-parameter first-order HMM with diagonal Gaussian emissions."""

    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    posterior: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.transition = np.asarray(self.transition, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.variances = np.asarray(self.variances, dtype=float)
        if self.transition.ndim != 2 or self.transition.shape[0] != self.transition.shape[1]:
            raise ValueError("transition must be square")
        self.n_states = self.transition.shape[0]
        if self.means.ndim != 2 or self.means.shape[0] != self.n_states:
            raise ValueError("means must have one row per state")
        if self.variances.shape != self.means.shape or np.any(self.variances <= 0):
            raise ValueError("variances must be positive and match means")
        self.transition = np.apply_along_axis(normalize_probabilities, 1, self.transition)
        self.posterior = normalize_probabilities(
            np.ones(self.n_states)
            if self.posterior is None
            else np.asarray(self.posterior, dtype=float)
        )

    def _emission_log_likelihood(self, observation: np.ndarray) -> np.ndarray:
        residual = observation[None, :] - self.means
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * self.variances) + residual**2 / self.variances,
            axis=1,
        )

    def step(self, observation: np.ndarray) -> np.ndarray:
        """Filter one observation and return the state posterior."""
        value = np.asarray(observation, dtype=float)
        if value.shape != (self.means.shape[1],) or not np.all(np.isfinite(value)):
            raise ValueError("observation must be a finite vector of the configured dimension")
        prior = normalize_probabilities(self.posterior @ self.transition)
        log_weights = np.log(np.maximum(prior, np.finfo(float).tiny))
        log_weights += self._emission_log_likelihood(value)
        log_weights -= np.max(log_weights)
        self.posterior = normalize_probabilities(np.exp(log_weights))
        return self.posterior.copy()

    @property
    def parameter_count(self) -> int:
        """Count persistent model parameters, excluding transient posterior."""
        return int(self.transition.size + self.means.size + self.variances.size)


@dataclass
class StudentTHMM(GaussianHMM):
    """Known-parameter first-order HMM with independent Student-t emissions."""

    degrees_of_freedom: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.degrees_of_freedom <= 0 or not np.isfinite(self.degrees_of_freedom):
            raise ValueError("degrees_of_freedom must be finite and positive")

    def _emission_log_likelihood(self, observation: np.ndarray) -> np.ndarray:
        return student_t_log_likelihood(
            observation,
            self.means,
            np.sqrt(self.variances),
            self.degrees_of_freedom,
        )

    @property
    def parameter_count(self) -> int:
        return super().parameter_count + 1
