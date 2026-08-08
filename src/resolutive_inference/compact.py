"""Reference compact sequential inference model."""

from dataclasses import dataclass, field

import numpy as np

from .emissions import gaussian_log_likelihood
from .transitions import normalize_probabilities, predict_state


@dataclass
class CompactPro:
    """Small Gaussian state-space filter used as the initial research baseline.

    Defaults are intentionally neutral. Fitting and task-specific initialization
    will be introduced only with benchmark-backed contracts.
    """

    n_states: int
    observation_dim: int
    transition: np.ndarray | None = None
    means: np.ndarray | None = None
    variances: np.ndarray | None = None
    posterior: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.n_states < 1 or self.observation_dim < 1:
            raise ValueError("n_states and observation_dim must be positive")
        self.transition = np.asarray(
            self.transition if self.transition is not None else np.eye(self.n_states),
            dtype=float,
        )
        self.means = np.asarray(
            self.means
            if self.means is not None
            else np.zeros((self.n_states, self.observation_dim)),
            dtype=float,
        )
        self.variances = np.asarray(
            self.variances
            if self.variances is not None
            else np.ones((self.n_states, self.observation_dim)),
            dtype=float,
        )
        self.posterior = normalize_probabilities(
            np.asarray(
                self.posterior
                if self.posterior is not None
                else np.ones(self.n_states),
                dtype=float,
            )
        )
        if self.transition.shape != (self.n_states, self.n_states):
            raise ValueError("transition has an incompatible shape")
        if self.means.shape != (self.n_states, self.observation_dim):
            raise ValueError("means has an incompatible shape")
        if self.variances.shape != self.means.shape or np.any(self.variances <= 0):
            raise ValueError("variances must be positive and match means")
        self.transition = np.apply_along_axis(normalize_probabilities, 1, self.transition)

    def step(self, observation: np.ndarray) -> np.ndarray:
        """Advance one observation and return the normalized state posterior."""
        value = np.asarray(observation, dtype=float)
        if value.shape != (self.observation_dim,):
            raise ValueError("observation has an incompatible shape")
        prior = predict_state(self.posterior, self.transition)
        log_weights = np.log(np.maximum(prior, np.finfo(float).tiny))
        log_weights += gaussian_log_likelihood(value, self.means, self.variances)
        log_weights -= np.max(log_weights)
        self.posterior = normalize_probabilities(np.exp(log_weights))
        return self.posterior.copy()

    @property
    def statistic_count(self) -> int:
        """Count currently stored scalar model statistics, including posterior state."""
        return int(
            self.transition.size + self.means.size + self.variances.size + self.posterior.size
        )
