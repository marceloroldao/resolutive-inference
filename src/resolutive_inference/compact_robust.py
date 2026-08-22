"""Compact-Robust reference configuration used for controlled Edge experiments.

This module intentionally fixes the reference shape to four latent states and seven
observation dimensions. That shape stores exactly 119 scalar model statistics:
28 means + 7 shared variances + 16 first-order transition probabilities +
64 second-order transition probabilities + 4 initial probabilities.

The implementation is a research reference, not a claim of general superiority and
not yet an integer-only MCU kernel.
"""

from dataclasses import dataclass  # noqa: I001

import numpy as np


N_STATES = 4
OBSERVATION_DIM = 7
REFERENCE_STATISTIC_COUNT = 119


def _normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum(axis=axis, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("probability rows must have positive mass")
    return values / total


@dataclass
class CompactRobust119:
    """Four-state, seven-feature second-order robust sequential reference.

    Emissions use a shared diagonal scale and a Student-t-like robust cost.
    State dynamics retain first- and second-order transition statistics. Fitting
    is supervised in this reference so benchmark behavior is auditable before
    unsupervised estimation is introduced.
    """

    means: np.ndarray
    shared_variances: np.ndarray
    transition: np.ndarray
    transition2: np.ndarray
    initial: np.ndarray
    degrees_of_freedom: float = 3.0

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=float)
        self.shared_variances = np.asarray(self.shared_variances, dtype=float)
        self.transition = np.asarray(self.transition, dtype=float)
        self.transition2 = np.asarray(self.transition2, dtype=float)
        self.initial = np.asarray(self.initial, dtype=float)

        if self.means.shape != (N_STATES, OBSERVATION_DIM):
            raise ValueError("means must have shape (4, 7)")
        if self.shared_variances.shape != (OBSERVATION_DIM,):
            raise ValueError("shared_variances must have shape (7,)")
        if self.transition.shape != (N_STATES, N_STATES):
            raise ValueError("transition must have shape (4, 4)")
        if self.transition2.shape != (N_STATES, N_STATES, N_STATES):
            raise ValueError("transition2 must have shape (4, 4, 4)")
        if self.initial.shape != (N_STATES,):
            raise ValueError("initial must have shape (4,)")
        if np.any(self.shared_variances <= 0) or self.degrees_of_freedom <= 0:
            raise ValueError("variances and degrees_of_freedom must be positive")

        self.transition = _normalize(self.transition, axis=1)
        self.transition2 = _normalize(self.transition2, axis=2)
        self.initial = _normalize(self.initial)

    @classmethod
    def fit_supervised(cls, observations: np.ndarray, states: np.ndarray) -> "CompactRobust119":
        """Fit the reference statistics from labelled sequences.

        ``observations`` must have shape ``(n_sequences, length, 7)`` and
        ``states`` shape ``(n_sequences, length)`` with labels 0..3.
        Additive smoothing keeps every transition probability finite.
        """
        x = np.asarray(observations, dtype=float)
        y = np.asarray(states, dtype=int)
        if x.ndim != 3 or x.shape[2] != OBSERVATION_DIM:
            raise ValueError("observations must have shape (n_sequences, length, 7)")
        if y.shape != x.shape[:2]:
            raise ValueError("states must match the first two observation dimensions")
        if np.any((y < 0) | (y >= N_STATES)) or not np.all(np.isfinite(x)):
            raise ValueError("states must be 0..3 and observations must be finite")

        flat_x = x.reshape(-1, OBSERVATION_DIM)
        flat_y = y.reshape(-1)
        means = np.vstack([flat_x[flat_y == state].mean(axis=0) for state in range(N_STATES)])
        if not np.all(np.isfinite(means)):
            raise ValueError("every state must occur at least once")
        residuals = flat_x - means[flat_y]
        shared_variances = residuals.var(axis=0) + 1e-6

        transition = np.full((N_STATES, N_STATES), 0.5, dtype=float)
        transition2 = np.full((N_STATES, N_STATES, N_STATES), 0.2, dtype=float)
        initial = np.full(N_STATES, 0.5, dtype=float)

        for sequence in y:
            initial[sequence[0]] += 1.0
            np.add.at(transition, (sequence[:-1], sequence[1:]), 1.0)
            if sequence.size >= 3:
                np.add.at(
                    transition2,
                    (sequence[:-2], sequence[1:-1], sequence[2:]),
                    1.0,
                )

        return cls(means, shared_variances, transition, transition2, initial)

    @property
    def statistic_count(self) -> int:
        """Return the exact number of stored scalar model statistics."""
        count = (
            self.means.size
            + self.shared_variances.size
            + self.transition.size
            + self.transition2.size
            + self.initial.size
        )
        return int(count)

    @property
    def q4_payload_bits(self) -> int:
        """Theoretical packed Q4 payload, excluding scales, offsets and firmware."""
        return self.statistic_count * 4

    @property
    def q4_payload_bytes_theoretical(self) -> float:
        """Theoretical payload size; byte-aligned storage requires rounding up."""
        return self.q4_payload_bits / 8.0

    @property
    def q4_payload_bytes_packed(self) -> int:
        """Whole bytes required by an ideal nibble-packing implementation."""
        return (self.q4_payload_bits + 7) // 8

    def emission_cost(self, observation: np.ndarray) -> np.ndarray:
        """Return Student-t-like robust costs for one seven-feature observation."""
        value = np.asarray(observation, dtype=float)
        if value.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(value)):
            raise ValueError("observation must be a finite vector of length 7")
        distance = ((value[None, :] - self.means) ** 2 / self.shared_variances).sum(axis=1)
        nu = self.degrees_of_freedom
        return 0.5 * (nu + OBSERVATION_DIM) * np.log1p(distance / nu)

    def decode(self, observations: np.ndarray) -> np.ndarray:
        """Decode a sequence with second-order dynamic programming.

        This is the full-backtrace float reference. Q4/LUT and bounded-backtrace
        variants are benchmarked separately so their approximation error remains visible.
        """
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != OBSERVATION_DIM or x.shape[0] < 2:
            raise ValueError("observations must have shape (length>=2, 7)")
        if not np.all(np.isfinite(x)):
            raise ValueError("observations must be finite")

        costs = np.vstack([self.emission_cost(row) for row in x])
        first = -0.6 * np.log(self.initial + 1e-12)
        trans1 = -0.62 * np.log(self.transition + 1e-12)
        trans2 = -0.55 * np.log(self.transition2 + 1e-12)

        dp = costs[0, :, None] + costs[1, None, :] + first[:, None] + trans1
        back = np.zeros((x.shape[0], N_STATES, N_STATES), dtype=np.uint8)

        for t in range(2, x.shape[0]):
            candidates = dp[:, :, None] + trans2 + costs[t][None, None, :]
            back[t] = np.argmin(candidates, axis=0).astype(np.uint8)
            dp = np.min(candidates, axis=0)

        b, c = np.unravel_index(np.argmin(dp), dp.shape)
        path = np.empty(x.shape[0], dtype=int)
        path[-2:] = [b, c]
        for t in range(x.shape[0] - 1, 1, -1):
            path[t - 2] = back[t, path[t - 1], path[t]]
        return path
