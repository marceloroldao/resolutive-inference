"""Hybrid quantized reference for bounded-backtrace Viterbi decoding.

Despite the historical ``FixedPointViterbi`` name, this module is not an
integer-only fixed-point MCU kernel: emission distances are evaluated in
floating point before a quantized lookup table is applied.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FixedPointViterbi:
    """Hybrid quantized Viterbi reference with bounded backtrace.

    Path scores, transition scores, and the emission lookup table are integer
    arrays.  The squared distance used to choose an emission LUT entry is still
    calculated in floating point.  Consequently this Python reference is not
    an integer-only fixed-point kernel suitable for an MCU.
    """

    means: np.ndarray
    transition_scores: np.ndarray
    emission_lut: np.ndarray
    backtrace_depth: int = 16
    distance_max: float = 16.0
    _scores: np.ndarray = field(init=False, repr=False)
    _next_scores: np.ndarray = field(init=False, repr=False)
    _backpointers: np.ndarray = field(init=False, repr=False)
    _position: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=float)
        self.transition_scores = np.asarray(self.transition_scores, dtype=np.int16)
        self.emission_lut = np.asarray(self.emission_lut, dtype=np.int16)
        if self.means.ndim != 2 or not self.means.shape[0]:
            raise ValueError("means must have shape (n_states, observation_dim)")
        n_states = self.means.shape[0]
        if self.transition_scores.shape != (n_states, n_states):
            raise ValueError("transition_scores has an incompatible shape")
        if self.emission_lut.ndim != 1 or self.emission_lut.size < 2:
            raise ValueError("emission_lut must contain at least two entries")
        if self.backtrace_depth < 1 or self.distance_max <= 0:
            raise ValueError("backtrace_depth and distance_max must be positive")
        pointer_dtype = np.min_scalar_type(n_states - 1)
        if pointer_dtype.kind != "u":
            pointer_dtype = np.dtype(np.uint8)
        self._scores = np.zeros(n_states, dtype=np.int32)
        self._next_scores = np.empty(n_states, dtype=np.int32)
        self._backpointers = np.empty(
            (self.backtrace_depth, n_states), dtype=pointer_dtype
        )

    @property
    def runtime_buffer_bytes(self) -> int:
        """Return bytes allocated for mutable score and backtrace buffers."""
        return self._scores.nbytes + self._next_scores.nbytes + self._backpointers.nbytes

    def step(self, observation: np.ndarray) -> int:
        """Consume one sample and return the current best state.

        Emission *distance* calculation below intentionally remains floating
        point; only its resulting LUT index and score participate in the
        quantized/integer dynamic program.
        """
        value = np.asarray(observation, dtype=float)
        if value.shape != (self.means.shape[1],) or not np.all(np.isfinite(value)):
            raise ValueError("observation must be finite and match observation_dim")
        distances = np.sum((self.means - value) ** 2, axis=1)
        indices = np.rint(
            np.clip(distances / self.distance_max, 0.0, 1.0)
            * (self.emission_lut.size - 1)
        ).astype(np.intp)
        emission_scores = self.emission_lut[indices].astype(np.int32)
        candidates = self._scores[:, None] + self.transition_scores.astype(np.int32)
        row = self._position % self.backtrace_depth
        self._backpointers[row] = np.argmax(candidates, axis=0)
        self._next_scores[:] = np.max(candidates, axis=0) + emission_scores
        self._scores, self._next_scores = self._next_scores, self._scores
        self._scores -= np.max(self._scores)
        self._position += 1
        return int(np.argmax(self._scores))
