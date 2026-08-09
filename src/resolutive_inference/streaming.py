"""Streaming Viterbi decoding with full or bounded backtrace."""

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .fixed_point import StudentTLUT


@dataclass(frozen=True)
class StreamingResult:
    """Result of one update; ``finalized_state`` is delayed for bounded decoding."""

    best_state: int
    finalized_state: int | None
    finalized_index: int | None


@dataclass
class FixedPointViterbi:
    """Integer, one-observation-at-a-time Viterbi reference decoder.

    ``lag=None`` retains the complete backtrace. A finite lag retains only the
    decisions required to finalize a state after that many observations.
    """

    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    lut: StudentTLUT = field(default_factory=StudentTLUT.build)
    lag: int | None = None
    score_scale: int = 256

    def __post_init__(self) -> None:
        self.transition = np.asarray(self.transition, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.variances = np.asarray(self.variances, dtype=float)
        if self.transition.ndim != 2 or self.transition.shape[0] != self.transition.shape[1]:
            raise ValueError("transition must be square")
        self.n_states = self.transition.shape[0]
        if self.means.shape[0] != self.n_states or self.variances.shape != self.means.shape:
            raise ValueError("emission arrays have incompatible shapes")
        if np.any(self.transition <= 0) or np.any(self.variances <= 0):
            raise ValueError("transition probabilities and variances must be positive")
        if self.lag is not None and self.lag < 1:
            raise ValueError("lag must be positive or None")
        normalized = self.transition / self.transition.sum(axis=1, keepdims=True)
        self.transition_scores = np.rint(np.log(normalized) * self.score_scale).astype(np.int32)
        self.reset()

    def reset(self) -> None:
        """Clear runtime state while preserving model parameters and the LUT."""
        self.scores = np.zeros(self.n_states, dtype=np.int32)
        self._backtrace: list[np.ndarray] = []
        self._time = -1
        self._emitted = 0

    def _emission_scores(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=float)
        if value.shape != (self.means.shape[1],) or not np.all(np.isfinite(value)):
            raise ValueError("observation has an incompatible shape or non-finite values")
        distance = np.sum((value[None, :] - self.means) ** 2 / self.variances, axis=1)
        return -self.lut.lookup(distance).astype(np.int32)

    def update(self, observation: np.ndarray) -> StreamingResult:
        """Consume one observation without requiring the full sequence in memory."""
        self._time += 1
        candidates = self.scores[:, None] + self.transition_scores
        predecessors = np.argmax(candidates, axis=0).astype(np.uint8)
        self.scores = candidates[predecessors, np.arange(self.n_states)] + self._emission_scores(
            observation
        )
        self._backtrace.append(predecessors)
        finalized = None
        finalized_index = None
        if self.lag is not None and len(self._backtrace) > self.lag:
            finalized = self._trace_state(int(np.argmax(self.scores)), self.lag)
            finalized_index = self._emitted
            self._emitted += 1
            self._backtrace.pop(0)
        return StreamingResult(int(np.argmax(self.scores)), finalized, finalized_index)

    def _trace_state(self, state: int, steps: int) -> int:
        for decisions in reversed(self._backtrace[-steps:]):
            state = int(decisions[state])
        return state

    def flush(self) -> np.ndarray:
        """Finalize the retained suffix and return it in chronological order."""
        state = int(np.argmax(self.scores))
        suffix = [state]
        for decisions in reversed(self._backtrace[1:]):
            state = int(decisions[state])
            suffix.append(state)
        suffix.reverse()
        self._backtrace.clear()
        self._emitted += len(suffix)
        return np.asarray(suffix, dtype=int)

    def decode(self, observations: Iterable[np.ndarray]) -> np.ndarray:
        """Batch convenience wrapper over exactly the same streaming interface."""
        self.reset()
        output: list[int] = []
        for observation in observations:
            result = self.update(observation)
            if result.finalized_state is not None:
                output.append(result.finalized_state)
        output.extend(self.flush().tolist())
        return np.asarray(output, dtype=int)

    @property
    def runtime_buffer_bytes(self) -> int:
        """Estimated decoder buffers, excluding Python/container overhead."""
        retained_steps = len(self._backtrace) if self.lag is None else self.lag
        return int(2 * self.n_states * 4 + retained_steps * self.n_states)

    @property
    def operations_per_observation(self) -> int:
        """Proxy: transition additions/comparisons plus one LUT lookup per state."""
        return int(2 * self.n_states * self.n_states + self.n_states)
