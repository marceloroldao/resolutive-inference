"""Incremental session runtime for the integer lag-8 decoder.

This module advances the second-order dynamic program one observation at a time.
It does not re-decode the full observation history. Current-path reconstruction is
bounded by the configured lag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compact_robust import N_STATES, OBSERVATION_DIM
from .integer_runtime import IntegerLag8Decoder


@dataclass
class IncrementalIntegerLag8:
    decoder: IntegerLag8Decoder
    _pending_q: list[np.ndarray] = field(default_factory=list)
    _dp: np.ndarray | None = None
    _ring: list[np.ndarray] = field(default_factory=list)
    _result: list[int] = field(default_factory=list)
    _count: int = 0

    @property
    def observation_count(self) -> int:
        return self._count

    @property
    def ready(self) -> bool:
        return self._dp is not None

    @property
    def compute_mode(self) -> str:
        return "incremental-lag8"

    def append_float(self, observation: list[float] | np.ndarray) -> None:
        q = self.decoder.emission.quantize_observation(np.asarray(observation, dtype=float))
        self.append_q(q)

    def append_q(self, observation_q: np.ndarray) -> None:
        value = np.asarray(observation_q)
        if value.shape != (OBSERVATION_DIM,) or value.dtype.kind not in "iu":
            raise ValueError("observation_q must be an integer vector of length 7")

        self._count += 1
        self._result.append(-1)

        if self._dp is None:
            self._pending_q.append(value.astype(np.int16, copy=True))
            if len(self._pending_q) < 2:
                return
            self._initialize(self._pending_q[0], self._pending_q[1])
            self._pending_q.clear()
            return

        self._advance(value)

    def _initialize(self, first_q: np.ndarray, second_q: np.ndarray) -> None:
        e0 = self.decoder.emission.emission_costs_q(first_q)
        e1 = self.decoder.emission.emission_costs_q(second_q)
        dp = (
            e0[:, None].astype(np.int32)
            + e1[None, :].astype(np.int32)
            + self.decoder.initial_costs[:, None].astype(np.int32)
            + self.decoder.transition1_costs.astype(np.int32)
        )
        dp -= int(dp.min())
        self._dp = dp.astype(np.int32)

    def _advance(self, observation_q: np.ndarray) -> None:
        assert self._dp is not None
        emission = self.decoder.emission.emission_costs_q(observation_q)
        candidates = (
            self._dp[:, :, None].astype(np.int64)
            + self.decoder.transition2_costs[:, :, :].astype(np.int64)
            + emission[None, None, :].astype(np.int64)
        )
        predecessors = np.argmin(candidates, axis=0).astype(np.uint8)
        next_dp = np.min(candidates, axis=0)
        next_dp -= int(next_dp.min())
        self._dp = np.clip(next_dp, 0, np.iinfo(np.int32).max).astype(np.int32)

        self._ring.append(predecessors)
        if len(self._ring) > self.decoder.lag:
            self._ring.pop(0)

        t = self._count - 1
        if t >= self.decoder.lag + 1:
            b, c = np.unravel_index(np.argmin(self._dp), self._dp.shape)
            for decisions in reversed(self._ring):
                a = int(decisions[b, c])
                b, c = a, b
            target = t - self.decoder.lag
            if target == 1:
                self._result[0] = int(b)
            self._result[target] = int(c)

    def current_states(self) -> list[int]:
        if self._dp is None:
            return []

        result = list(self._result)
        b, c = np.unravel_index(np.argmin(self._dp), self._dp.shape)
        reverse_states = [int(c), int(b)]
        for decisions in reversed(self._ring):
            a = int(decisions[b, c])
            reverse_states.append(a)
            b, c = a, b
        suffix = list(reversed(reverse_states))
        start = self._count - len(suffix)
        for offset, state in enumerate(suffix):
            index = start + offset
            if index >= 0 and result[index] < 0:
                result[index] = int(state)

        if any(state < 0 for state in result):
            fallback_state = int(np.argmin(self._dp)) % N_STATES
            result = [fallback_state if state < 0 else state for state in result]
        return result
