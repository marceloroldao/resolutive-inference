"""Bounded-backtrace decoder for the Compact-Robust second-order reference.

The implementation is a hybrid Python reference. Q4 parameters are reconstructed to
floating point and emission distances remain floating point; bounded backtrace itself
uses a fixed-size uint8 predecessor ring.
"""

from dataclasses import dataclass

import numpy as np

from .compact_robust import N_STATES, OBSERVATION_DIM
from .edge_compact import Q4CompactRobust119, StudentTCostLUT


@dataclass
class BoundedSecondOrderDecoder:
    """Fixed-lag second-order Viterbi reference with bounded predecessor memory."""

    model: Q4CompactRobust119
    lag: int
    lut: StudentTCostLUT | None = None

    def __post_init__(self) -> None:
        if self.lag < 1:
            raise ValueError("lag must be positive")

    @property
    def runtime_buffer_bytes(self) -> int:
        """Algorithmic score/backpointer bytes, excluding Python/container overhead.

        Two ``4 x 4`` int32 score planes require 128 bytes. The predecessor ring
        requires ``lag x 4 x 4`` uint8 entries. Observation buffers, float emission
        temporaries, firmware, stack and allocator overhead are deliberately excluded.
        """
        score_bytes = 2 * N_STATES * N_STATES * np.dtype(np.int32).itemsize
        backtrace_bytes = self.lag * N_STATES * N_STATES * np.dtype(np.uint8).itemsize
        return int(score_bytes + backtrace_bytes)

    @property
    def operations_per_observation_proxy(self) -> int:
        """Simple second-order candidate/add/compare proxy, not MCU instructions."""
        return int(N_STATES**3 * 3 + N_STATES * OBSERVATION_DIM * 4)

    def _costs(self, observations: np.ndarray) -> tuple[np.ndarray, object]:
        float_model = self.model.to_float_model()
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != OBSERVATION_DIM or x.shape[0] < 2:
            raise ValueError("observations must have shape (length>=2, 7)")
        if not np.all(np.isfinite(x)):
            raise ValueError("observations must be finite")

        distance = np.stack(
            [
                ((row[None, :] - float_model.means) ** 2 / float_model.shared_variances).sum(
                    axis=1
                )
                for row in x
            ]
        )
        if self.lut is None:
            nu = float_model.degrees_of_freedom
            costs = 0.5 * (nu + OBSERVATION_DIM) * np.log1p(distance / nu)
        else:
            costs = self.lut.lookup(distance)
        return costs, float_model

    def decode(self, observations: np.ndarray) -> np.ndarray:
        """Decode one sequence while retaining at most ``lag`` predecessor planes.

        States are finalized with fixed lag. The retained suffix is reconstructed from
        the final best pair. For sequences shorter than the lag, behavior reduces to
        the full-backtrace Q4 reference.
        """
        costs, model = self._costs(observations)
        n = costs.shape[0]
        first = -0.6 * np.log(model.initial + 1e-12)
        trans1 = -0.62 * np.log(model.transition + 1e-12)
        trans2 = -0.55 * np.log(model.transition2 + 1e-12)

        dp = costs[0, :, None] + costs[1, None, :] + first[:, None] + trans1
        result = np.full(n, -1, dtype=int)
        ring: list[np.ndarray] = []

        for t in range(2, n):
            candidates = dp[:, :, None] + trans2 + costs[t][None, None, :]
            predecessors = np.argmin(candidates, axis=0).astype(np.uint8)
            dp = np.min(candidates, axis=0)
            ring.append(predecessors)
            if len(ring) > self.lag:
                ring.pop(0)

            # At t = lag + 1 the ring spans transitions 2..t and can establish
            # the first pair (state 0, state 1). Thereafter one state is finalized.
            if t >= self.lag + 1:
                b, c = np.unravel_index(np.argmin(dp), dp.shape)
                for decisions in reversed(ring):
                    a = int(decisions[b, c])
                    b, c = a, b
                target = t - self.lag
                if target == 1:
                    result[0] = b
                result[target] = c

        # Fill the still-unfinalized suffix from the final best path through the
        # bounded predecessor ring. Already finalized states are not overwritten.
        b, c = np.unravel_index(np.argmin(dp), dp.shape)
        reverse_states = [int(c), int(b)]
        for decisions in reversed(ring):
            a = int(decisions[b, c])
            reverse_states.append(a)
            b, c = a, b
        suffix = np.asarray(reverse_states[::-1], dtype=int)
        start = n - suffix.size
        for offset, state in enumerate(suffix):
            index = start + offset
            if index >= 0 and result[index] < 0:
                result[index] = state

        if np.any(result < 0):
            # This only occurs for very small lag/sequence boundary combinations.
            # Recover missing boundary states from the full Q4 reference without
            # changing already finalized fixed-lag decisions.
            full = self.model.decode(np.asarray(observations, dtype=float), lut=self.lut)
            result[result < 0] = full[result < 0]
        return result
