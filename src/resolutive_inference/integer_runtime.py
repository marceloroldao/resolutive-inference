"""Integer-only runtime references for Compact-Robust Edge inference.

Compilation from the floating/Q4 research model may use floating point, but the
runtime APIs consume already-quantized int16 observations and perform emission,
transition scoring and bounded second-order decoding using integer arithmetic.

These are Python references intended to make the fixed-point contract explicit
before a C/C++ MCU port.
"""

from dataclasses import dataclass

import numpy as np

from .compact_robust import N_STATES, OBSERVATION_DIM
from .edge_compact import Q4CompactRobust119, StudentTCostLUT


@dataclass(frozen=True)
class IntegerEmissionRuntime:
    """Compiled integer-only emission kernel for four states and seven features."""

    means_q: np.ndarray
    inv_variances_q: np.ndarray
    lut_values: np.ndarray
    observation_scale: int
    inverse_variance_scale: int
    max_distance_q: int

    @classmethod
    def compile(
        cls,
        model: Q4CompactRobust119,
        lut: StudentTCostLUT,
        *,
        observation_scale: int = 32,
        inverse_variance_scale: int = 128,
    ) -> "IntegerEmissionRuntime":
        """Compile Q4 parameters into integer runtime tables.

        Compilation is allowed to use floating point. Runtime methods do not.
        """
        if observation_scale <= 0 or inverse_variance_scale <= 0:
            raise ValueError("fixed-point scales must be positive")

        float_model = model.to_float_model()
        means_q = np.rint(float_model.means * observation_scale)
        means_q = np.clip(means_q, -32768, 32767).astype(np.int16)

        inv_variances = 1.0 / float_model.shared_variances
        inv_variances_q = np.rint(inv_variances * inverse_variance_scale)
        inv_variances_q = np.clip(inv_variances_q, 1, 65535).astype(np.uint16)

        max_distance_q = int(round(lut.max_distance * observation_scale * observation_scale))
        if max_distance_q <= 0:
            raise ValueError("compiled maximum distance must be positive")

        return cls(
            means_q=means_q,
            inv_variances_q=inv_variances_q,
            lut_values=np.asarray(lut.values, dtype=np.uint16).copy(),
            observation_scale=int(observation_scale),
            inverse_variance_scale=int(inverse_variance_scale),
            max_distance_q=max_distance_q,
        )

    @property
    def persistent_bytes(self) -> int:
        """Bytes stored by the compiled emission tables only."""
        return int(
            self.means_q.nbytes
            + self.inv_variances_q.nbytes
            + self.lut_values.nbytes
            + 3 * np.dtype(np.int32).itemsize
        )

    def quantize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Convenience preprocessing helper; excluded from the integer-runtime claim."""
        value = np.asarray(observation, dtype=float)
        if value.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(value)):
            raise ValueError("observation must be a finite vector of length 7")
        quantized = np.rint(value * self.observation_scale)
        return np.clip(quantized, -32768, 32767).astype(np.int16)

    def emission_costs_q(self, observation_q: np.ndarray) -> np.ndarray:
        """Return LUT costs using integer arithmetic only at runtime."""
        value = np.asarray(observation_q)
        if value.shape != (OBSERVATION_DIM,) or value.dtype.kind not in "iu":
            raise ValueError("observation_q must be an integer vector of length 7")

        x = value.astype(np.int32, copy=False)
        means = self.means_q.astype(np.int32, copy=False)
        invv = self.inv_variances_q.astype(np.int64, copy=False)

        diff = x[None, :] - means
        squared = diff.astype(np.int64) * diff.astype(np.int64)
        weighted = (squared * invv[None, :]) // self.inverse_variance_scale
        distance_q = weighted.sum(axis=1, dtype=np.int64)

        numerator = distance_q * (self.lut_values.size - 1) + self.max_distance_q // 2
        indices = numerator // self.max_distance_q
        indices = np.clip(indices, 0, self.lut_values.size - 1).astype(np.intp)
        return self.lut_values[indices].astype(np.int32)

    def emission_costs_from_float(self, observation: np.ndarray) -> np.ndarray:
        """Reference convenience path: preprocess float then run integer kernel."""
        return self.emission_costs_q(self.quantize_observation(observation))


@dataclass(frozen=True)
class IntegerLag8Decoder:
    """Integer-only second-order decoder with bounded lag-8 predecessor memory."""

    emission: IntegerEmissionRuntime
    initial_costs: np.ndarray
    transition1_costs: np.ndarray
    transition2_costs: np.ndarray
    lag: int = 8

    @classmethod
    def compile(
        cls,
        model: Q4CompactRobust119,
        lut: StudentTCostLUT,
        *,
        lag: int = 8,
        observation_scale: int = 32,
        inverse_variance_scale: int = 128,
    ) -> "IntegerLag8Decoder":
        """Compile Q4 probabilities and emissions into integer runtime tables."""
        if lag < 1:
            raise ValueError("lag must be positive")
        emission = IntegerEmissionRuntime.compile(
            model,
            lut,
            observation_scale=observation_scale,
            inverse_variance_scale=inverse_variance_scale,
        )
        float_model = model.to_float_model()
        scale = int(lut.cost_scale)
        initial = np.rint(-0.60 * np.log(float_model.initial + 1e-12) * scale)
        transition1 = np.rint(-0.62 * np.log(float_model.transition + 1e-12) * scale)
        transition2 = np.rint(-0.55 * np.log(float_model.transition2 + 1e-12) * scale)
        return cls(
            emission=emission,
            initial_costs=np.clip(initial, 0, 32767).astype(np.int16),
            transition1_costs=np.clip(transition1, 0, 32767).astype(np.int16),
            transition2_costs=np.clip(transition2, 0, 32767).astype(np.int16),
            lag=int(lag),
        )

    @property
    def persistent_bytes(self) -> int:
        """Compiled runtime tables: 510 bytes for the default Q4/LUT-128 reference."""
        return int(
            self.emission.persistent_bytes
            + self.initial_costs.nbytes
            + self.transition1_costs.nbytes
            + self.transition2_costs.nbytes
            + np.dtype(np.int32).itemsize
        )

    @property
    def runtime_buffer_bytes(self) -> int:
        """Score, lag-ring and emission bytes: 272 bytes when lag equals eight."""
        score_bytes = 2 * N_STATES * N_STATES * np.dtype(np.int32).itemsize
        backtrace_bytes = self.lag * N_STATES * N_STATES * np.dtype(np.uint8).itemsize
        emission_bytes = N_STATES * np.dtype(np.int32).itemsize
        return int(score_bytes + backtrace_bytes + emission_bytes)

    @property
    def core_bytes(self) -> int:
        """Persistent compiled tables plus algorithmic runtime buffers."""
        return self.persistent_bytes + self.runtime_buffer_bytes

    def quantize_sequence(self, observations: np.ndarray) -> np.ndarray:
        """Convenience preprocessing helper; excluded from the integer-runtime claim."""
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != OBSERVATION_DIM or x.shape[0] < 2:
            raise ValueError("observations must have shape (length>=2, 7)")
        if not np.all(np.isfinite(x)):
            raise ValueError("observations must be finite")
        q = np.rint(x * self.emission.observation_scale)
        return np.clip(q, -32768, 32767).astype(np.int16)

    def decode_q(self, observations_q: np.ndarray) -> np.ndarray:
        """Decode a quantized sequence using integer arithmetic at runtime."""
        x = np.asarray(observations_q)
        if x.ndim != 2 or x.shape[1] != OBSERVATION_DIM or x.shape[0] < 2:
            raise ValueError("observations_q must have shape (length>=2, 7)")
        if x.dtype.kind not in "iu":
            raise ValueError("observations_q must use an integer dtype")

        n = x.shape[0]
        e0 = self.emission.emission_costs_q(x[0])
        e1 = self.emission.emission_costs_q(x[1])
        dp = (
            e0[:, None].astype(np.int32)
            + e1[None, :].astype(np.int32)
            + self.initial_costs[:, None].astype(np.int32)
            + self.transition1_costs.astype(np.int32)
        )
        dp -= int(dp.min())

        result = np.full(n, -1, dtype=np.int16)
        ring: list[np.ndarray] = []
        trans2 = self.transition2_costs.astype(np.int32)

        for t in range(2, n):
            emission = self.emission.emission_costs_q(x[t])
            candidates = (
                dp[:, :, None].astype(np.int64)
                + trans2[:, :, :].astype(np.int64)
                + emission[None, None, :].astype(np.int64)
            )
            predecessors = np.argmin(candidates, axis=0).astype(np.uint8)
            next_dp = np.min(candidates, axis=0)
            next_dp -= int(next_dp.min())
            dp = np.clip(next_dp, 0, np.iinfo(np.int32).max).astype(np.int32)

            ring.append(predecessors)
            if len(ring) > self.lag:
                ring.pop(0)

            if t >= self.lag + 1:
                b, c = np.unravel_index(np.argmin(dp), dp.shape)
                for decisions in reversed(ring):
                    a = int(decisions[b, c])
                    b, c = a, b
                target = t - self.lag
                if target == 1:
                    result[0] = b
                result[target] = c

        b, c = np.unravel_index(np.argmin(dp), dp.shape)
        reverse_states = [int(c), int(b)]
        for decisions in reversed(ring):
            a = int(decisions[b, c])
            reverse_states.append(a)
            b, c = a, b
        suffix = np.asarray(reverse_states[::-1], dtype=np.int16)
        start = n - suffix.size
        for offset, state in enumerate(suffix):
            index = start + offset
            if index >= 0 and result[index] < 0:
                result[index] = state

        if np.any(result < 0):
            # Boundary fallback stays integer-only and is deterministic.
            fallback_state = int(np.argmin(dp)) % N_STATES
            result[result < 0] = fallback_state
        return result.astype(int)

    def decode_from_float(self, observations: np.ndarray) -> np.ndarray:
        """Convenience path: quantize externally visible float input then decode integers."""
        return self.decode_q(self.quantize_sequence(observations))
