"""Integer-only runtime reference for Compact-Robust Edge inference.

Compilation from the floating/Q4 research model may use floating point, but the
runtime API in :class:`IntegerEmissionRuntime` consumes already-quantized int16
observations and computes robust emission costs using integer arithmetic only.

This is still a Python reference. It is intended to make the fixed-point contract
explicit before a C/C++ MCU port.
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
        """Convenience preprocessing helper; not part of the integer-only runtime claim."""
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
