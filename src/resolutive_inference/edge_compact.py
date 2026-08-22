"""Quantized Edge references for :class:`CompactRobust119`.

The Q4 payload calculation covers the 119 quantized model values only. Per-tensor
scale/offset metadata and the Student-t LUT are accounted for separately.
"""

from dataclasses import dataclass

import numpy as np

from .compact_robust import OBSERVATION_DIM, CompactRobust119


@dataclass(frozen=True)
class QuantizedTensor:
    """Uniform n-bit tensor with explicit affine reconstruction metadata."""

    codes: np.ndarray
    lower: float
    scale: float
    bits: int

    @classmethod
    def from_float(cls, values: np.ndarray, bits: int = 4) -> "QuantizedTensor":
        if bits < 2 or bits > 8:
            raise ValueError("bits must be between 2 and 8")
        array = np.asarray(values, dtype=float)
        lower = float(array.min())
        upper = float(array.max())
        levels = (1 << bits) - 1
        if upper == lower:
            scale = 1.0
            codes = np.zeros(array.shape, dtype=np.uint8)
        else:
            scale = (upper - lower) / levels
            codes = np.clip(np.rint((array - lower) / scale), 0, levels).astype(np.uint8)
        return cls(codes=codes, lower=lower, scale=float(scale), bits=bits)

    def dequantize(self) -> np.ndarray:
        return self.codes.astype(float) * self.scale + self.lower

    @property
    def payload_bits(self) -> int:
        return int(self.codes.size * self.bits)


@dataclass(frozen=True)
class StudentTCostLUT:
    """Unsigned integer LUT for the robust Student-t-like emission cost."""

    values: np.ndarray
    max_distance: float
    cost_scale: int
    degrees_of_freedom: float

    @classmethod
    def build(
        cls,
        entries: int = 128,
        *,
        max_distance: float = 220.0,
        cost_scale: int = 64,
        degrees_of_freedom: float = 3.0,
    ) -> "StudentTCostLUT":
        if entries < 2 or max_distance <= 0 or cost_scale <= 0 or degrees_of_freedom <= 0:
            raise ValueError("invalid LUT configuration")
        distance = np.linspace(0.0, max_distance, entries)
        cost = 0.5 * (degrees_of_freedom + OBSERVATION_DIM) * np.log1p(
            distance / degrees_of_freedom
        )
        values = np.rint(cost * cost_scale).astype(np.uint16)
        return cls(values, max_distance, cost_scale, degrees_of_freedom)

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)

    def lookup(self, distance: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(distance, dtype=float), 0.0, self.max_distance)
        index = np.rint(clipped * (len(self.values) - 1) / self.max_distance).astype(int)
        return self.values[index].astype(float) / self.cost_scale


@dataclass(frozen=True)
class Q4CompactRobust119:
    """Five-tensor Q4 representation of the 119-value reference configuration."""

    means: QuantizedTensor
    shared_variances: QuantizedTensor
    transition: QuantizedTensor
    transition2: QuantizedTensor
    initial: QuantizedTensor
    degrees_of_freedom: float = 3.0

    @classmethod
    def from_model(cls, model: CompactRobust119) -> "Q4CompactRobust119":
        return cls(
            means=QuantizedTensor.from_float(model.means, 4),
            shared_variances=QuantizedTensor.from_float(model.shared_variances, 4),
            transition=QuantizedTensor.from_float(model.transition, 4),
            transition2=QuantizedTensor.from_float(model.transition2, 4),
            initial=QuantizedTensor.from_float(model.initial, 4),
            degrees_of_freedom=model.degrees_of_freedom,
        )

    @property
    def tensors(self) -> tuple[QuantizedTensor, ...]:
        return (
            self.means,
            self.shared_variances,
            self.transition,
            self.transition2,
            self.initial,
        )

    @property
    def payload_bits(self) -> int:
        return sum(tensor.payload_bits for tensor in self.tensors)

    @property
    def payload_bytes_theoretical(self) -> float:
        return self.payload_bits / 8.0

    @property
    def payload_bytes_packed(self) -> int:
        return (self.payload_bits + 7) // 8

    @property
    def affine_metadata_bytes_float32(self) -> int:
        """Two float32 values (lower, scale) for each of five tensors."""
        return len(self.tensors) * 2 * 4

    def to_float_model(self) -> CompactRobust119:
        transition = np.maximum(self.transition.dequantize(), 1e-12)
        transition2 = np.maximum(self.transition2.dequantize(), 1e-12)
        initial = np.maximum(self.initial.dequantize(), 1e-12)
        variances = np.maximum(self.shared_variances.dequantize(), 1e-6)
        return CompactRobust119(
            means=self.means.dequantize(),
            shared_variances=variances,
            transition=transition,
            transition2=transition2,
            initial=initial,
            degrees_of_freedom=self.degrees_of_freedom,
        )

    def decode(self, observations: np.ndarray, lut: StudentTCostLUT | None = None) -> np.ndarray:
        """Decode with dequantized Q4 parameters and optional LUT emission cost.

        This remains a hybrid Python reference: distances are floating point. The
        LUT isolates approximation error before an integer-only kernel is attempted.
        """
        model = self.to_float_model()
        if lut is None:
            return model.decode(observations)

        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != OBSERVATION_DIM or x.shape[0] < 2:
            raise ValueError("observations must have shape (length>=2, 7)")
        distance = np.stack(
            [((row[None, :] - model.means) ** 2 / model.shared_variances).sum(axis=1) for row in x]
        )
        costs = lut.lookup(distance)

        first = -0.6 * np.log(model.initial + 1e-12)
        trans1 = -0.62 * np.log(model.transition + 1e-12)
        trans2 = -0.55 * np.log(model.transition2 + 1e-12)
        dp = costs[0, :, None] + costs[1, None, :] + first[:, None] + trans1
        back = np.zeros((x.shape[0], 4, 4), dtype=np.uint8)
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
