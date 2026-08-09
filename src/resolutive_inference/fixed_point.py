"""Integer reference primitives for robust Student-t-like emissions."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StudentTLUT:
    """Lookup table approximating ``log(1 + d / 3)`` with integer values."""

    values: np.ndarray
    max_distance: float
    value_scale: int

    @classmethod
    def build(
        cls, entries: int = 128, *, max_distance: float = 48.0, value_scale: int = 256
    ) -> "StudentTLUT":
        if entries < 2 or max_distance <= 0 or value_scale <= 0:
            raise ValueError("invalid LUT configuration")
        distances = np.linspace(0.0, max_distance, entries)
        values = np.rint(np.log1p(distances / 3.0) * value_scale).astype(np.uint16)
        return cls(values=values, max_distance=max_distance, value_scale=value_scale)

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)

    def lookup(self, squared_distance: np.ndarray | float) -> np.ndarray:
        distance = np.clip(np.asarray(squared_distance, dtype=float), 0.0, self.max_distance)
        indices = np.rint(distance * (len(self.values) - 1) / self.max_distance).astype(int)
        return self.values[indices]


def quantize_q(values: np.ndarray, *, fractional_bits: int, storage_bits: int = 16) -> np.ndarray:
    """Convert real values to a signed fixed-point integer representation."""
    if fractional_bits < 0 or storage_bits not in (8, 16, 32):
        raise ValueError("unsupported fixed-point format")
    scale = 1 << fractional_bits
    limits = np.iinfo(getattr(np, f"int{storage_bits}"))
    return np.clip(np.rint(np.asarray(values, dtype=float) * scale), limits.min, limits.max).astype(
        limits.dtype
    )
