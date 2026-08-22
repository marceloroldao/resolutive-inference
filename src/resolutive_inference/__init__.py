"""Compact components for experimental sequential inference."""

from .compact import CompactPro
from .compact_robust import CompactRobust119
from .fixed_point import FixedPointViterbi
from .integer_runtime import IntegerEmissionRuntime, IntegerLag8Decoder

__all__ = [
    "CompactPro",
    "CompactRobust119",
    "FixedPointViterbi",
    "IntegerEmissionRuntime",
    "IntegerLag8Decoder",
]
__version__ = "0.1.0"
