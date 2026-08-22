"""Bounded process-local cache for compiled inference engines."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Literal

import numpy as np

from .compact_robust import CompactRobust119
from .edge_compact import Q4CompactRobust119, StudentTCostLUT
from .integer_runtime import IntegerLag8Decoder

CachedEngineName = Literal["q4_lut128", "integer_lag8"]


@dataclass(frozen=True)
class CacheStats:
    entries: int
    hits: int
    misses: int
    max_entries: int


class CompiledEngineCache:
    """LRU cache keyed by deterministic model content and engine name."""

    def __init__(self, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[str, CachedEngineName], object] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    @staticmethod
    def fingerprint(model: CompactRobust119) -> str:
        digest = hashlib.sha256()
        for values in (
            model.means,
            model.shared_variances,
            model.transition,
            model.transition2,
            model.initial,
        ):
            array = np.ascontiguousarray(values, dtype=np.float64)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        digest.update(np.float64(model.degrees_of_freedom).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _compile(model: CompactRobust119, engine: CachedEngineName) -> object:
        q4 = Q4CompactRobust119.from_model(model)
        lut = StudentTCostLUT.build(128, degrees_of_freedom=model.degrees_of_freedom)
        if engine == "q4_lut128":
            return (q4, lut)
        if engine == "integer_lag8":
            return IntegerLag8Decoder.compile(q4, lut, lag=8)
        raise ValueError(f"unsupported cached engine: {engine}")

    def get(self, model: CompactRobust119, engine: CachedEngineName) -> object:
        key = (self.fingerprint(model), engine)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._hits += 1
                self._entries.move_to_end(key)
                return cached

            self._misses += 1
            compiled = self._compile(model, engine)
            self._entries[key] = compiled
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return compiled

    def decode(
        self,
        model: CompactRobust119,
        observations: np.ndarray,
        engine: CachedEngineName,
    ) -> np.ndarray:
        compiled = self.get(model, engine)
        if engine == "q4_lut128":
            q4, lut = compiled
            return q4.decode(observations, lut=lut)
        decoder = compiled
        return decoder.decode_from_float(observations)

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                entries=len(self._entries),
                hits=self._hits,
                misses=self._misses,
                max_entries=self.max_entries,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0


DEFAULT_ENGINE_CACHE = CompiledEngineCache()
