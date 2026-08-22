import numpy as np

from resolutive_inference.compact_robust import CompactRobust119
from resolutive_inference.engine_cache import CompiledEngineCache


def _model(offset: float = 0.0) -> CompactRobust119:
    return CompactRobust119(
        means=np.arange(28, dtype=float).reshape(4, 7) / 10.0 + offset,
        shared_variances=np.linspace(0.7, 1.3, 7),
        transition=np.eye(4) + 0.2,
        transition2=np.ones((4, 4, 4), dtype=float),
        initial=np.ones(4, dtype=float),
        degrees_of_freedom=3.0,
    )


def test_identical_model_content_reuses_compiled_engine() -> None:
    cache = CompiledEngineCache(max_entries=4)
    first = _model()
    reloaded = _model()

    compiled_a = cache.get(first, "integer_lag8")
    compiled_b = cache.get(reloaded, "integer_lag8")

    assert compiled_a is compiled_b
    stats = cache.stats()
    assert stats.entries == 1
    assert stats.misses == 1
    assert stats.hits == 1


def test_changed_model_content_creates_distinct_entry() -> None:
    cache = CompiledEngineCache(max_entries=4)
    cache.get(_model(0.0), "integer_lag8")
    cache.get(_model(0.1), "integer_lag8")
    stats = cache.stats()
    assert stats.entries == 2
    assert stats.misses == 2


def test_lru_bound_evicts_old_entry() -> None:
    cache = CompiledEngineCache(max_entries=1)
    first = _model(0.0)
    second = _model(0.2)
    cache.get(first, "q4_lut128")
    cache.get(second, "q4_lut128")
    assert cache.stats().entries == 1

    cache.get(first, "q4_lut128")
    stats = cache.stats()
    assert stats.entries == 1
    assert stats.misses == 3


def test_cached_decode_matches_repeated_calls() -> None:
    cache = CompiledEngineCache(max_entries=4)
    model = _model()
    observations = np.tile(np.linspace(0.0, 0.6, 7), (12, 1))

    first = cache.decode(model, observations, "integer_lag8")
    second = cache.decode(model, observations, "integer_lag8")

    assert np.array_equal(first, second)
    assert cache.stats().hits == 1
