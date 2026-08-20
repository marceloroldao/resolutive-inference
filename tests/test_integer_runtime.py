import numpy as np

from resolutive_inference.compact_robust import CompactRobust119
from resolutive_inference.edge_compact import Q4CompactRobust119, StudentTCostLUT
from resolutive_inference.integer_runtime import IntegerEmissionRuntime, IntegerLag8Decoder


def _q4() -> Q4CompactRobust119:
    model = CompactRobust119(
        means=np.arange(28, dtype=float).reshape(4, 7) / 10.0,
        shared_variances=np.linspace(0.7, 1.3, 7),
        transition=np.eye(4) + 0.2,
        transition2=np.ones((4, 4, 4)) + np.eye(4)[:, None, :],
        initial=np.ones(4),
    )
    return Q4CompactRobust119.from_model(model)


def _runtime() -> IntegerEmissionRuntime:
    return IntegerEmissionRuntime.compile(_q4(), StudentTCostLUT.build(128))


def test_integer_runtime_shapes_and_storage() -> None:
    runtime = _runtime()
    assert runtime.means_q.shape == (4, 7)
    assert runtime.inv_variances_q.shape == (7,)
    assert runtime.lut_values.shape == (128,)
    assert runtime.persistent_bytes > runtime.lut_values.nbytes


def test_runtime_accepts_only_integer_observations() -> None:
    runtime = _runtime()
    q = runtime.quantize_observation(np.linspace(0.0, 0.6, 7))
    costs = runtime.emission_costs_q(q)
    assert costs.shape == (4,)
    assert np.issubdtype(costs.dtype, np.integer)


def test_integer_runtime_is_deterministic() -> None:
    runtime = _runtime()
    observation = np.array([0.1, -0.3, 0.7, 0.0, 0.4, -0.2, 0.5])
    q = runtime.quantize_observation(observation)
    first = runtime.emission_costs_q(q)
    second = runtime.emission_costs_q(q.copy())
    assert np.array_equal(first, second)


def test_integer_lag8_decoder_uses_bounded_memory() -> None:
    decoder = IntegerLag8Decoder.compile(_q4(), StudentTCostLUT.build(128), lag=8)
    assert decoder.lag == 8
    assert decoder.runtime_buffer_bytes == 272
    assert decoder.core_bytes < 1024


def test_integer_lag8_returns_valid_path() -> None:
    decoder = IntegerLag8Decoder.compile(_q4(), StudentTCostLUT.build(128), lag=8)
    observations = np.tile(np.linspace(0.0, 0.6, 7), (24, 1))
    quantized = decoder.quantize_sequence(observations)
    path = decoder.decode_q(quantized)
    assert path.shape == (24,)
    assert np.all((path >= 0) & (path < 4))


def test_integer_lag8_is_deterministic() -> None:
    decoder = IntegerLag8Decoder.compile(_q4(), StudentTCostLUT.build(128), lag=8)
    observations = np.tile(np.linspace(-0.3, 0.4, 7), (20, 1))
    quantized = decoder.quantize_sequence(observations)
    first = decoder.decode_q(quantized)
    second = decoder.decode_q(quantized.copy())
    assert np.array_equal(first, second)
