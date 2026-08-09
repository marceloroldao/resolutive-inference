import numpy as np

from benchmarks.edge.streaming_backtrace import generate_sequence, run_experiment
from resolutive_inference import FixedPointViterbi, StudentTLUT


def decoder(lag: int | None = None) -> FixedPointViterbi:
    return FixedPointViterbi(
        np.array([[0.95, 0.05], [0.05, 0.95]]),
        np.array([[-2.0], [2.0]]),
        np.ones((2, 1)),
        lut=StudentTLUT.build(128),
        lag=lag,
    )


def test_streaming_equals_batch_for_full_backtrace() -> None:
    observations = np.array([[-2.1], [-1.8], [1.9], [2.2]])
    model = decoder()
    expected = model.decode(observations)
    model.reset()
    for observation in observations:
        result = model.update(observation)
        assert result.finalized_state is None
    np.testing.assert_array_equal(model.flush(), expected)


def test_bounded_backtrace_emits_with_configured_lag() -> None:
    model = decoder(lag=4)
    for _ in range(4):
        assert model.update(np.array([-2.0])).finalized_state is None
    result = model.update(np.array([-2.0]))
    assert result.finalized_index == 0
    assert result.finalized_state == 0
    assert len(model.decode(np.full((20, 1), -2.0))) == 20


def test_seed_and_benchmark_are_deterministic_except_timing() -> None:
    first = generate_sequence(100, 23)
    second = generate_sequence(100, 23)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    rows_a = run_experiment(length=100, seeds=(23,))
    rows_b = run_experiment(length=100, seeds=(23,))
    for a, b in zip(rows_a, rows_b, strict=True):
        a.pop("elapsed_ns_per_observation")
        b.pop("elapsed_ns_per_observation")
        assert a == b
