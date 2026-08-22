import numpy as np

from resolutive_inference.compact_robust import (
    OBSERVATION_DIM,
    REFERENCE_STATISTIC_COUNT,
    CompactRobust119,
)


def _model() -> CompactRobust119:
    return CompactRobust119(
        means=np.zeros((4, OBSERVATION_DIM)),
        shared_variances=np.ones(OBSERVATION_DIM),
        transition=np.ones((4, 4)),
        transition2=np.ones((4, 4, 4)),
        initial=np.ones(4),
    )


def test_reference_stores_exactly_119_statistics() -> None:
    model = _model()
    assert model.statistic_count == REFERENCE_STATISTIC_COUNT == 119


def test_q4_payload_accounting_is_explicit() -> None:
    model = _model()
    assert model.q4_payload_bits == 476
    assert model.q4_payload_bytes_theoretical == 59.5
    assert model.q4_payload_bytes_packed == 60


def test_decode_returns_one_state_per_observation() -> None:
    model = _model()
    observations = np.zeros((8, OBSERVATION_DIM))
    decoded = model.decode(observations)
    assert decoded.shape == (8,)
    assert np.all((decoded >= 0) & (decoded < 4))


def test_supervised_fit_is_deterministic() -> None:
    observations = np.zeros((4, 6, OBSERVATION_DIM), dtype=float)
    states = np.vstack([np.full(6, state, dtype=int) for state in range(4)])
    for state in range(4):
        observations[state] = state

    first = CompactRobust119.fit_supervised(observations, states)
    second = CompactRobust119.fit_supervised(observations, states)

    assert np.array_equal(first.means, second.means)
    assert np.array_equal(first.transition, second.transition)
    assert np.array_equal(first.transition2, second.transition2)
