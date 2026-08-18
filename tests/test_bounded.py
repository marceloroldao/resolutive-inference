import numpy as np

from resolutive_inference.bounded import BoundedSecondOrderDecoder
from resolutive_inference.compact_robust import CompactRobust119
from resolutive_inference.edge_compact import Q4CompactRobust119, StudentTCostLUT


def _q4() -> Q4CompactRobust119:
    model = CompactRobust119(
        means=np.arange(28, dtype=float).reshape(4, 7) / 10.0,
        shared_variances=np.ones(7),
        transition=np.eye(4) + 0.2,
        transition2=np.ones((4, 4, 4)) + np.eye(4)[:, None, :],
        initial=np.ones(4),
    )
    return Q4CompactRobust119.from_model(model)


def test_bounded_runtime_buffer_accounting() -> None:
    q4 = _q4()
    assert BoundedSecondOrderDecoder(q4, lag=4).runtime_buffer_bytes == 192
    assert BoundedSecondOrderDecoder(q4, lag=8).runtime_buffer_bytes == 256
    assert BoundedSecondOrderDecoder(q4, lag=16).runtime_buffer_bytes == 384


def test_long_lag_matches_full_reference() -> None:
    q4 = _q4()
    lut = StudentTCostLUT.build(128)
    observations = np.tile(np.linspace(0.0, 0.6, 7), (10, 1))
    full = q4.decode(observations, lut=lut)
    bounded = BoundedSecondOrderDecoder(q4, lag=16, lut=lut).decode(observations)
    assert np.array_equal(bounded, full)


def test_lag_4_8_16_return_aligned_paths() -> None:
    q4 = _q4()
    lut = StudentTCostLUT.build(128)
    observations = np.tile(np.linspace(0.0, 0.6, 7), (24, 1))
    for lag in (4, 8, 16):
        path = BoundedSecondOrderDecoder(q4, lag=lag, lut=lut).decode(observations)
        assert path.shape == (24,)
        assert np.all((path >= 0) & (path < 4))
