import numpy as np

from resolutive_inference.compact_robust import CompactRobust119
from resolutive_inference.edge_compact import Q4CompactRobust119, StudentTCostLUT


def _float_model() -> CompactRobust119:
    means = np.arange(28, dtype=float).reshape(4, 7) / 10.0
    return CompactRobust119(
        means=means,
        shared_variances=np.linspace(0.5, 1.1, 7),
        transition=np.eye(4) + 0.1,
        transition2=np.ones((4, 4, 4)) + np.eye(4)[:, None, :],
        initial=np.array([0.4, 0.3, 0.2, 0.1]),
    )


def test_q4_reference_has_expected_payload_and_metadata_accounting() -> None:
    q4 = Q4CompactRobust119.from_model(_float_model())
    assert q4.payload_bits == 476
    assert q4.payload_bytes_theoretical == 59.5
    assert q4.payload_bytes_packed == 60
    assert q4.affine_metadata_bytes_float32 == 40


def test_lut128_uses_256_bytes() -> None:
    lut = StudentTCostLUT.build(128)
    assert lut.values.shape == (128,)
    assert lut.values.dtype == np.uint16
    assert lut.nbytes == 256


def test_q4_lut_decode_returns_valid_path() -> None:
    model = _float_model()
    q4 = Q4CompactRobust119.from_model(model)
    lut = StudentTCostLUT.build(128)
    observations = np.tile(np.linspace(0.0, 0.6, 7), (10, 1))
    path = q4.decode(observations, lut=lut)
    assert path.shape == (10,)
    assert np.all((path >= 0) & (path < 4))
