import numpy as np

from resolutive_inference.fixed_point import StudentTLUT, quantize_q


def test_lut_is_monotonic_and_has_declared_memory() -> None:
    lut = StudentTLUT.build(128)
    assert lut.nbytes == 256
    assert np.all(np.diff(lut.values.astype(int)) >= 0)
    assert lut.lookup(0.0) == 0


def test_fixed_point_quantization_clips_deterministically() -> None:
    actual = quantize_q(np.array([-200.0, 0.5, 200.0]), fractional_bits=8, storage_bits=16)
    np.testing.assert_array_equal(actual, np.array([-32768, 128, 32767], dtype=np.int16))
