import numpy as np

from resolutive_inference.quantization import uniform_quantize


def test_uniform_quantize_clips_and_maps_to_grid() -> None:
    actual = uniform_quantize(np.array([-2.0, 0.4, 2.0]), 3, -1.0, 1.0)
    np.testing.assert_allclose(actual, np.array([-1.0, 0.0, 1.0]))
