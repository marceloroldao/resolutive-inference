import numpy as np
import pytest

from resolutive_inference.synthetic import generate_gaussian_hmm


def test_generator_is_reproducible_and_has_expected_shapes() -> None:
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    means = np.array([[0.0], [2.0]])
    variances = np.ones_like(means)

    first = generate_gaussian_hmm(50, transition, means, variances, seed=42)
    second = generate_gaussian_hmm(50, transition, means, variances, seed=42)

    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_allclose(first.observations, second.observations)
    assert first.observations.shape == (50, 1)
    assert first.states.shape == (50,)


def test_generator_rejects_zero_mass_transition_row() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        generate_gaussian_hmm(
            10,
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0], [1.0]]),
            np.ones((2, 1)),
            seed=1,
        )
