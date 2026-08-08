import numpy as np
import pytest

from resolutive_inference import CompactPro


def test_step_returns_normalized_posterior() -> None:
    model = CompactPro(
        n_states=2,
        observation_dim=1,
        transition=np.array([[0.9, 0.1], [0.2, 0.8]]),
        means=np.array([[0.0], [3.0]]),
    )

    posterior = model.step(np.array([0.1]))

    assert posterior.sum() == pytest.approx(1.0)
    assert posterior[0] > posterior[1]


def test_invalid_observation_shape_is_rejected() -> None:
    model = CompactPro(n_states=2, observation_dim=1)

    with pytest.raises(ValueError, match="observation"):
        model.step(np.array([1.0, 2.0]))
