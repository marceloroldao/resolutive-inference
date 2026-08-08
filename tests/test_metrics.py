import numpy as np
import pytest

from resolutive_inference.metrics import (
    change_detection_delays,
    mean_brier_score,
    mean_negative_log_likelihood,
    state_accuracy,
)


def test_state_metrics_match_hand_calculation() -> None:
    truth = np.array([0, 0, 1, 1])
    posteriors = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
    predictions = np.argmax(posteriors, axis=1)

    assert state_accuracy(truth, predictions) == pytest.approx(0.75)
    assert mean_brier_score(truth, posteriors) == pytest.approx(0.45)
    assert mean_negative_log_likelihood(posteriors[np.arange(4), truth]) > 0


def test_change_detection_delays_handles_detection_and_censoring() -> None:
    truth = np.array([0, 0, 1, 1, 1, 0, 0])
    predictions = np.array([0, 0, 0, 1, 1, 1, 1])

    np.testing.assert_array_equal(change_detection_delays(truth, predictions), np.array([1, 2]))
    np.testing.assert_array_equal(
        change_detection_delays(truth, predictions, horizon=1), np.array([1, 1])
    )


def test_brier_score_rejects_unnormalized_posteriors() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        mean_brier_score(np.array([0]), np.array([[0.7, 0.4]]))
