import numpy as np
import pytest

from resolutive_inference.baselines import GaussianHMM, StudentTHMM, student_t_log_likelihood


def test_standard_cauchy_density_at_zero() -> None:
    actual = student_t_log_likelihood(
        np.array([0.0]), np.array([[0.0]]), np.array([[1.0]]), 1.0
    )

    assert actual[0] == pytest.approx(-np.log(np.pi))


def test_baseline_posteriors_are_normalized() -> None:
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    means = np.array([[0.0], [2.0]])
    variances = np.ones_like(means)

    for model in (GaussianHMM(transition, means, variances), StudentTHMM(transition, means, variances)):
        posterior = model.step(np.array([0.25]))
        assert posterior.sum() == pytest.approx(1.0)
        assert posterior[0] > posterior[1]


def test_student_t_emission_penalizes_extreme_value_less_than_gaussian() -> None:
    transition = np.eye(2)
    means = np.array([[0.0], [1.0]])
    variances = np.ones_like(means)
    gaussian = GaussianHMM(transition, means, variances)
    student = StudentTHMM(transition, means, variances, degrees_of_freedom=3.0)
    observation = np.array([20.0])

    gaussian_log_density = gaussian._emission_log_likelihood(observation)[0]
    student_log_density = student._emission_log_likelihood(observation)[0]

    assert student_log_density > gaussian_log_density


def test_parameter_counts_distinguish_persistent_model_parameters() -> None:
    transition = np.eye(3)
    means = np.zeros((3, 2))
    variances = np.ones_like(means)

    assert GaussianHMM(transition, means, variances).parameter_count == 21
    assert StudentTHMM(transition, means, variances).parameter_count == 22
