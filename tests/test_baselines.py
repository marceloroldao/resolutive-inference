import numpy as np
import pytest

from benchmarks.baselines.robust119 import run
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
    for model in (
        GaussianHMM(transition, means, variances),
        StudentTHMM(transition, means, variances),
    ):
        posterior = model.step(np.array([0.25]))
        assert posterior.sum() == pytest.approx(1.0)
        assert posterior[0] > posterior[1]


def test_parameter_counts() -> None:
    transition = np.eye(4)
    means = np.zeros((4, 7))
    variances = np.ones_like(means)
    assert GaussianHMM(transition, means, variances).parameter_count == 72
    assert StudentTHMM(transition, means, variances).parameter_count == 73


def test_robust119_baseline_benchmark_smoke() -> None:
    result = run(length=40, seeds=1)
    assert result["seed_count"] == 1
    scenarios = result["scenarios"]
    assert set(scenarios) == {"first_order_gaussian", "second_order_contaminated"}
    for scenario in scenarios.values():
        for model_name in ("compact_robust119", "gaussian_hmm", "student_t_hmm"):
            summary = scenario[model_name]
            assert 0.0 <= summary["mean_accuracy"] <= 1.0
            assert summary["median_us_per_observation"] >= 0.0
