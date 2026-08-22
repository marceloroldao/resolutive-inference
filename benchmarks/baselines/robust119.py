"""Apples-to-apples known-parameter comparison for CompactRobust119.

The benchmark uses the same generated observations and known parameters for all
models. It reports two scenarios so second-order dynamics and robust emissions
are not conflated into one result.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from resolutive_inference.baselines import GaussianHMM, StudentTHMM
from resolutive_inference.compact_robust import CompactRobust119


def _parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.array(
        [
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5],
            [-1.0, 0.5, 1.5, -0.5, 1.0, -1.5, 0.0],
            [1.0, -0.5, -1.5, 0.5, -1.0, 1.5, 0.0],
        ],
        dtype=float,
    )
    shared_variances = np.array([0.55, 0.7, 0.6, 0.8, 0.65, 0.75, 0.7], dtype=float)
    transition = np.full((4, 4), 0.04, dtype=float)
    np.fill_diagonal(transition, 0.88)
    transition /= transition.sum(axis=1, keepdims=True)
    transition2 = np.empty((4, 4, 4), dtype=float)
    for a in range(4):
        for b in range(4):
            row = transition[b].copy()
            # modest second-order memory: returning to state a is more likely
            row[a] += 0.22
            transition2[a, b] = row / row.sum()
    initial = np.full(4, 0.25, dtype=float)
    return means, shared_variances, transition, transition2, initial


def _generate(
    *,
    length: int,
    seed: int,
    second_order: bool,
    contamination: float,
) -> tuple[np.ndarray, np.ndarray]:
    means, variances, transition, transition2, initial = _parameters()
    rng = np.random.default_rng(seed)
    states = np.empty(length, dtype=int)
    states[0] = rng.choice(4, p=initial)
    states[1] = rng.choice(4, p=transition[states[0]])
    for t in range(2, length):
        probabilities = transition2[states[t - 2], states[t - 1]] if second_order else transition[states[t - 1]]
        states[t] = rng.choice(4, p=probabilities)
    observations = rng.normal(means[states], np.sqrt(variances), size=(length, 7))
    if contamination > 0:
        mask = rng.random(length) < contamination
        if np.any(mask):
            observations[mask] += rng.standard_t(df=2.5, size=(int(mask.sum()), 7)) * 2.5
    return observations, states


def _accuracy(expected: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(expected == predicted))


def _evaluate_hmm(model: GaussianHMM, observations: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter_ns()
    posteriors = np.vstack([model.step(row) for row in observations])
    elapsed_us = (time.perf_counter_ns() - start) / 1_000.0
    return np.argmax(posteriors, axis=1), elapsed_us


def _scenario(*, length: int, seed: int, second_order: bool, contamination: float) -> dict[str, object]:
    means, variances, transition, transition2, initial = _parameters()
    observations, states = _generate(
        length=length,
        seed=seed,
        second_order=second_order,
        contamination=contamination,
    )
    robust = CompactRobust119(means, variances, transition, transition2, initial, degrees_of_freedom=3.0)
    gaussian = GaussianHMM(transition, means, np.tile(variances, (4, 1)))
    student = StudentTHMM(
        transition,
        means,
        np.tile(variances, (4, 1)),
        degrees_of_freedom=3.0,
    )

    start = time.perf_counter_ns()
    robust_path = robust.decode(observations)
    robust_us = (time.perf_counter_ns() - start) / 1_000.0
    gaussian_path, gaussian_us = _evaluate_hmm(gaussian, observations)
    student_path, student_us = _evaluate_hmm(student, observations)

    return {
        "length": length,
        "second_order_generator": second_order,
        "contamination_probability": contamination,
        "models": {
            "compact_robust119": {
                "accuracy": _accuracy(states, robust_path),
                "elapsed_us": robust_us,
                "us_per_observation": robust_us / length,
                "stored_statistics": robust.statistic_count,
            },
            "gaussian_hmm": {
                "accuracy": _accuracy(states, gaussian_path),
                "elapsed_us": gaussian_us,
                "us_per_observation": gaussian_us / length,
                "stored_parameters": gaussian.parameter_count,
            },
            "student_t_hmm": {
                "accuracy": _accuracy(states, student_path),
                "elapsed_us": student_us,
                "us_per_observation": student_us / length,
                "stored_parameters": student.parameter_count,
            },
        },
    }


def run(*, length: int, seeds: int) -> dict[str, object]:
    scenarios = {
        "first_order_gaussian": (False, 0.0),
        "second_order_contaminated": (True, 0.08),
    }
    output: dict[str, object] = {
        "methodology": "known-parameter synthetic controls; same observations and transition/emission parameters for all models",
        "length_per_seed": length,
        "seed_count": seeds,
        "scenarios": {},
    }
    for name, (second_order, contamination) in scenarios.items():
        runs = [
            _scenario(
                length=length,
                seed=2026 + seed,
                second_order=second_order,
                contamination=contamination,
            )
            for seed in range(seeds)
        ]
        summary: dict[str, object] = {"runs": runs}
        for model_name in ("compact_robust119", "gaussian_hmm", "student_t_hmm"):
            accuracies = [float(run["models"][model_name]["accuracy"]) for run in runs]  # type: ignore[index]
            timings = [float(run["models"][model_name]["us_per_observation"]) for run in runs]  # type: ignore[index]
            summary[model_name] = {
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "median_us_per_observation": float(np.median(timings)),
            }
        output["scenarios"][name] = summary  # type: ignore[index]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    if args.length < 20 or args.seeds < 1:
        raise SystemExit("length must be >=20 and seeds must be >=1")
    print(json.dumps(run(length=args.length, seeds=args.seeds), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
