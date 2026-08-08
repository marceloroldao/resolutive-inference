"""Compare known-parameter sequential filters on one seeded sequence."""

import argparse
import json

import numpy as np

from resolutive_inference import CompactPro
from resolutive_inference.baselines import GaussianHMM, StudentTHMM
from resolutive_inference.metrics import mean_brier_score, state_accuracy
from resolutive_inference.synthetic import generate_gaussian_hmm


def evaluate(model: object, observations: np.ndarray, states: np.ndarray) -> dict[str, float]:
    """Evaluate a model exposing a sequential ``step`` method."""
    posteriors = np.vstack([model.step(value) for value in observations])  # type: ignore[attr-defined]
    predictions = np.argmax(posteriors, axis=1)
    return {
        "state_accuracy": state_accuracy(states, predictions),
        "mean_brier_score": mean_brier_score(states, posteriors),
    }


def run_comparison(*, length: int, seed: int) -> dict[str, object]:
    transition = np.array([[0.97, 0.03], [0.05, 0.95]])
    means = np.array([[-1.5], [1.5]])
    variances = np.ones((2, 1))
    sequence = generate_gaussian_hmm(length, transition, means, variances, seed=seed)
    models = {
        "compact_pro": CompactPro(2, 1, transition, means, variances),
        "gaussian_hmm": GaussianHMM(transition, means, variances),
        "student_t_hmm": StudentTHMM(transition, means, variances, degrees_of_freedom=5.0),
    }
    return {
        "scenario": "known_parameter_gaussian_hmm",
        "seed": seed,
        "length": length,
        "results": {
            name: evaluate(model, sequence.observations, sequence.states)
            for name, model in models.items()
        },
        "parameter_counts": {
            "compact_pro_stored_statistics": models["compact_pro"].statistic_count,
            "gaussian_hmm": models["gaussian_hmm"].parameter_count,
            "student_t_hmm": models["student_t_hmm"].parameter_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run_comparison(length=args.length, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
