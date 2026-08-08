"""Run the initial known-parameter Compact-Pro synthetic benchmark."""

import argparse
import json

import numpy as np

from resolutive_inference import CompactPro
from resolutive_inference.metrics import (
    change_detection_delays,
    mean_brier_score,
    mean_negative_log_likelihood,
    state_accuracy,
)
from resolutive_inference.synthetic import generate_gaussian_hmm


def run_benchmark(*, length: int, seed: int) -> dict[str, float | int]:
    """Evaluate filtering under the exact data-generating parameters."""
    transition = np.array([[0.97, 0.03], [0.05, 0.95]])
    means = np.array([[-1.5], [1.5]])
    variances = np.ones((2, 1))
    sequence = generate_gaussian_hmm(
        length, transition, means, variances, seed=seed, initial=np.array([0.5, 0.5])
    )
    model = CompactPro(
        n_states=2,
        observation_dim=1,
        transition=transition,
        means=means,
        variances=variances,
    )
    posteriors = np.vstack([model.step(value) for value in sequence.observations])
    predictions = np.argmax(posteriors, axis=1)
    assigned = posteriors[np.arange(length), sequence.states]
    delays = change_detection_delays(sequence.states, predictions)
    return {
        "seed": seed,
        "length": length,
        "statistic_count": model.statistic_count,
        "state_accuracy": state_accuracy(sequence.states, predictions),
        "mean_negative_log_likelihood": mean_negative_log_likelihood(assigned),
        "mean_brier_score": mean_brier_score(sequence.states, posteriors),
        "change_count": int(delays.size),
        "mean_detection_delay": float(np.mean(delays)) if delays.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(length=args.length, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
