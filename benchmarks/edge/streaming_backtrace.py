"""Reproducible synthetic LUT-128 bounded-backtrace experiment."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from resolutive_inference.fixed_point import StudentTLUT
from resolutive_inference.metrics import state_accuracy
from resolutive_inference.streaming import FixedPointViterbi

DEFAULT_SEEDS = (7, 23, 101, 2026, 9001)


def generate_sequence(length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a controlled three-state heavy-tailed synthetic sequence."""
    transition = np.array([[0.965, 0.025, 0.010], [0.020, 0.960, 0.020], [0.010, 0.025, 0.965]])
    means = np.array([[-3.0], [0.0], [3.0]])
    rng = np.random.default_rng(seed)
    states = np.empty(length, dtype=int)
    observations = np.empty((length, 1), dtype=float)
    states[0] = int(rng.integers(3))
    for index in range(length):
        if index:
            states[index] = rng.choice(3, p=transition[states[index - 1]])
        observations[index, 0] = means[states[index], 0] + rng.standard_t(df=3) * 0.55
    return observations, states


def run_experiment(*, length: int, seeds: tuple[int, ...]) -> list[dict[str, int | float | str]]:
    transition = np.array([[0.965, 0.025, 0.010], [0.020, 0.960, 0.020], [0.010, 0.025, 0.965]])
    means = np.array([[-3.0], [0.0], [3.0]])
    variances = np.full_like(means, 0.55**2)
    lut = StudentTLUT.build(128)
    persistent_model_bytes = 42  # deployment-format estimate; see docs/architecture.md
    rows: list[dict[str, int | float | str]] = []
    for seed in seeds:
        observations, truth = generate_sequence(length, seed)
        predictions: dict[str, np.ndarray] = {}
        for label, lag in (("full", None), ("lag16", 16), ("lag8", 8), ("lag4", 4)):
            decoder = FixedPointViterbi(transition, means, variances, lut=lut, lag=lag)
            started = time.perf_counter_ns()
            predicted = decoder.decode(observations)
            elapsed_ns = time.perf_counter_ns() - started
            predictions[label] = predicted
            runtime = 2 * 3 * 4 + (length if lag is None else lag) * 3
            total = persistent_model_bytes + lut.nbytes + runtime
            rows.append(
                {
                    "seed": seed,
                    "length": length,
                    "decoder": label,
                    "lag": length if lag is None else lag,
                    "accuracy": state_accuracy(truth, predicted),
                    "delta_vs_full": 0.0,
                    "persistent_model_bytes": persistent_model_bytes,
                    "lut_bytes": lut.nbytes,
                    "runtime_buffer_bytes": runtime,
                    "total_estimated_memory_bytes": total,
                    "latency_observations": length if lag is None else lag,
                    "operations_per_observation_proxy": decoder.operations_per_observation,
                    "elapsed_ns_per_observation": elapsed_ns / length,
                }
            )
        full_accuracy = state_accuracy(truth, predictions["full"])
        for row in rows[-4:]:
            row["delta_vs_full"] = float(row["accuracy"]) - full_accuracy
    return rows


def aggregate(rows: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    summary = []
    for decoder in ("full", "lag16", "lag8", "lag4"):
        selected = [row for row in rows if row["decoder"] == decoder]
        summary.append(
            {
                "decoder": decoder,
                "runs": len(selected),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in selected])),
                "mean_delta_vs_full": float(np.mean([row["delta_vs_full"] for row in selected])),
                "total_estimated_memory_bytes": selected[0]["total_estimated_memory_bytes"],
                "latency_observations": selected[0]["latency_observations"],
                "operations_per_observation_proxy": selected[0]["operations_per_observation_proxy"],
                "mean_elapsed_ns_per_observation": float(
                    np.mean([row["elapsed_ns_per_observation"] for row in selected])
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=2_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=Path("results/edge/streaming_backtrace"))
    args = parser.parse_args()
    rows = run_experiment(length=args.length, seeds=tuple(args.seeds))
    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    configuration = {"length": args.length, "seeds": args.seeds, "lut_entries": 128}
    (args.output_dir / "summary.json").write_text(
        json.dumps({"configuration": configuration, "summary": summary}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
