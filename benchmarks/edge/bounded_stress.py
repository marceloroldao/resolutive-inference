"""Stress benchmark for fixed-lag second-order decoding.

Synthetic exploratory evidence only. The scenarios deliberately reduce emission
separability and add switching or burst contamination so lag 16/8/4 can be
compared against full Q4+LUT-128 decoding under temporal ambiguity.
"""

import argparse
import json

import numpy as np

from resolutive_inference.bounded import BoundedSecondOrderDecoder
from resolutive_inference.compact_robust import CompactRobust119
from resolutive_inference.edge_compact import Q4CompactRobust119, StudentTCostLUT
from resolutive_inference.metrics import state_accuracy

SEEDS = (7, 23, 101, 2026, 9001, 12345)
LAGS = (16, 8, 4, 2, 1)
SCENARIOS = ("ambiguous", "switching", "bursty")
N_STATES = 4
OBS_DIM = 7

BASE_MEANS = np.array(
    [
        [-1.6, 0.1, -1.0, 0.7, -0.4, 0.8, 0.2],
        [-0.3, 1.5, 0.6, -0.8, 1.0, -0.2, 0.8],
        [1.3, -1.1, 1.1, 0.8, -0.9, 0.5, -0.6],
        [0.9, 0.4, -0.4, -1.4, 0.3, 1.3, 1.1],
    ],
    dtype=float,
)


def generate_dataset(
    seed: int,
    scenario: str,
    sequences: int = 120,
    length: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")

    rng = np.random.default_rng(seed)
    mean_scale = {"ambiguous": 0.45, "switching": 0.60, "bursty": 0.55}[scenario]
    means = BASE_MEANS * mean_scale
    x = np.zeros((sequences, length, OBS_DIM), dtype=float)
    y = np.zeros((sequences, length), dtype=int)

    for sequence in range(sequences):
        state = int(rng.integers(N_STATES))
        for t in range(length):
            if scenario == "ambiguous":
                change_probability = 0.11
            elif scenario == "switching":
                change_probability = 0.05 if (t // 8) % 2 == 0 else 0.18
            else:
                change_probability = 0.08

            if rng.random() < change_probability:
                candidates = [candidate for candidate in range(N_STATES) if candidate != state]
                state = int(rng.choice(candidates))

            noise_scale = 0.95 if scenario != "bursty" else 0.75
            observation = means[state] + rng.standard_t(df=3, size=OBS_DIM) * noise_scale
            if scenario == "bursty" and rng.random() < 0.12:
                observation += rng.normal(0.0, 2.2, OBS_DIM)

            x[sequence, t] = observation
            y[sequence, t] = state

    return x, y


def run_one(seed: int, scenario: str, sequences: int, length: int) -> dict[str, object]:
    x, y = generate_dataset(seed, scenario, sequences, length)
    split = int(sequences * 0.75)
    model = CompactRobust119.fit_supervised(x[:split], y[:split])
    q4 = Q4CompactRobust119.from_model(model)
    lut = StudentTCostLUT.build(128)
    truth = y[split:]

    full_prediction = np.vstack([q4.decode(sequence, lut=lut) for sequence in x[split:]])
    full_accuracy = state_accuracy(truth.ravel(), full_prediction.ravel())

    bounded: dict[str, dict[str, float | int]] = {}
    for lag in LAGS:
        decoder = BoundedSecondOrderDecoder(q4, lag=lag, lut=lut)
        prediction = np.vstack([decoder.decode(sequence) for sequence in x[split:]])
        accuracy = state_accuracy(truth.ravel(), prediction.ravel())
        bounded[f"lag{lag}"] = {
            "accuracy": accuracy,
            "delta_vs_full": accuracy - full_accuracy,
            "runtime_buffer_bytes": decoder.runtime_buffer_bytes,
        }

    return {
        "scenario": scenario,
        "seed": seed,
        "full_accuracy": full_accuracy,
        "bounded": bounded,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=120)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()

    rows = [
        run_one(seed, scenario, args.sequences, args.length)
        for scenario in SCENARIOS
        for seed in args.seeds
    ]

    scenarios: dict[str, object] = {}
    for scenario in SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        lag_summary = {}
        for lag in LAGS:
            key = f"lag{lag}"
            lag_summary[key] = {
                "mean_accuracy": float(
                    np.mean([row["bounded"][key]["accuracy"] for row in selected])
                ),
                "mean_delta_vs_full": float(
                    np.mean([row["bounded"][key]["delta_vs_full"] for row in selected])
                ),
                "runtime_buffer_bytes": selected[0]["bounded"][key]["runtime_buffer_bytes"],
            }
        scenarios[scenario] = {
            "mean_full_accuracy": float(np.mean([row["full_accuracy"] for row in selected])),
            "lags": lag_summary,
        }

    print(
        json.dumps(
            {
                "configuration": {
                    "sequences": args.sequences,
                    "length": args.length,
                    "seeds": args.seeds,
                    "scenarios": list(SCENARIOS),
                    "lags": list(LAGS),
                },
                "summary": scenarios,
                "runs": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
