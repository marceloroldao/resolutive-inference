"""Reproducible synthetic comparison for Compact-Robust 119/Q4/LUT-128.

This benchmark is synthetic exploratory evidence only. It does not reproduce the
conversation-local experiments and must generate its own results from repository code.
"""

import argparse
import json

import numpy as np

from resolutive_inference.compact_robust import CompactRobust119
from resolutive_inference.edge_compact import Q4CompactRobust119, StudentTCostLUT
from resolutive_inference.metrics import state_accuracy

DEFAULT_SEEDS = (7, 23, 101, 2026, 9001)
N_STATES = 4
OBS_DIM = 7

MEANS = np.array(
    [
        [-1.6, 0.1, -1.0, 0.7, -0.4, 0.8, 0.2],
        [-0.3, 1.5, 0.6, -0.8, 1.0, -0.2, 0.8],
        [1.3, -1.1, 1.1, 0.8, -0.9, 0.5, -0.6],
        [0.9, 0.4, -0.4, -1.4, 0.3, 1.3, 1.1],
    ],
    dtype=float,
)


def generate_dataset(seed: int, sequences: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((sequences, length, OBS_DIM), dtype=float)
    y = np.zeros((sequences, length), dtype=int)
    for sequence in range(sequences):
        state = int(rng.integers(N_STATES))
        for t in range(length):
            if rng.random() < 0.07:
                state = int(rng.choice([candidate for candidate in range(N_STATES) if candidate != state]))
            x[sequence, t] = MEANS[state] + rng.standard_t(df=3, size=OBS_DIM) * 0.8
            y[sequence, t] = state
    return x, y


def run_one(seed: int, sequences: int, length: int) -> dict[str, float | int]:
    x, y = generate_dataset(seed, sequences, length)
    split = max(4, int(sequences * 0.75))
    model = CompactRobust119.fit_supervised(x[:split], y[:split])
    q4 = Q4CompactRobust119.from_model(model)
    lut = StudentTCostLUT.build(128)

    float_pred = np.vstack([model.decode(sequence) for sequence in x[split:]])
    q4_pred = np.vstack([q4.decode(sequence) for sequence in x[split:]])
    q4_lut_pred = np.vstack([q4.decode(sequence, lut=lut) for sequence in x[split:]])
    truth = y[split:]

    return {
        "seed": seed,
        "float_accuracy": state_accuracy(truth.ravel(), float_pred.ravel()),
        "q4_accuracy": state_accuracy(truth.ravel(), q4_pred.ravel()),
        "q4_lut128_accuracy": state_accuracy(truth.ravel(), q4_lut_pred.ravel()),
        "statistic_count": model.statistic_count,
        "q4_payload_bits": q4.payload_bits,
        "q4_packed_bytes": q4.payload_bytes_packed,
        "affine_metadata_bytes_float32": q4.affine_metadata_bytes_float32,
        "lut_bytes": lut.nbytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, default=80)
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    rows = [run_one(seed, args.sequences, args.length) for seed in args.seeds]
    summary = {
        "configuration": {
            "sequences": args.sequences,
            "length": args.length,
            "seeds": args.seeds,
            "lut_entries": 128,
            "quantization_bits": 4,
        },
        "mean_float_accuracy": float(np.mean([row["float_accuracy"] for row in rows])),
        "mean_q4_accuracy": float(np.mean([row["q4_accuracy"] for row in rows])),
        "mean_q4_lut128_accuracy": float(
            np.mean([row["q4_lut128_accuracy"] for row in rows])
        ),
        "accounting": {
            "statistic_count": rows[0]["statistic_count"],
            "q4_payload_bits": rows[0]["q4_payload_bits"],
            "q4_packed_bytes": rows[0]["q4_packed_bytes"],
            "affine_metadata_bytes_float32": rows[0]["affine_metadata_bytes_float32"],
            "lut_bytes": rows[0]["lut_bytes"],
        },
        "runs": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
