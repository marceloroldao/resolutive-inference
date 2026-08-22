"""Synthetic comparison of hybrid Q4/LUT-128 and integer-only lag-8 runtime.

The float-to-int observation conversion is treated as preprocessing. The measured
integer decoder receives int16 observations and uses integer emission/transition
scoring at runtime. Results are synthetic exploratory evidence only.
"""

import argparse
import json

import numpy as np

from benchmarks.edge.compact_robust_q4 import DEFAULT_SEEDS, generate_dataset
from resolutive_inference.compact_robust import CompactRobust119
from resolutive_inference.edge_compact import Q4CompactRobust119, StudentTCostLUT
from resolutive_inference.integer_runtime import IntegerLag8Decoder
from resolutive_inference.metrics import state_accuracy


def run_one(seed: int, sequences: int, length: int) -> dict[str, float | int]:
    x, y = generate_dataset(seed, sequences, length)
    split = max(4, int(sequences * 0.75))
    model = CompactRobust119.fit_supervised(x[:split], y[:split])
    q4 = Q4CompactRobust119.from_model(model)
    lut = StudentTCostLUT.build(128)
    integer = IntegerLag8Decoder.compile(q4, lut, lag=8)

    hybrid_pred = np.vstack([q4.decode(sequence, lut=lut) for sequence in x[split:]])
    integer_pred = np.vstack(
        [integer.decode_q(integer.quantize_sequence(sequence)) for sequence in x[split:]]
    )
    truth = y[split:]

    return {
        "seed": seed,
        "hybrid_q4_lut128_accuracy": state_accuracy(truth.ravel(), hybrid_pred.ravel()),
        "integer_lag8_accuracy": state_accuracy(truth.ravel(), integer_pred.ravel()),
        "path_agreement": state_accuracy(hybrid_pred.ravel(), integer_pred.ravel()),
        "integer_persistent_bytes": integer.persistent_bytes,
        "integer_runtime_buffer_bytes": integer.runtime_buffer_bytes,
        "integer_core_bytes": integer.core_bytes,
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
            "lag": 8,
            "observation_runtime_dtype": "int16",
        },
        "mean_hybrid_q4_lut128_accuracy": float(
            np.mean([row["hybrid_q4_lut128_accuracy"] for row in rows])
        ),
        "mean_integer_lag8_accuracy": float(
            np.mean([row["integer_lag8_accuracy"] for row in rows])
        ),
        "mean_path_agreement": float(np.mean([row["path_agreement"] for row in rows])),
        "accounting": {
            "integer_persistent_bytes": rows[0]["integer_persistent_bytes"],
            "integer_runtime_buffer_bytes": rows[0]["integer_runtime_buffer_bytes"],
            "integer_core_bytes": rows[0]["integer_core_bytes"],
        },
        "runs": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
