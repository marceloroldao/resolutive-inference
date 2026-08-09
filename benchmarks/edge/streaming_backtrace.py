"""Preliminary 3-state/1D streaming and bounded-backtrace benchmark.

This is not a reproduction of the approximately 119-value Compact-Robust
reference configuration, which has not yet been implemented.
"""

import json
import time

import numpy as np

from resolutive_inference.fixed_point import FixedPointViterbi

N_STATES = 3
OBSERVATION_DIM = 1
LUT_SIZE = 128
BACKTRACE_DEPTH = 16


def manual_runtime_buffer_bytes() -> int:
    """Independently account for the mutable buffers used by this benchmark."""
    score_bytes = 2 * N_STATES * np.dtype(np.int32).itemsize
    # Three states fit in the uint8 indices selected by the implementation.
    backpointer_bytes = BACKTRACE_DEPTH * N_STATES * np.dtype(np.uint8).itemsize
    return score_bytes + backpointer_bytes


def run_benchmark(length: int = 10_000, seed: int = 2026) -> dict[str, int | float | str]:
    """Run the preliminary streaming benchmark and return reproduced metrics."""
    model = FixedPointViterbi(
        means=np.array([[-2.0], [0.0], [2.0]]),
        transition_scores=np.array([[0, -24, -32], [-24, 0, -24], [-32, -24, 0]]),
        emission_lut=np.rint(np.linspace(0, -512, LUT_SIZE)).astype(np.int16),
        backtrace_depth=BACKTRACE_DEPTH,
    )
    observations = np.random.default_rng(seed).normal(size=(length, OBSERVATION_DIM))
    started = time.perf_counter()
    for observation in observations:
        model.step(observation)
    elapsed = time.perf_counter() - started
    manual_bytes = manual_runtime_buffer_bytes()
    if manual_bytes != model.runtime_buffer_bytes:
        raise RuntimeError("manual memory estimate and runtime buffers disagree")
    return {
        "benchmark": "preliminary-3-state-1d-streaming-bounded-backtrace",
        "length": length,
        "runtime_buffer_bytes": model.runtime_buffer_bytes,
        "manual_runtime_buffer_bytes": manual_bytes,
        "seconds": elapsed,
        "samples_per_second": length / elapsed,
        "seed": seed,
    }


def main() -> None:
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
