import numpy as np

from benchmarks.edge.streaming_backtrace import (
    BACKTRACE_DEPTH,
    LUT_SIZE,
    N_STATES,
    manual_runtime_buffer_bytes,
)
from resolutive_inference.fixed_point import FixedPointViterbi


def test_runtime_buffer_bytes_matches_manual_benchmark_estimate() -> None:
    model = FixedPointViterbi(
        means=np.zeros((N_STATES, 1)),
        transition_scores=np.zeros((N_STATES, N_STATES), dtype=np.int16),
        emission_lut=np.zeros(LUT_SIZE, dtype=np.int16),
        backtrace_depth=BACKTRACE_DEPTH,
    )

    assert model.runtime_buffer_bytes == manual_runtime_buffer_bytes()
    assert model.runtime_buffer_bytes == 72
