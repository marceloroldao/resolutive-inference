# Resolutive Inference

Resolutive Inference is an experimental, compact engine for sequential inference. The project studies whether deliberately small state-space models can provide interpretable, reproducible baselines for filtering, regime inference, anomaly scoring, and compressed representation of time series.

The project does **not** claim general superiority over neural networks. Neural sequence models can be substantially more expressive and may be the appropriate choice when data, compute, and task complexity justify them. This repository instead investigates a narrower engineering and scientific question: what can be achieved with a small, inspectable inference state and an explicitly controlled statistical budget?

## Research lines

- **Compact-Pro** is the first reference implementation. Its intended operating point is approximately **119 learned parameters or maintained statistics**, depending on the configured state and observation dimensions. The exact count must be reported for every experiment rather than treated as a universal constant.
- **Compact-Robust** remains an experimental line. The repository now includes a narrow fixed-point Student-t-like emission and streaming Viterbi decoder for controlled Edge experiments; the general `CompactRobust` estimator remains a placeholder and is not presented as validated.

## Baselines

Benchmark comparisons will include conventional hidden Markov models (HMMs) and HMMs with Student-t emissions. Comparisons must use matched data splits, clearly documented parameter counts, repeated seeds, and uncertainty estimates. Neural baselines may be added when appropriate, but no comparison should imply a general architectural ranking beyond the evaluated tasks and budgets.

## Layout

```text
src/resolutive_inference/   Core inference components
benchmarks/synthetic/       Controlled synthetic benchmarks
benchmarks/baselines/       HMM and Student-t HMM baselines
experiments/                Compression, robustness, and validation studies
tests/                      Automated checks
docs/                       Architecture, methodology, and benchmark protocol
results/                    Generated tables and figures (not source evidence)
examples/                   Minimal usage examples
```

## Quick start

```bash
python -m pip install -e .
python -m pytest
```

Run the initial known-parameter synthetic benchmark with:

```bash
python benchmarks/synthetic/run.py --length 1000 --seed 2026
```

Run the reproducible exploratory bounded-backtrace benchmark with:

```bash
python -m benchmarks.edge.streaming_backtrace
```

It writes per-seed CSV and an aggregate JSON summary under `results/edge/streaming_backtrace/`. Its memory estimates separate the theoretical Q4 model-value payload, LUT, and decoder buffers; they are not whole-firmware measurements.

The initial benchmark command emits machine-readable JSON containing the seed, sequence length,
state accuracy, negative log likelihood, Brier score, change-detection summary,
and exact stored-statistic count.

```python
import numpy as np
from resolutive_inference import CompactPro

model = CompactPro(n_states=2, observation_dim=1)
posterior = model.step(np.array([0.25]))
print(posterior)
```

## Reproducibility

Every reported result should record the code revision, environment, configuration, random seeds, dataset provenance and checksum, split construction, fitted parameter/statistic count, runtime budget, and evaluation metrics. Experiments should preserve raw per-run measurements and summarize repeated trials with uncertainty intervals. See [the benchmark protocol](docs/benchmark_protocol.md).

## Evidence terminology

**Synthetic results** come from controlled generators. **Empirical results** are reserved for independent external datasets with provenance. Synthetic benchmarks are not described as real-world validation.

## Status

This repository is an early research scaffold. APIs, algorithms, and claims are expected to change as evidence accumulates.
