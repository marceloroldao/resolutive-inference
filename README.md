# Resolutive Inference

Resolutive Inference is an experimental, compact engine for sequential inference. The project studies whether deliberately small state-space models can provide interpretable, reproducible baselines for filtering, regime inference, anomaly scoring, and compressed representation of time series.

The project does **not** claim general superiority over neural networks. Neural sequence models can be substantially more expressive and may be the appropriate choice when data, compute, and task complexity justify them. This repository instead investigates a narrower engineering and scientific question: what can be achieved with a small, inspectable inference state and an explicitly controlled statistical budget?

## Research lines

- **Compact-Pro** is the first reference implementation. Its intended operating point is approximately **119 learned parameters or maintained statistics**, depending on the configured state and observation dimensions. The exact count must be reported for every experiment rather than treated as a universal constant.
- **Compact-Robust** is a future experimental line aimed at heavy-tailed observations, contamination, and distribution shift. The current module is an explicit placeholder and is not presented as a validated method.

## Baselines

Benchmark comparisons will include conventional hidden Markov models (HMMs) and HMMs with Student-t emissions. Comparisons must use matched data splits, clearly documented parameter counts, repeated seeds, and uncertainty estimates. Neural baselines may be added when appropriate, but no comparison should imply a general architectural ranking beyond the evaluated tasks and budgets.

## Relationship to Resolutive Science

`resolutive-science` is the normative source of truth for shared Resolutive Science terminology, notation and scientific-status conventions.

- Resolutive Science repository baseline: `v0.1.1`
- RSMS compatibility: `1.0-rc.1` — candidate compatibility, subject to re-audit when RSMS 1.0 becomes stable
- Project governance baseline: `RSPS 1.0-draft`

Resolutive Inference is an independently testable computational project. Resolutive terminology used here is an engineering abstraction unless a direct mathematical dependency on RSMS is explicitly identified. Computational benchmark success does not constitute validation of Resolutive Physics.

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

The command emits machine-readable JSON containing the seed, sequence length, state accuracy, negative log likelihood, Brier score, change-detection summary, and exact stored-statistic count.

## Reproducibility

Every reported result should record the code revision, environment, configuration, random seeds, dataset provenance and checksum, split construction, fitted parameter/statistic count, runtime budget, and evaluation metrics. Experiments should preserve raw per-run measurements and summarize repeated trials with uncertainty intervals. See `docs/benchmark_protocol.md`.

## Status

**Maturity:** pre-alpha / early research scaffold.

APIs, algorithms, and claims are expected to change as evidence accumulates. This maturity designation should be retained until a reproducible release gate, frozen benchmark protocol and publication-readiness review are completed.

## Licensing

This repository is source-available for academic, educational and non-commercial research use under `LICENSE`. Commercial use requires separate written authorization or a commercial license from the rights holder. Because commercial use is restricted, the project must not be represented as OSI-approved open-source software.

## Citation

Citation metadata is provided in `CITATION.cff`. Cite the exact commit or future release used.

## Author

Marcelo Roldão Matos  
ORCID: 0009-0003-6075-4680
