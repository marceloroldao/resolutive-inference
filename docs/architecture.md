# Architecture

The core is decomposed into explicit state transitions, emission likelihoods, quantization, and metrics. `CompactPro` composes those primitives into an online Bayesian filtering step. This separation is intended to make assumptions auditable and allow each component to be replaced in controlled ablations.

The initial implementation uses finite latent states, a row-stochastic transition matrix, and diagonal Gaussian emissions. It is a research baseline rather than a finalized algorithm. Compact-Robust remains deliberately unimplemented until its estimator, contamination model, and validation criteria are specified.

Public APIs should remain small, typed, deterministic under fixed inputs, and independent of experiment orchestration. Benchmarks and experiments may depend on the core package; the core package must not depend on them.
