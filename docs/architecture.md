# Architecture

The core is decomposed into explicit state transitions, emission likelihoods, quantization, and metrics. `CompactPro` composes those primitives into an online Bayesian filtering step. This separation is intended to make assumptions auditable and allow each component to be replaced in controlled ablations.

The initial implementation uses finite latent states, a row-stochastic transition matrix, and diagonal Gaussian emissions. It is a research baseline rather than a finalized algorithm. Compact-Robust remains deliberately unimplemented until its estimator, contamination model, and validation criteria are specified.

Public APIs should remain small, typed, deterministic under fixed inputs, and independent of experiment orchestration. Benchmarks and experiments may depend on the core package; the core package must not depend on them.

## Hybrid quantized decoder

The historically named `FixedPointViterbi` is a hybrid reference rather than
an integer-only implementation. Scores, transitions, and LUT entries are
quantized/integer values, but the emission distance is calculated in floating
point before LUT indexing. It must not be described as a fixed-point
integer-only MCU kernel.

The 3-state/1D streaming benchmark exercises bounded-backtrace behavior and
buffer accounting only. It does not reproduce a Compact-Robust configuration
of approximately 119 maintained values.

## Implementation roadmap

The next reference is **Compact-Robust 119-value reference**, progressing
through Q4, LUT-128, bounded backtrace 16/8/4, integer-only emission, and
finally a C/C++ MCU kernel.
