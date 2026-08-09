# Architecture

The core is decomposed into explicit state transitions, emission likelihoods, quantization, and metrics. `CompactPro` composes those primitives into an online Bayesian filtering step. This separation is intended to make assumptions auditable and allow each component to be replaced in controlled ablations.

The initial implementation uses finite latent states, a row-stochastic transition matrix, and diagonal Gaussian emissions. It is a research baseline rather than a finalized algorithm. Compact-Robust remains deliberately unimplemented until its estimator, contamination model, and validation criteria are specified.

Public APIs should remain small, typed, deterministic under fixed inputs, and independent of experiment orchestration. Benchmarks and experiments may depend on the core package; the core package must not depend on them.

## Edge reference path

The fixed-point reference uses an integer LUT to approximate the robust emission penalty
`log(1 + d/3)`. `FixedPointViterbi.update()` consumes exactly one observation and exposes a
finalized state only after the configured lag. Full backtrace is retained as the equivalence
reference, not as the intended bounded-memory deployment mode. Memory tables report separately:
model-value payload, LUT storage, and algorithmic runtime buffers; they do not estimate firmware,
Python objects, stack, alignment, scales, offsets, or acquisition buffers.

The planned Edge roadmap is:

```text
Python reference
      ↓
Quantized reference
      ↓
Fixed-point reference
      ↓
Streaming decoder
      ↓
Bounded backtrace
      ↓
C/C++ kernel
      ↓
ESP32/STM32 benchmark
      ↓
Real sensor experiment
```

For the current three-state, one-dimensional benchmark, persistent model storage is estimated as
42 bytes: 9 int16 transition scores, 3 int16 means, 3 uint16 inverse variances, and 3 int32
initial scores. This is a deployment-format estimate, not the size of the Python object. It is also
not the previously discussed 119-value Compact-Robust configuration; that estimator is absent
from the current repository and its earlier reported accuracies cannot yet be reproduced here.
