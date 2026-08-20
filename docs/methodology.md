# Methodology

The project separates model definition, compilation, runtime inference, and evaluation so that each approximation can be audited independently.

## Reference-first development

A floating-point reference is established before introducing quantization or bounded memory. Approximation layers are added one at a time and benchmarked against the immediately preceding reference.

For the current Edge line the chain is:

```text
Compact-Robust 119 float
→ Q4 parameters
→ Student-t-like LUT-128
→ bounded second-order lag
→ integer-only lag-8 runtime
```

## Compile/runtime boundary

The integer Edge work distinguishes **offline compilation** from **runtime inference**.

Offline compilation may use floating point to transform the research model into integer tables. This stage produces fixed-point means, reciprocal-variance factors, transition costs and the LUT.

The integer-runtime claim begins only after an observation has been quantized to the declared `int16` input scale. Runtime then uses integer differences, products, divisions, table indexing, additions, comparisons, score renormalization and predecessor storage.

A convenience Python method may quantize floating-point observations for experiments, but that preprocessing must not be presented as part of the integer-only runtime.

## Memory accounting

Memory is reported in categories:

- research-model payload;
- quantization metadata;
- compiled runtime tables;
- bounded inference buffers;
- executable/platform memory when eventually measured on hardware.

Python object/container overhead is excluded from algorithmic reference byte counts and must never be conflated with measured MCU RAM.

## Synthetic evidence

Synthetic experiments are used to test determinism, robustness, quantization error, fixed-lag behavior and implementation regressions. They are not a substitute for external datasets or target-hardware measurements.

A result is considered repository-reproduced only when the benchmark can be run from a complete checkout with its configuration and seeds recorded. Conversation-local or independently reconstructed experiments are exploratory evidence and should remain labeled as such.

## Negative results

Accuracy loss, unstable quantization, memory regressions and cases where conventional baselines outperform the Resolutive reference should be preserved when scientifically relevant. The project should not tune away negative results solely to improve presentation.
