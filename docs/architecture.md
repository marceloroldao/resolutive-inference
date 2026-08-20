# Architecture

The core is decomposed into explicit state transitions, emission likelihoods, quantization, metrics, and experiment orchestration. Public APIs should remain small, typed, deterministic under fixed inputs, and independent of benchmark code.

## Compact-Pro

`CompactPro` is the initial finite-state Gaussian research baseline. It is intentionally conventional and auditable.

## Compact-Robust 119

The Edge reference fixes four latent states and seven observed features. The stored model statistics are:

- means: `4 x 7 = 28`;
- shared diagonal variances: `7`;
- first-order transition table: `4 x 4 = 16`;
- second-order transition table: `4 x 4 x 4 = 64`;
- initial probabilities: `4`.

Total: **119 scalar statistics**.

The full reference uses Student-t-like emission costs and second-order dynamic programming. Fitting is supervised at this stage so that the estimator is not mixed with the decoder while the Edge approximation chain is being audited.

## Q4 representation

Each of the five parameter tensors is uniformly quantized to four bits. Payload accounting and reconstruction metadata are deliberately separate. The 119 codes occupy 476 theoretical bits (59.5 bytes), or 60 byte-aligned bytes under ideal nibble packing. Five affine tensors currently require lower/scale metadata separately.

## LUT-128

A 128-entry `uint16` lookup table approximates the Student-t-like robust cost. The hybrid Python Q4/LUT decoder still calculates normalized observation distance in floating point before LUT indexing, so it is not an integer-only MCU kernel.

## Bounded second-order decoding

`BoundedSecondOrderDecoder` constrains predecessor history to a fixed lag. Its algorithmic score/backpointer budget is:

```text
2 x (4 x 4) int32 score planes
+ lag x (4 x 4) uint8 predecessor ring
```

Lags 16, 8 and 4 are primary candidates; lag 2/1 are retained for stress studies. Easy synthetic data can hide the effect of bounded history, therefore the repository includes ambiguous/noisy switching scenarios.

## Integer-only runtime contract

`IntegerEmissionRuntime` and `IntegerLag8Decoder` introduce a strict compile/runtime boundary.

Compilation may use the floating/Q4 research representation to construct:

- `int16` state means;
- `uint16` reciprocal variance factors;
- `uint16` Student-t-like LUT entries;
- integer initial, first-order and second-order transition costs.

Runtime consumes pre-quantized `int16` observations. Emission squared differences, reciprocal-variance weighting, LUT indexing, transition accumulation, minimum selection, score renormalization, predecessor storage and lag-8 traceback are integer operations. A convenience float input quantizer exists only for experiments and is excluded from the integer-runtime claim.

Reference core accounting for lag 8 is currently:

```text
compiled persistent integer tables : 510 B
algorithmic runtime buffers         : 272 B
------------------------------------------
core reference data                 : 782 B
```

This excludes executable code, stack, allocator/container overhead, Python object overhead, I/O buffers, preprocessing buffers and MCU/platform-specific state. It therefore must not be described as total firmware RAM.

## Implementation roadmap

```text
Compact-Pro
    ↓
Compact-Robust 119
    ↓
Q4
    ↓
LUT-128
    ↓
bounded lag 16 / 8 / 4
    ↓
integer-only lag-8 Python reference  ← current
    ↓
C/C++ fixed-point kernel             ← next gated step
    ↓
ESP32 / STM32 measurement
    ↓
real sensor/control experiment
```

The C/C++ port is gated on reproducible validation of the Python integer reference and should preserve the same explicit accounting and fixed-point contract. GitHub Actions is currently failing before job steps are exposed, so the Python branch remains Draft until CI or an equivalent full-checkout validation is available.
