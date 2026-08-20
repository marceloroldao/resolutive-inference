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

## C++17 kernel

The C++17 implementation mirrors the integer lag-8 contract with fixed-size arrays and no heap allocation inside `decode()`. The deterministic host fixture checks parity against the Python integer reference.

On the development host ABI, `sizeof(IntegerLag8Model)` was 512 bytes and `sizeof(IntegerLag8Kernel)` was 276 bytes, for 788 bytes combined. These values are compiler/ABI dependent and are not MCU measurements.

## ESP32 target gate

`cpp/esp32/` contains the first PlatformIO + Arduino harness for a generic `esp32dev` target. It runs deterministic parity before timing and reports target object sizes plus microseconds per sequence and observation.

The ESP32 gate is passed only after all of the following are recorded from an actual target toolchain/board:

1. successful cross-compilation;
2. parity PASS;
3. compiler-reported flash and RAM usage;
4. target `sizeof` values;
5. repeated target latency measurements;
6. exact board, framework, compiler, optimization and clock configuration.

Energy per observation is a separate later measurement and must not be inferred from host timing.

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
integer-only lag-8 Python reference
    ↓
C++17 fixed-point kernel
    ↓
ESP32 / STM32 target harness        ← current gate
    ↓
target RAM / Flash / latency / energy
    ↓
real sensor/control experiment
```

GitHub Actions is currently failing before job steps are exposed, and the present environment does not include PlatformIO/Arduino/Xtensa toolchains. Therefore the branch remains Draft until CI or equivalent full-checkout validation and target compilation are available.
