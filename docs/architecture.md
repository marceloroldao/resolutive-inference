# Architecture

## Core design

The core is decomposed into explicit state transitions, emission likelihoods, quantization, and metrics. `CompactPro` composes those primitives into an online Bayesian filtering step. This separation is intended to make assumptions auditable and allow each component to be replaced in controlled ablations.

Public APIs should remain small, typed, deterministic under fixed inputs, and independent of experiment orchestration. Benchmarks and experiments may depend on the core package; the core package must not depend on them.

## Role in the Resolutive computational stack

The proposed architectural role is:

```text
physical world / data streams
            |
            v
     sensors / telemetry
            |
            v
       bit.analyze
  signal structure/features
            |
            v
  resolutive-inference
 state / regime / anomaly
            |
            v
       memoria.ia
 memory / prior trajectories
            |
            v
    decision / control
```

This is an architectural direction, not a claim that all integrations shown above are already implemented.

Responsibility boundaries:

- **bit.analyze**: candidate upstream layer for extracting or organizing structure from raw or partially processed signals.
- **resolutive-inference**: estimates the current latent state, regime, transition, or anomaly from sequential observations.
- **memoria.ia**: candidate downstream memory layer for associating inferred states and trajectories with prior experience.
- **decision/control**: application-specific logic consuming the inferred state and memory result.

Resolutive Inference remains independently testable and must not require sibling projects for its benchmark claims to be valid.

## Conceptual distinction from neural sequence models

A neural predictor is often represented as `y = f_theta(x)`, where learned parameters encode a fitted mapping. Resolutive Inference investigates an explicit sequential state representation, conceptually `X_t -> R_t -> S_t`, with inference conditioned by transition history such as `S_(t-2) -> S_(t-1) -> S_t`.

This does not imply that neural networks lack state, temporal structure, or interpretable variants. Neural models remain established and often substantially more expressive for complex perception, language, high-dimensional representation learning, and tasks supported by large training corpora. This project does not claim to replace them generally.

The narrower research question is:

> For which classes of sequential inference can a deliberately compact, explicit state-space engine achieve useful accuracy while materially reducing memory, compute, latency, or energy cost?

## Intended engineering properties

The project investigates:

1. small state footprint;
2. inspectable states, transitions, scores, and trajectory decisions;
3. reproducible and deterministic execution where the algorithm permits it;
4. streaming operation without unbounded history retention;
5. bounded backtrace with explicit memory cost;
6. low-precision and integer-only execution; and
7. efficient operation on CPU and MCU-class hardware.

These are research objectives and must not be presented as demonstrated advantages until supported by reproducible measurements.

## Evaluation principle

Accuracy alone is insufficient for the intended contribution. Matched comparisons should report, as applicable:

- predictive/state accuracy or task-specific quality;
- negative log likelihood and calibration;
- anomaly/change-detection quality;
- fitted parameter or maintained-statistic count;
- peak and persistent memory footprint;
- latency and throughput;
- compute/operation budget;
- measured energy consumption when reliable hardware instrumentation is available; and
- executable/code footprint for embedded targets.

Baseline families should include conventional HMMs, robust/heavy-tailed HMM variants such as Student-t emissions, classical state estimators such as Kalman-family methods when their assumptions fit the task, and small neural sequence baselines when justified by the dataset and task.

A benchmark establishes evidence only for the evaluated dataset, configuration, hardware, metric, and resource budget. It must not be generalized into a universal architectural ranking.

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

The full reference uses Student-t-like emission costs and second-order dynamic programming. Fitting is supervised at this stage so that the estimator is not mixed with the decoder while the approximation chain is being audited.

## Q4 representation

Each of the five parameter tensors is uniformly quantized to four bits. Payload accounting and reconstruction metadata are deliberately separate. The 119 codes occupy 476 theoretical bits (59.5 bytes), or 60 byte-aligned bytes under ideal nibble packing. Five affine tensors require lower/scale metadata separately.

## LUT-128

A 128-entry `uint16` lookup table approximates the Student-t-like robust cost. The hybrid Python Q4/LUT decoder still calculates normalized observation distance in floating point before LUT indexing, so it is not an integer-only kernel.

## Bounded second-order decoding

`BoundedSecondOrderDecoder` constrains predecessor history to a fixed lag. Its algorithmic score/backpointer budget is:

```text
2 x (4 x 4) int32 score planes
+ lag x (4 x 4) uint8 predecessor ring
```

Lags 16, 8 and 4 are primary candidates; lag 2/1 are retained for stress studies. Easy synthetic data can hide the effect of bounded history, therefore the repository includes ambiguous/noisy switching scenarios.

## Integer-only runtime contract

`IntegerEmissionRuntime` and `IntegerLag8Decoder` introduce a strict compile/runtime boundary.

Compilation may use the floating/Q4 research representation to construct integer means, reciprocal variance factors, LUT entries, and integer transition costs. Runtime consumes pre-quantized `int16` observations. Emission squared differences, reciprocal-variance weighting, LUT indexing, transition accumulation, minimum selection, score renormalization, predecessor storage and lag-8 traceback are integer operations.

Reference core accounting for lag 8 is currently:

```text
compiled persistent integer tables : 510 B
algorithmic runtime buffers         : 272 B
------------------------------------------
core reference data                 : 782 B
```

This excludes executable code, stack, allocator/container overhead, Python object overhead, I/O buffers, preprocessing buffers and platform-specific state. It therefore must not be described as total firmware RAM.

## C++17 kernel

The C++17 implementation mirrors the integer lag-8 contract with fixed-size arrays and no heap allocation inside `decode()`. The deterministic host fixture checks parity against the Python integer reference.

On the development host ABI, `sizeof(IntegerLag8Model)` was 512 bytes and `sizeof(IntegerLag8Kernel)` was 276 bytes, for 788 bytes combined. These values are compiler/ABI dependent and are not MCU measurements.

## ESP32 target status

`cpp/esp32/` preserves a PlatformIO + Arduino harness for a generic `esp32dev` target. The MCU track is currently **frozen** while PC/server/API development is prioritized. No ESP32 performance claim is made without target compilation and hardware measurements.

If resumed, the target gate requires successful cross-compilation, parity PASS, compiler-reported flash/RAM usage, target `sizeof`, repeated target latency measurements, and exact board/toolchain configuration. Energy per observation is a separate measurement.

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
PC/server API track                 ← current priority

ESP32 / STM32 target harness        ← frozen future track
```

## Scientific-status rule

Architecture, hypothesis, implementation, benchmark result, and validated conclusion are distinct statuses.

In particular:

- `bit.analyze -> resolutive-inference -> memoria.ia` is an **architectural direction** until integrated;
- compactness measured from an implementation is an **engineering result** for that configuration;
- superiority over another method requires a **matched reproducible benchmark**; and
- general superiority over neural networks is **not claimed**.
