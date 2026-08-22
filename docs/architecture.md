# Architecture

## Core design

The core is decomposed into explicit state transitions, emission likelihoods, quantization, and metrics. `CompactPro` composes those primitives into an online Bayesian filtering step. This separation is intended to make assumptions auditable and allow each component to be replaced in controlled ablations.

The initial implementation uses finite latent states, a row-stochastic transition matrix, and diagonal Gaussian emissions. It is a research baseline rather than a finalized algorithm. Compact-Robust remains deliberately unimplemented until its estimator, contamination model, and validation criteria are specified.

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

Resolutive Inference remains independently testable and must not require these sibling projects for its benchmark claims to be valid.

## Conceptual distinction from neural sequence models

A conventional neural model can be summarized abstractly as:

```text
input
  |
  v
learned weights + numerical operations
  |
  v
layers / recurrent or attention computation
  |
  v
predicted output
```

Resolutive Inference investigates a different engineering abstraction:

```text
sequential observation
        |
        v
compact representation / relations
        |
        v
state-space transition structure
        |
        v
trajectory-conditioned inference
        |
        v
inferred state / regime
```

A neural predictor is often represented as `y = f_theta(x)`, where learned parameters `theta` encode a fitted mapping. Resolutive Inference instead emphasizes an explicit sequential state representation, conceptually `X_t -> R_t -> S_t`, with inference conditioned by transition history such as `S_(t-2) -> S_(t-1) -> S_t`.

This does not imply that neural networks lack state, temporal structure, or interpretable variants. It describes this project's design emphasis: explicit compact state and trajectory inference rather than large learned function approximators.

Neural models remain established and often substantially more expressive choices for complex perception, language, high-dimensional representation learning, and tasks supported by large training corpora. This project does not claim to replace them generally.

The narrower research question is:

> For which classes of sequential inference can a deliberately compact, explicit state-space engine achieve useful accuracy while materially reducing memory, compute, latency, or energy cost?

## Intended engineering properties

The project investigates:

1. small state footprint;
2. inspectable states, transitions, scores, and trajectory decisions;
3. reproducible and deterministic execution where the algorithm permits it;
4. streaming operation without unbounded history retention;
5. bounded backtrace with explicit memory cost;
6. low-precision and eventually integer-only execution; and
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

Derived measures such as accuracy per byte, accuracy per unit of compute, or accuracy per joule may be useful when their definitions and measurement protocols are explicit.

Baseline families should include conventional HMMs, robust/heavy-tailed HMM variants such as Student-t emissions, classical state estimators such as Kalman-family methods when their assumptions fit the task, and small neural sequence baselines when justified by the dataset and task.

A benchmark establishes evidence only for the evaluated dataset, configuration, hardware, metric, and resource budget. It must not be generalized into a universal architectural ranking.

## Hybrid quantized decoder

The historically named `FixedPointViterbi` is a hybrid reference rather than an integer-only implementation. Scores, transitions, and LUT entries are quantized/integer values, but the emission distance is calculated in floating point before LUT indexing. It must not be described as a fixed-point integer-only MCU kernel.

The 3-state/1D streaming benchmark exercises bounded-backtrace behavior and buffer accounting only. It does not reproduce a Compact-Robust configuration of approximately 119 maintained values.

## Implementation roadmap

The next reference is **Compact-Robust 119-value reference**, progressing through:

```text
floating-point reference
        |
        v
Q4 quantization
        |
        v
LUT-128 scoring
        |
        v
bounded backtrace 16 / 8 / 4
        |
        v
integer-only emissions
        |
        v
C/C++ MCU kernel
        |
        v
hardware latency / RAM / energy benchmarks
```

Only after those stages are reproducibly measured should embedded or energy advantages be promoted from hypotheses to demonstrated results.

## Scientific-status rule

Architecture, hypothesis, implementation, benchmark result, and validated conclusion are distinct statuses.

In particular:

- `bit.analyze -> resolutive-inference -> memoria.ia` is an **architectural direction** until integrated;
- compactness measured from an implementation is an **engineering result** for that configuration;
- superiority over another method requires a **matched reproducible benchmark**; and
- general superiority over neural networks is **not claimed**.
