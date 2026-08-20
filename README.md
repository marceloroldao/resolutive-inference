# Resolutive Inference

**Experimental compact sequential inference for auditable Edge research.**

> Maturity: **pre-alpha / experimental**. Results are exploratory unless a repository-native benchmark and its configuration are explicitly cited.

## Scope

`resolutive-inference` investigates compact sequential models for filtering, latent-regime inference, anomaly-sensitive temporal processing, and resource-constrained Edge deployment. The project emphasizes deterministic experiments, explicit state transitions, small model footprints, quantization, and reproducible comparisons with conventional baselines.

This repository does **not** claim that Resolutive Inference generally replaces neural networks or that it is universally superior to HMM, Semi-Markov, or other statistical models.

## Current references

### Compact-Pro

`CompactPro` is the initial Gaussian state-space research baseline. It keeps the public API small and separates emissions, transitions, quantization, metrics, and experiment orchestration.

### Compact-Robust 119

The current Edge research branch defines a fixed four-state, seven-feature second-order reference with exactly **119 stored scalar statistics**:

- 28 means;
- 7 shared variances;
- 16 first-order transition probabilities;
- 64 second-order transition probabilities;
- 4 initial probabilities.

Its Q4 payload is **476 bits = 59.5 theoretical bytes**, requiring 60 whole bytes with ideal nibble packing. This number is the quantized payload only; scale/offset metadata, LUTs, runtime buffers, firmware, stack, allocator overhead, input buffers, and platform-specific structures are separate.

### Hybrid Q4 + LUT-128

The Q4 reference may use a 128-entry Student-t-like cost LUT. The current LUT uses 128 `uint16` entries (256 bytes). The Python hybrid reference still computes observation distances in floating point before LUT lookup.

### Bounded second-order decoding

Fixed-lag references evaluate lags 16, 8, and 4, with smaller lags used in stress tests. Algorithmic score/backpointer accounting is explicit and excludes Python/container overhead. Synthetic stress testing is used to expose the memory/accuracy tradeoff rather than treating an easy generator as evidence of equivalence.

### Integer-only runtime reference

`IntegerEmissionRuntime` and `IntegerLag8Decoder` separate **model compilation** from **runtime**:

- compilation from the Q4 research model may use floating point;
- observations are pre-quantized to `int16`;
- runtime emission distance, LUT indexing, transition scoring, score renormalization, predecessor storage, and second-order lag-8 decoding use integer arrays/arithmetic;
- float-to-int input conversion is a preprocessing convenience and is explicitly excluded from the integer-runtime claim.

The current accounting target for the lag-8 compiled reference is **510 bytes of persistent integer tables + 272 bytes of algorithmic runtime buffers = 782 bytes of core data**, excluding firmware, stack, allocator/Python overhead, I/O buffers and platform-specific data. This is a reference accounting result, not yet a measured ESP32/STM32 firmware footprint.

A repository-native benchmark is provided at:

```bash
python -m benchmarks.edge.integer_lag8
```

During independent reconstruction of the current algorithm, a 20-seed noisy synthetic study produced approximately **95.544%** mean accuracy for the hybrid Q4/LUT-128 lag-8 reference and **95.536%** for the integer lag-8 reference, with **99.728%** mean path agreement. These are development findings, not repository-native benchmark results, until reproduced from a complete checkout.

### C++17 integer kernel reference

`cpp/` now contains a dependency-free host reference matching the integer lag-8 contract. The decode path uses fixed-size arrays, integer arithmetic and no heap allocation inside the kernel.

The explicit table accounting remains 510 bytes and the algorithmic arrays remain 272 bytes. ABI padding is measured separately: on the development host compiler, `sizeof(IntegerLag8Model)` was 512 bytes and `sizeof(IntegerLag8Kernel)` was 276 bytes, for **788 bytes combined**, excluding caller-owned input/output buffers, stack frames, firmware and platform runtime.

The deterministic C++ parity fixture was compiled with C++17 and strict warnings and reproduced the expected 24-state path. Host validation can be run with:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

This is not yet ESP32/STM32 validation; cross-compilation and hardware timing remain separate gates.

## Scientific status

Synthetic benchmarks are controlled experiments. They are useful for regression, ablation, quantization error, bounded-memory behavior and stress testing, but they are not real-world validation. External datasets and hardware measurements must be reported separately.

## Edge roadmap

```text
Python reference
    ↓
Compact-Robust 119
    ↓
Q4
    ↓
LUT-128
    ↓
bounded backtrace 16 / 8 / 4
    ↓
integer-only runtime reference
    ↓
C++17 fixed-point kernel
    ↓
ESP32 / STM32 benchmark
    ↓
real sensor/control experiment
```

A future physical experiment will evaluate a bounded adaptive controller where an ADC measurement is used to tune a PWM/frequency command toward a maximum-response operating point. Hardware work is not part of the current pre-alpha validation.

## Reproducibility

Experiments should use explicit seeds and record configuration, metrics and assumptions. Current Edge entry points include:

```bash
python -m benchmarks.edge.streaming_backtrace
python -m benchmarks.edge.compact_robust_q4
python -m benchmarks.edge.bounded_stress
python -m benchmarks.edge.integer_lag8
```

Run tests and lint with:

```bash
python -m pytest
python -m ruff check .
```

## Resolutive compatibility

RSMS compatibility: **1.0-rc.1**.

New terminology or mathematical conventions should first be checked against the central `resolutive-science` specification to avoid incompatible project dialects.

## License

This repository uses the **Resolutive Research Non-Commercial License (RRNCL) 1.0**. Academic, educational, and research use is permitted under the license terms; commercial exploitation requires separate commercial authorization/licensing. Because of the commercial-use restriction, this should not be described simply as OSI Open Source.

See `LICENSE` for the complete terms.

## Citation

See `CITATION.cff` for author and citation metadata.
