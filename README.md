# Resolutive Inference

**Experimental compact sequential inference for auditable PC/server and Edge research.**

> Release candidate: **0.2.0rc1**. This is an experimental publication candidate, not a stable v1.0 API.

## Scope

`resolutive-inference` investigates compact sequential models for filtering, latent-regime inference, anomaly-sensitive temporal processing and resource-constrained deployment. The project emphasizes deterministic experiments, explicit state transitions, small model footprints, quantization and reproducible comparisons with conventional baselines.

The current engineering priority is **PC/server deployment with an independent API**. The API can be used directly by third-party applications, services, robots or IoT systems; it is not tied to a metaverse or any sibling Resolutive project. The ESP32/MCU line remains preserved as an experimental target path but is not the primary release focus.

This repository does **not** claim that Resolutive Inference generally replaces neural networks or that it is universally superior to HMM, Semi-Markov or other statistical models.

## Current reference model

### Compact-Robust 119

The current reference fixes four latent states and seven observed features with exactly **119 stored scalar statistics**:

- 28 means;
- 7 shared variances;
- 16 first-order transition probabilities;
- 64 second-order transition probabilities;
- 4 initial probabilities.

The Q4 payload is **476 bits = 59.5 theoretical bytes**, requiring 60 whole bytes under ideal nibble packing. This is payload accounting only; scale/offset metadata, LUTs, runtime buffers, firmware, stack, I/O and platform overhead are separate.

The project also includes Q4/LUT-128, bounded second-order decoding, integer lag-8 execution, an incremental lag-8 session runtime and a dependency-free C++17 reference kernel.

## PC/server API

Install the server extras:

```bash
python -m pip install -e ".[server]"
```

Run the API:

```bash
uvicorn resolutive_inference.server_app:app --host 0.0.0.0 --port 8000
```

The server surface includes:

- health and model information endpoints;
- model registration with immutable version history;
- single and batch inference;
- stateful sessions;
- incremental `integer_lag8` sessions;
- WebSocket transport for sessions;
- optional API-key protection through `RESOLUTIVE_API_KEY`;
- optional JSON model persistence through `RESOLUTIVE_MODEL_STORE`;
- bounded compiled-engine cache keyed by model content.

See `docs/server_api.md` for the current HTTP/WebSocket contract and deployment notes.

## Controlled benchmark results

All numbers below are repository-native GitHub Actions measurements and are limited to the stated configuration.

### Server latency

On GitHub-hosted Ubuntu / Python 3.12, the precompiled `integer_lag8` reference measured approximately **1.51 ms per 24-observation sequence** at the direct Python runtime layer. REST and WebSocket add application/serialization overhead; they are benchmarked separately under `benchmarks/server/`.

### Single-process load

Using `httpx.ASGITransport`, one ASGI process, a warmed engine cache, 200 requests per level and sequence length 24:

- concurrency 1: **371.9 req/s**, p50 2.63 ms, p95 2.86 ms, 0 errors;
- concurrency 4: **256.2 req/s**, p50 14.56 ms, p95 23.09 ms, 0 errors;
- concurrency 16: **257.1 req/s**, p50 57.48 ms, p95 83.05 ms, 0 errors;
- concurrency 32: **267.1 req/s**, p50 107.43 ms, p95 145.18 ms, 0 errors.

Across all four levels, **800/800 requests succeeded**. This is an in-process application benchmark, not a production network SLA.

### HMM controls

A 20-seed × 1000-observation known-parameter synthetic comparison produced:

| Scenario | CompactRobust119 | Gaussian HMM | Student-t HMM |
|---|---:|---:|---:|
| First-order Gaussian accuracy | **99.52%** | 99.21% | 98.79% |
| Second-order + 8% contamination accuracy | **98.56%** | 95.35% | 96.93% |
| Median runtime in contaminated scenario | **23.94 µs/obs** | 43.63 µs/obs | 52.93 µs/obs |

Persistent counts in that comparison are 119 stored statistics for CompactRobust119, 72 parameters for Gaussian HMM and 73 for Student-t HMM.

These results establish an observed advantage only for the tested synthetic known-parameter scenarios. They must not be generalized to all sequential-inference workloads.

## Reproducibility

Run Python validation with:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

Useful benchmark entry points include:

```bash
python -m benchmarks.server.latency
python -m benchmarks.server.load
python -m benchmarks.baselines.robust119
python -m benchmarks.edge.integer_lag8
```

Host C++ parity:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

## Architecture boundary

The intended composition is modular:

```text
Resolutive Inference core
        ↓
independent API
        ↓
apps / IoT / robots / services / metaverse integrations
```

Consumers should be able to use the API independently. Integrations must not require a private metaverse-only contract.

## Edge status

The C++17 fixed-array kernel and ESP32 target harness remain in the repository as an experimental path. No current release claim is made for ESP32 latency, flash, RAM or energy because those require target-hardware measurements.

## Scientific status

Synthetic benchmarks are controlled experiments useful for regression, ablation, approximation and stress testing. They are not real-world validation. External datasets and target hardware measurements should be reported separately when available.

## Resolutive compatibility

RSMS compatibility: **1.0-rc.1**.

## License

This repository uses the **Resolutive Research Non-Commercial License (RRNCL) 1.0**. Academic, educational and non-commercial research use is permitted under the license terms, including qualifying use by universities, schools, public research institutions, non-profit organizations and NGOs. Commercial exploitation or use supporting commercial advantage requires separate written authorization or a commercial license.

Because commercial use is restricted, this project is **source-available** and must not be represented as OSI-approved open-source software.

See `LICENSE` for the complete terms.

## Citation

See `CITATION.cff`. Cite the exact release or commit used.
