# Release notes

## 0.2.0-rc1

Experimental PC/server release candidate for Resolutive Inference.

### Highlights

- CompactRobust119 four-state / seven-feature second-order reference with 119 stored statistics.
- Q4/LUT-128 and integer lag-8 execution paths.
- Stateful incremental `integer_lag8` sessions.
- FastAPI REST API with model registration, immutable version history, single inference and batch inference.
- WebSocket session transport.
- Optional API-key authentication using `RESOLUTIVE_API_KEY`.
- Optional versioned JSON model persistence using `RESOLUTIVE_MODEL_STORE`.
- Bounded thread-safe compiled-engine LRU cache keyed by deterministic model content.
- Reproducible server latency and single-process concurrency benchmarks in GitHub Actions.
- Conventional Gaussian HMM and Student-t HMM controls aligned to the current 4-state / 7-feature reference.

### Controlled comparative result

GitHub-hosted Ubuntu 24.04 / Python 3.12, 20 seeds × 1000 observations per scenario:

- first-order Gaussian accuracy: CompactRobust119 99.52%, Gaussian HMM 99.21%, Student-t HMM 98.79%;
- second-order + 8% heavy-tailed contamination: CompactRobust119 98.56%, Gaussian HMM 95.35%, Student-t HMM 96.93%;
- median runtime in the contaminated scenario: CompactRobust119 23.94 µs/observation, Gaussian HMM 43.63 µs/observation, Student-t HMM 52.93 µs/observation.

These are synthetic, known-parameter controls only. They do not establish universal superiority over HMMs or other inference methods.

### Server load result

In-process `httpx.ASGITransport`, one Python/ASGI process, warmed engine cache, 200 requests per concurrency level and sequence length 24:

- concurrency 1: 371.9 req/s, p50 2.63 ms, p95 2.86 ms;
- concurrency 4: 256.2 req/s, p50 14.56 ms, p95 23.09 ms;
- concurrency 16: 257.1 req/s, p50 57.48 ms, p95 83.05 ms;
- concurrency 32: 267.1 req/s, p50 107.43 ms, p95 145.18 ms.

All 800 requests succeeded. This is an application-level benchmark, not a production network SLA.

### Known limitations

- API compatibility is not frozen as v1.0.
- Sessions are process-memory state and do not survive server restart.
- `float` and `q4_lut128` sessions still use buffered re-decode rather than a fully incremental decoder.
- Current persistence uses versioned JSON files rather than a transactional database backend.
- High single-process concurrency increases tail latency and does not scale throughput linearly.
- Comparative accuracy evidence is synthetic; real external datasets remain future validation.
- ESP32/MCU target timing, RAM, flash and energy are not release claims.

### License

Resolutive Research Non-Commercial License (RRNCL) 1.0. Non-commercial research/education use is permitted under its terms; commercial use requires separate authorization or license.
