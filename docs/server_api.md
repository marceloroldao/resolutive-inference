# PC/Server API

This document describes the experimental `0.2.0-rc1` server contract. It is intended for direct use by applications, services, IoT systems, robots and metaverse integrations. The API is independent of any specific higher-level application.

## Install and run

```bash
python -m pip install -e ".[server]"
uvicorn resolutive_inference.server_app:app --host 0.0.0.0 --port 8000
```

OpenAPI documentation is available from FastAPI at `/docs` while the server is running.

## Configuration

Optional environment variables:

- `RESOLUTIVE_MODEL_STORE=/path/to/models` enables immutable JSON model versions on disk;
- `RESOLUTIVE_API_KEY=<secret>` enables API-key authentication for all `/v1/*` HTTP routes and WebSocket sessions.

When authentication is enabled, send:

```text
X-API-Key: <secret>
```

No key is stored in the repository.

## Engines

Current engine names:

- `float` — full floating-point CompactRobust119 decoder;
- `q4_lut128` — Q4/LUT-128 reference path;
- `integer_lag8` — compiled integer lag-8 path with cache support.

## Core endpoints

### Health and metadata

- `GET /health`
- `GET /v1/info`
- `GET /v1/server-info`

`/v1/info` reports API version, maturity, engines, persistence mode, cache statistics and request limits.

### Models

- `GET /v1/models`
- `PUT /v1/models/{model_id}`
- `GET /v1/models/{model_id}/versions`

With persistent storage enabled, each `PUT` creates a new immutable numeric version. Existing sessions remain pinned to the version selected when they were opened.

### Single inference

```text
POST /v1/infer/{model_id}?version=N
```

Request body:

```json
{
  "engine": "integer_lag8",
  "observations": [
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1]
  ]
}
```

`version` is optional; when omitted, the latest model version is used.

### Batch inference

```text
POST /v1/infer-batch/{model_id}?version=N
```

Current guardrails:

- maximum 64 sequences per batch;
- maximum 4096 observations per sequence;
- maximum 65536 observations across one batch.

These are API safety limits, not performance guarantees.

## Stateful sessions

Create a session:

```text
POST /v1/sessions
```

Example body:

```json
{
  "model_id": "sensor-a",
  "version": 2,
  "engine": "integer_lag8"
}
```

Session endpoints:

- `GET /v1/sessions/{session_id}`
- `POST /v1/sessions/{session_id}/observations`
- `DELETE /v1/sessions/{session_id}`

`integer_lag8` sessions use `incremental-lag8` compute mode. The other engines currently use `buffered-redecode` inside a stateful transport session.

Current process-level session guardrails:

- maximum 1024 simultaneous sessions;
- maximum 4096 observations per session.

Sessions are process-memory state in this release candidate and do not survive process restart.

## WebSocket

```text
WS /v1/ws/sessions/{session_id}
```

The session must already exist. The server first sends a `session` metadata message. The client can then send:

```json
{
  "type": "observations",
  "observations": [[0, 0, 0, 0, 0, 0, 0]]
}
```

The server replies with an `inference` message containing the current state sequence and latest state when enough observations are available.

To close cleanly:

```json
{"type": "close"}
```

When API-key protection is enabled, the WebSocket handshake also requires `X-API-Key`. Invalid authentication closes with code `4401`; unknown sessions use `4404`.

## Persistence and cache

Persistent model storage is intentionally simple in `0.2.0-rc1`: versioned JSON files are used as the baseline backend. The HTTP contract is designed so the backend can later be replaced by SQLite or another Resolutive storage engine without changing clients.

Compiled Q4/LUT-128 and integer-lag8 engines use a bounded, thread-safe LRU cache keyed by a deterministic model-content fingerprint. The default capacity is 256 entries.

## Performance scope

Repository benchmarks separate direct model runtime, JSON serialization, REST, WebSocket and single-process concurrent application load. Results exclude real network, TLS, reverse proxy, container orchestration and multi-process deployment overhead unless explicitly stated.

Do not interpret benchmark values as production SLAs.

## Stability

This is a release-candidate API for an experimental `v0.x` line. Endpoint and payload compatibility are not yet frozen as `v1.0`.
