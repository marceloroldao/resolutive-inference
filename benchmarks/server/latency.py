"""Reproducible server-path latency benchmark.

Reports local process timings only. Results depend on hardware, Python version,
OS, load and dependency versions and must not be generalized across machines.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
from fastapi.testclient import TestClient

from resolutive_inference.compact_robust import CompactRobust119
from resolutive_inference.edge_compact import Q4CompactRobust119, StudentTCostLUT
from resolutive_inference.integer_runtime import IntegerLag8Decoder
from resolutive_inference.server_app import create_server_app
from resolutive_inference.sessions import MAX_SESSION_OBSERVATIONS


def _model_payload() -> dict[str, object]:
    transition = np.eye(4) + 0.2
    transition2 = np.ones((4, 4, 4), dtype=float)
    return {
        "means": (np.arange(28, dtype=float).reshape(4, 7) / 10.0).tolist(),
        "shared_variances": np.linspace(0.7, 1.3, 7).tolist(),
        "transition": transition.tolist(),
        "transition2": transition2.tolist(),
        "initial": [1.0, 1.0, 1.0, 1.0],
        "degrees_of_freedom": 3.0,
    }


def _model() -> CompactRobust119:
    payload = _model_payload()
    return CompactRobust119(
        means=np.asarray(payload["means"], dtype=float),
        shared_variances=np.asarray(payload["shared_variances"], dtype=float),
        transition=np.asarray(payload["transition"], dtype=float),
        transition2=np.asarray(payload["transition2"], dtype=float),
        initial=np.asarray(payload["initial"], dtype=float),
        degrees_of_freedom=float(payload["degrees_of_freedom"]),
    )


def _compile_decoder(model: CompactRobust119) -> IntegerLag8Decoder:
    q4 = Q4CompactRobust119.from_model(model)
    lut = StudentTCostLUT.build(128, degrees_of_freedom=model.degrees_of_freedom)
    return IntegerLag8Decoder.compile(q4, lut, lag=8)


def _observations(length: int) -> list[list[float]]:
    return np.tile(np.linspace(0.0, 0.6, 7), (length, 1)).tolist()


def _summary(samples_ns: list[int], observation_count: int | None = None) -> dict[str, float]:
    ordered = sorted(samples_ns)
    p95_index = max(0, round(0.95 * (len(ordered) - 1)))
    result = {
        "median_us": statistics.median(ordered) / 1_000.0,
        "p95_us": ordered[p95_index] / 1_000.0,
        "mean_us": statistics.fmean(ordered) / 1_000.0,
    }
    if observation_count:
        result["median_us_per_observation"] = result["median_us"] / observation_count
    return result


def _websocket_samples(
    client: TestClient,
    block: list[list[float]],
    iterations: int,
) -> list[int]:
    opened = client.post(
        "/v1/sessions",
        json={"model_id": "bench", "engine": "integer_lag8"},
    )
    opened.raise_for_status()
    session_id = opened.json()["session_id"]
    samples: list[int] = []
    with client.websocket_connect(f"/v1/ws/sessions/{session_id}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "session"
        for _ in range(iterations):
            start = time.perf_counter_ns()
            ws.send_json({"type": "observations", "observations": block})
            result = ws.receive_json()
            assert result["type"] == "inference"
            samples.append(time.perf_counter_ns() - start)
    return samples


def run(iterations: int, length: int) -> dict[str, object]:
    model = _model()
    observations = _observations(length)
    observations_array = np.asarray(observations, dtype=float)
    precompiled = _compile_decoder(model)
    app = create_server_app()
    client = TestClient(app)
    registered = client.put("/v1/models/bench", json=_model_payload())
    registered.raise_for_status()

    compile_samples: list[int] = []
    runtime_samples: list[int] = []
    rest: list[int] = []
    serialization: list[int] = []
    request_payload = {"observations": observations, "engine": "integer_lag8"}

    for _ in range(iterations):
        start = time.perf_counter_ns()
        _compile_decoder(model)
        compile_samples.append(time.perf_counter_ns() - start)

        start = time.perf_counter_ns()
        precompiled.decode_from_float(observations_array)
        runtime_samples.append(time.perf_counter_ns() - start)

        start = time.perf_counter_ns()
        json.dumps(request_payload, separators=(",", ":"))
        serialization.append(time.perf_counter_ns() - start)

        start = time.perf_counter_ns()
        response = client.post("/v1/infer/bench", json=request_payload)
        response.raise_for_status()
        rest.append(time.perf_counter_ns() - start)

    cache_info_response = client.get("/v1/info")
    cache_info_response.raise_for_status()
    engine_cache = cache_info_response.json().get("engine_cache")

    ws2_iterations = min(iterations, MAX_SESSION_OBSERVATIONS // 2)
    websocket_2 = _websocket_samples(client, _observations(2), ws2_iterations)

    ws_full_iterations = min(iterations, MAX_SESSION_OBSERVATIONS // length)
    websocket_full = _websocket_samples(client, observations, ws_full_iterations)

    return {
        "iterations": iterations,
        "sequence_length": length,
        "engine": "integer_lag8",
        "units": "microseconds",
        "model_compile": _summary(compile_samples),
        "precompiled_runtime": _summary(runtime_samples, length),
        "json_serialization": _summary(serialization),
        "rest_roundtrip_current_api": _summary(rest, length),
        "engine_cache_after_rest": engine_cache,
        "websocket_incremental_2_observations": {
            "iterations": ws2_iterations,
            **_summary(websocket_2, 2),
        },
        "websocket_comparable_block": {
            "iterations": ws_full_iterations,
            "observation_count_per_message": length,
            **_summary(websocket_full, length),
        },
        "methodology": {
            "rest_current_behavior": "compiled engines are reused through the bounded process-local engine cache",
            "precompiled_runtime": "decoder compiled once; timing includes float-to-int quantization and decode",
            "websocket": "session uses persistent incremental decoder state",
            "transport_scope": "FastAPI TestClient in-process; no real network/TLS/socket deployment latency",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--length", type=int, default=24)
    args = parser.parse_args()
    if args.iterations < 5 or args.length < 2:
        raise SystemExit("iterations must be >=5 and length must be >=2")
    print(json.dumps(run(args.iterations, args.length), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
