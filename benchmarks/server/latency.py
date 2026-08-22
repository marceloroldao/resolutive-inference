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

from resolutive_inference.api import _decode_observations
from resolutive_inference.compact_robust import CompactRobust119
from resolutive_inference.server_app import create_server_app


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


def _observations(length: int) -> list[list[float]]:
    return np.tile(np.linspace(0.0, 0.6, 7), (length, 1)).tolist()


def _summary(samples_ns: list[int]) -> dict[str, float]:
    ordered = sorted(samples_ns)
    p95_index = max(0, round(0.95 * (len(ordered) - 1)))
    return {
        "median_us": statistics.median(ordered) / 1_000.0,
        "p95_us": ordered[p95_index] / 1_000.0,
        "mean_us": statistics.fmean(ordered) / 1_000.0,
    }


def run(iterations: int, length: int) -> dict[str, object]:
    model = _model()
    observations = _observations(length)
    app = create_server_app()
    client = TestClient(app)
    registered = client.put("/v1/models/bench", json=_model_payload())
    registered.raise_for_status()

    direct: list[int] = []
    rest: list[int] = []
    serialization: list[int] = []
    websocket: list[int] = []

    request_payload = {"observations": observations, "engine": "integer_lag8"}

    for _ in range(iterations):
        start = time.perf_counter_ns()
        _decode_observations(model, observations, "integer_lag8")
        direct.append(time.perf_counter_ns() - start)

        start = time.perf_counter_ns()
        json.dumps(request_payload, separators=(",", ":"))
        serialization.append(time.perf_counter_ns() - start)

        start = time.perf_counter_ns()
        response = client.post("/v1/infer/bench", json=request_payload)
        response.raise_for_status()
        rest.append(time.perf_counter_ns() - start)

    opened = client.post(
        "/v1/sessions",
        json={"model_id": "bench", "engine": "integer_lag8"},
    )
    opened.raise_for_status()
    session_id = opened.json()["session_id"]
    block = _observations(2)
    with client.websocket_connect(f"/v1/ws/sessions/{session_id}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "session"
        for _ in range(iterations):
            start = time.perf_counter_ns()
            ws.send_json({"type": "observations", "observations": block})
            result = ws.receive_json()
            assert result["type"] == "inference"
            websocket.append(time.perf_counter_ns() - start)

    return {
        "iterations": iterations,
        "sequence_length": length,
        "engine": "integer_lag8",
        "units": "microseconds",
        "direct_model": _summary(direct),
        "json_serialization": _summary(serialization),
        "rest_roundtrip": _summary(rest),
        "websocket_roundtrip_2_observations": _summary(websocket),
        "interpretation": "local in-process TestClient benchmark; not a network or production result",
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
