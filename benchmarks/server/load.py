"""Concurrent in-process ASGI load benchmark for the PC/server API.

This benchmark measures application-level concurrency using httpx ASGITransport.
It does not include network, TCP, TLS, reverse-proxy or multi-process overhead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from typing import Any

import httpx
import numpy as np

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


def _observations(length: int) -> list[list[float]]:
    return np.tile(np.linspace(0.0, 0.6, 7), (length, 1)).tolist()


def _percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index] / 1_000.0


def _summary(samples_ns: list[int], elapsed_s: float, errors: int) -> dict[str, float | int]:
    ordered = sorted(samples_ns)
    successes = len(samples_ns)
    return {
        "requests": successes + errors,
        "successes": successes,
        "errors": errors,
        "error_rate": errors / max(1, successes + errors),
        "throughput_rps": successes / max(elapsed_s, 1e-12),
        "mean_us": statistics.fmean(ordered) / 1_000.0 if ordered else 0.0,
        "p50_us": statistics.median(ordered) / 1_000.0 if ordered else 0.0,
        "p95_us": _percentile(ordered, 0.95),
        "p99_us": _percentile(ordered, 0.99),
        "elapsed_s": elapsed_s,
    }


async def _run_level(
    client: httpx.AsyncClient,
    *,
    concurrency: int,
    requests: int,
    payload: dict[str, object],
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    samples: list[int] = []
    errors = 0

    async def one() -> tuple[int | None, bool]:
        async with semaphore:
            start = time.perf_counter_ns()
            response = await client.post("/v1/infer/bench", json=payload)
            elapsed = time.perf_counter_ns() - start
            if response.status_code != 200:
                return None, False
            data = response.json()
            if data.get("observation_count") != len(payload["observations"]):
                return None, False
            return elapsed, True

    start_all = time.perf_counter()
    results = await asyncio.gather(*(one() for _ in range(requests)))
    elapsed_s = time.perf_counter() - start_all
    for elapsed, ok in results:
        if ok and elapsed is not None:
            samples.append(elapsed)
        else:
            errors += 1
    return {
        "concurrency": concurrency,
        **_summary(samples, elapsed_s, errors),
    }


async def run_async(
    *,
    requests: int,
    length: int,
    concurrency_levels: list[int],
) -> dict[str, object]:
    if requests < 1 or length < 2:
        raise ValueError("requests must be >=1 and length must be >=2")
    if not concurrency_levels or any(level < 1 for level in concurrency_levels):
        raise ValueError("concurrency levels must be positive")

    app = create_server_app()
    transport = httpx.ASGITransport(app=app)
    payload = {"observations": _observations(length), "engine": "integer_lag8"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        registered = await client.put("/v1/models/bench", json=_model_payload())
        registered.raise_for_status()

        # Warm the compiled-engine cache before measuring concurrency.
        warm = await client.post("/v1/infer/bench", json=payload)
        warm.raise_for_status()

        levels: list[dict[str, Any]] = []
        for concurrency in concurrency_levels:
            levels.append(
                await _run_level(
                    client,
                    concurrency=concurrency,
                    requests=requests,
                    payload=payload,
                )
            )

        info = (await client.get("/v1/info")).json()

    return {
        "engine": "integer_lag8",
        "sequence_length": length,
        "requests_per_level": requests,
        "concurrency_levels": concurrency_levels,
        "results": levels,
        "engine_cache": info.get("engine_cache", {}),
        "methodology": {
            "transport": "httpx ASGITransport in-process",
            "network": "not measured",
            "cache": "warmed once before measurements",
            "scope": "single Python process / single ASGI application instance",
        },
    }


def run(requests: int, length: int, concurrency_levels: list[int]) -> dict[str, object]:
    return asyncio.run(
        run_async(
            requests=requests,
            length=length,
            concurrency_levels=concurrency_levels,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--length", type=int, default=24)
    parser.add_argument("--concurrency", default="1,4,16,32")
    args = parser.parse_args()
    levels = [int(value.strip()) for value in args.concurrency.split(",") if value.strip()]
    print(json.dumps(run(args.requests, args.length, levels), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
