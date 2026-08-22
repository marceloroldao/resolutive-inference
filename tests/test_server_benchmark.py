from benchmarks.server.latency import run


def test_server_latency_benchmark_smoke() -> None:
    result = run(iterations=5, length=4)
    assert result["engine"] == "integer_lag8"
    assert result["iterations"] == 5
    assert result["sequence_length"] == 4
    for key in (
        "model_compile",
        "precompiled_runtime",
        "json_serialization",
        "rest_roundtrip_current_api",
        "websocket_incremental_2_observations",
        "websocket_comparable_block",
    ):
        summary = result[key]
        assert summary["median_us"] >= 0.0
        assert summary["p95_us"] >= 0.0
        assert summary["mean_us"] >= 0.0

    assert result["precompiled_runtime"]["median_us_per_observation"] >= 0.0
    assert result["websocket_comparable_block"]["observation_count_per_message"] == 4
