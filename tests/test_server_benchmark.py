from benchmarks.server.latency import run


def test_server_latency_benchmark_smoke() -> None:
    result = run(iterations=5, length=4)
    assert result["engine"] == "integer_lag8"
    assert result["iterations"] == 5
    assert result["sequence_length"] == 4
    for key in (
        "direct_model",
        "json_serialization",
        "rest_roundtrip",
        "websocket_roundtrip_2_observations",
    ):
        summary = result[key]
        assert summary["median_us"] >= 0.0
        assert summary["p95_us"] >= 0.0
        assert summary["mean_us"] >= 0.0
