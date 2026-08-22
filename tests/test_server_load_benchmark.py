from benchmarks.server.load import run


def test_server_load_benchmark_smoke() -> None:
    result = run(requests=4, length=4, concurrency_levels=[1, 2])
    assert result["engine"] == "integer_lag8"
    assert result["sequence_length"] == 4
    assert result["requests_per_level"] == 4
    assert result["concurrency_levels"] == [1, 2]
    levels = result["results"]
    assert len(levels) == 2
    for level in levels:
        assert level["requests"] == 4
        assert level["errors"] == 0
        assert level["error_rate"] == 0.0
        assert level["throughput_rps"] > 0.0
        assert level["p50_us"] >= 0.0
        assert level["p95_us"] >= level["p50_us"]
        assert level["p99_us"] >= level["p95_us"]
