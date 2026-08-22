# Benchmark Protocol

Benchmarks in this repository are intended to be reproducible controlled experiments, not marketing claims.

## General requirements

Every benchmark should record:

- explicit random seeds;
- dataset/generator configuration;
- train/test split rule;
- model configuration;
- metric definitions;
- persistent-model accounting;
- runtime-buffer accounting when applicable;
- clear separation between measured values and analytical proxies.

Synthetic results must be labeled as synthetic. External datasets must be identified separately. Hardware claims require measurements on the named hardware.

## Compact-Robust Edge chain

The Edge approximation chain should be evaluated incrementally:

```text
float Compact-Robust 119
→ Q4
→ Q4 + LUT-128
→ bounded lag 16 / 8 / 4
→ integer-only lag 8
```

Each transition should report its delta against the immediately preceding reference, not only absolute accuracy.

## Integer runtime benchmark

Run:

```bash
python -m benchmarks.edge.integer_lag8
```

The benchmark compares the hybrid Q4/LUT-128 reference with `IntegerLag8Decoder` and reports:

- hybrid accuracy;
- integer lag-8 accuracy;
- exact path agreement between hybrid and integer outputs;
- compiled persistent bytes;
- algorithmic runtime-buffer bytes;
- combined core bytes.

The float-to-int observation conversion is preprocessing and is excluded from the claim that the decoder runtime is integer-only. The integer runtime claim applies only after observations have been quantized to the expected `int16` scale.

## Bounded-memory stress benchmark

Run:

```bash
python -m benchmarks.edge.bounded_stress
```

Stress scenarios should include ambiguous emissions, switching dynamics and burst contamination. Lags 16/8/4 are the primary engineering candidates, while lag 2/1 may be used to expose the failure boundary.

## Memory language

Do not equate payload size with total model or firmware size. Report separately:

1. quantized payload;
2. quantization metadata;
3. LUT/tables;
4. runtime buffers;
5. executable/firmware and platform overhead when measured.

Analytical byte counts in Python are reference accounting only until confirmed in the C/C++ implementation and on target hardware.
