# C++ integer Lag-8 reference

This directory contains the first dependency-free C++17 reference for the Compact-Robust integer runtime.

## Scope

The kernel is intentionally narrow:

- 4 latent states;
- 7 observation features;
- second-order transitions;
- 128-entry integer Student-t cost LUT;
- fixed lag = 8;
- quantized `int16_t` observations as runtime input;
- integer arithmetic in the inference path;
- no heap allocation inside `IntegerLag8Kernel::decode`.

The Python/Q4 model compilation step is outside this runtime contract.

## Footprint accounting

The explicit table accounting is 510 bytes and the algorithmic arrays are 272 bytes, for 782 bytes before ABI padding. On the host compiler used during development (`g++`, C++17), `sizeof(IntegerLag8Model)` was 512 bytes and `sizeof(IntegerLag8Kernel)` was 276 bytes, for 788 bytes combined. These `sizeof` values are compiler/ABI dependent.

The caller-owned observation sequence and output-state buffer are not included. Firmware, stack frames, code flash, peripheral drivers and allocator/runtime overhead are also excluded.

## Host validation

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

The deterministic parity fixture was generated from the repository Python integer reference. It checks that the C++ kernel recovers the same 24-state path for the compiled fixed-point tables and quantized observations.

## ESP32 target harness

`cpp/esp32/` now contains a PlatformIO + Arduino target harness for a generic `esp32dev` board. It uses the same deterministic 24-observation parity vector and then measures target-side latency with `micros()`.

The harness reports:

- parity PASS/FAIL;
- `sizeof(IntegerLag8Model)`;
- `sizeof(IntegerLag8Kernel)`;
- combined ABI object size;
- algorithmic core accounting;
- microseconds per sequence;
- microseconds per observation.

The current development environment does not include PlatformIO, Arduino CLI, or the Xtensa ESP32 compiler. Therefore no ESP32 cross-compile, flash/RAM, or hardware latency result is claimed yet.

## Status

Experimental. Passing the host parity fixture is not equivalent to ESP32/STM32 validation. The next gates are target compilation, compiler-size reporting, physical-board timing, energy measurement, and comparison against the Python reference over a larger exported fixture set.
