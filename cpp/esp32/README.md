# ESP32 integer lag-8 harness

This directory contains the first target-hardware harness for the dependency-free C++17 `IntegerLag8Kernel`.

## Purpose

The harness does **not** yet control ADC/PWM hardware. It performs two engineering checks on an ESP32 target:

1. deterministic parity against the same 24-observation reference vector used by the host C++ test;
2. target-side timing and object-size reporting.

The serial output reports:

- parity PASS/FAIL;
- `sizeof(IntegerLag8Model)`;
- `sizeof(IntegerLag8Kernel)`;
- combined ABI object size;
- algorithmic core accounting;
- microseconds per 24-observation sequence;
- microseconds per observation.

These measurements exclude firmware/framework overhead, stack peaks, input/output buffers beyond the local fixture, ADC/PWM drivers, and energy consumption.

## Build with PlatformIO

From this directory:

```bash
pio run
```

Upload and monitor:

```bash
pio run --target upload
pio device monitor
```

The default environment targets the generic `esp32dev` board using Arduino framework and C++17.

## Scientific status

Pre-alpha / experimental. Host timing must not be reported as ESP32 timing. ESP32 latency and memory claims should only be made after this harness has been compiled and run on physical target hardware, with the exact board/toolchain configuration recorded.
