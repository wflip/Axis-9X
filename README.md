
# Axis9X — Adaptive VRAM Controller

**Axis9X** (`RS250Controller`) is a lightweight, real-time adaptive batch-size controller built by **RS-250 Systems** to eliminate GPU out-of-memory (OOM) crashes and maximize throughput under dynamic VRAM pressure.

Instead of locking workloads into static batch sizes, Axis9X continuously monitors live CUDA memory utilization and scales batch sizing up or down before an allocation failure can occur.

---

## Key Features

- **Live CUDA Pressure Telemetry:** Monitors active `allocated / total_memory` pressure at the front door of every execution step.
- **Four-Tier State Machine:** Classifies memory pressure into `SAFE`, `WATCH`, `THROTTLE`, and `HARD` states with automated batch-size reactions.
- **Hybrid Early-Exit + Fast Ramp:** Jumps immediately to `max_batch` when rolling memory history remains under the `LOW` threshold (`< 12%`), and ramps rapidly (`+4` per step) in `SAFE` mode to recover throughput without slow warm-up lag.
- **OOM Interception & Reserved-Pool Reclaim:** Catches `RuntimeError` OOM events in-flight, halves the active batch size, and invokes `torch.cuda.empty_cache()` to reclaim fragmented blocks without crashing the run.
- **TRUE_COMMIT Verification:** Tracks completed sample delivery, peak VRAM footprint, latency, and OOM counts to verify full workload completion.

---

## Control Policy & Thresholds

| State | Memory Pressure (`p`) | Batch Adjustment | Cache Action |
| :--- | :---: | :--- | :--- |
| **SAFE (Early-Exit)** | `max(history) < 0.12` | Jump to `max_batch` (`36`) | None |
| **SAFE (Fast Ramp)** | `p < 0.28` | `batch + 4` (up to `max_batch`) | None |
| **WATCH** | `0.28 <= p < 0.42` | `max(min_batch, batch - 2)` | None |
| **THROTTLE** | `0.42 <= p < 0.70` | `max(min_batch, batch // 2)` | None |
| **HARD / OOM** | `p >= 0.70` or OOM caught | `max(min_batch, batch // 2)` | `torch.cuda.empty_cache()` |

---

## Benchmark Results (NVIDIA Tesla T4 — 14.5 GB)

Evaluated across light-load, high-pressure shrink, and 75% noise stress scenarios:

| Test Case | Peak VRAM | Throughput (`samp/s`) | Status |
| :--- | :---: | :---: | :--- |
| **Best-Case Batch-Scaled Reduction** | `8.0 GB → 3.0 GB` (`-62.5%`) | — | Verified |
| **Normal (Light Load, Slow Ramp)** | `2,000 MB` | `557` | `TRUE_COMMIT` |
| **Hybrid (Early-Exit + Fast Ramp)** | `3,600 MB` | `2,249` (`4x` speedup) | `TRUE_COMMIT` |
| **Latency / VRAM / Commit** | `3,840 MB` | `1,196` | `TRUE_COMMIT` |
| **75% Noise Stress Test** | `2,560 MB` | `44,075` | `TRUE_COMMIT` |
| **High-Pressure Shrink Test** | `9,040 MB` | — | `0 OOM` |

---

## Quick Start

### Requirements
- Python 3.9+
- PyTorch with CUDA enabled (`torch.cuda.is_available() == True`)

### Run the Demo

```bash
python rs250_controller.py
