import torch
import gc
import time
from collections import deque


class RS250Controller:
    """
    RS-250 Adaptive VRAM Controller
    Hybrid early-exit + fast ramp + shrink / hard-hold
    """

    def __init__(
        self,
        max_batch=36,
        min_batch=1,
        low_threshold=0.12,
        watch_threshold=0.28,
        throttle_threshold=0.42,
        hard_threshold=0.70,
        growth_step=4,
        history_len=5
    ):
        self.max_batch = max_batch
        self.min_batch = min_batch
        self.low = low_threshold
        self.watch = watch_threshold
        self.throttle = throttle_threshold
        self.hard = hard_threshold
        self.growth_step = growth_step
        self.batch = 8
        self.history = deque(maxlen=history_len)
        self.state = "SAFE"
        self.oom_count = 0

    def pressure(self):
        if not torch.cuda.is_available():
            return 0.0
        alloc = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return alloc / total

    def classify(self, p):
        if p >= self.hard:
            return "HARD"
        if p >= self.throttle:
            return "THROTTLE"
        if p >= self.watch:
            return "WATCH"
        return "SAFE"

    def update(self):
        p = self.pressure()
        self.history.append(p)
        recent_max = max(self.history) if self.history else 0.0

        # Early-exit under light load
        if recent_max < self.low:
            self.batch = self.max_batch
            self.state = "SAFE"
            return p, self.batch, self.state

        self.state = self.classify(p)

        if self.state == "HARD":
            self.batch = max(self.min_batch, self.batch // 2)
            torch.cuda.empty_cache()
        elif self.state == "THROTTLE":
            self.batch = max(self.min_batch, self.batch // 2)
        elif self.state == "WATCH":
            self.batch = max(self.min_batch, self.batch - 2)
        else:
            # SAFE fast ramp
            self.batch = min(self.max_batch, self.batch + self.growth_step)

        return p, self.batch, self.state

    def step(self, mb_per_sample=100):
        try:
            n = max(self.batch, 1) * mb_per_sample * 1024 * 1024 // 4
            buf = torch.empty(n, device="cuda")
            buf[0] = 1.0
            p = self.pressure()
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            del buf
            return {
                "pressure": p,
                "batch": self.batch,
                "state": self.state,
                "alloc_mb": alloc,
                "reserved_mb": reserved,
                "oom": 0
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.oom_count += 1
                torch.cuda.empty_cache()
                self.batch = max(self.min_batch, self.batch // 2)
                return {
                    "pressure": self.pressure(),
                    "batch": self.batch,
                    "state": "HARD",
                    "alloc_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "oom": 1
                }
            raise


def run_demo(total_samples=160, mb_per_sample=100):
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("GPU:", torch.cuda.get_device_name(0))
    ctrl = RS250Controller()
    samples = steps = 0
    peak_alloc = 0.0
    t0 = time.perf_counter()

    while samples < total_samples and steps < 40:
        ctrl.update()
        result = ctrl.step(mb_per_sample)
        peak_alloc = max(peak_alloc, result["alloc_mb"])
        if result["oom"] == 0:
            samples += result["batch"]
        steps += 1

    elapsed = time.perf_counter() - t0
    verified = samples >= total_samples

    print()
    print("LATENCY")
    print("  elapsed_s    : %.4f" % elapsed)
    print("  samples/sec  : %.1f" % (samples / elapsed if elapsed > 0 else 0))
    print()
    print("VRAM")
    print("  peak alloc MB: %.1f" % peak_alloc)
    print()
    print("TRUE COMMIT")
    print("  samples      : %d" % samples)
    print("  verified     : %s" % verified)
    print("  status       : %s" % ("TRUE_COMMIT" if verified else "FAIL"))
    print("  OOM events   : %d" % ctrl.oom_count)


if __name__ == "__main__":
    run_demo()