"""Check CUDA-event timing when original CUPTI profiler cannot see kernels."""
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path.cwd()))
from utils.profiling import initialize_models


def main():
    eager, candidate, inputs, candidate_inputs = initialize_models()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    models = {"eager": (eager, inputs), "compiled": (torch.compile(eager), inputs), "candidate": (candidate, candidate_inputs)}
    samples = {name: [] for name in models}
    rng = random.Random(20261019)
    def timed(model, args, repeats):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            model(*args)
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) * 1000 / repeats

    with torch.no_grad():
        for model, args in models.values():
            for _ in range(10):
                model(*args)
        torch.cuda.synchronize()
        # Stabilize first-use clocks and choose one common repeat count that
        # does not turn large GEMM benchmarks into thousands of costly calls.
        until = time.monotonic() + 0.3
        while time.monotonic() < until:
            eager(*inputs)
            torch.cuda.synchronize()
        calibration_us = max(timed(model, args, 2) for model, args in models.values())
        iterations = max(2, min(100, math.ceil(20000 / calibration_us)))
        for _ in range(7):
            names = list(models)
            rng.shuffle(names)
            for name in names:
                model, args = models[name]
                samples[name].append(timed(model, args, iterations))
    if any(not all(value > 0 for value in values) for values in samples.values()):
        raise RuntimeError("invalid CUDA-event measurement")
    result = {"method": "CUDA events around repeated complete model calls, including launch gaps; not summed kernel-only time", "iterations_per_sample": iterations, "calibration_us": calibration_us, "matmul_precision": "highest, baseline TF32 disabled", "samples_us": samples, "median_us": {name: statistics.median(values) for name, values in samples.items()}, "purpose": "paired candidate/reference timing for the current frozen task"}
    print(json.dumps(result, indent=2))
    Path("../cuda_event_timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
