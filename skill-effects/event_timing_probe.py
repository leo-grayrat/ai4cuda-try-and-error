"""Check CUDA-event timing when original CUPTI profiler cannot see kernels."""
import json
from pathlib import Path
import random
import statistics
import sys

import torch

sys.path.insert(0, str(Path.cwd()))
from utils.profiling import initialize_models


def main():
    eager, candidate, inputs, candidate_inputs = initialize_models()
    models = {"eager": (eager, inputs), "compiled": (torch.compile(eager), inputs), "candidate": (candidate, candidate_inputs)}
    samples = {name: [] for name in models}
    rng = random.Random(20261019)
    iterations = 100
    with torch.no_grad():
        for model, args in models.values():
            for _ in range(10):
                model(*args)
        torch.cuda.synchronize()
        for _ in range(7):
            names = list(models)
            rng.shuffle(names)
            for name in names:
                model, args = models[name]
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    model(*args)
                stop.record()
                stop.synchronize()
                samples[name].append(start.elapsed_time(stop) * 1000 / iterations)
    if any(not all(value > 0 for value in values) for values in samples.values()):
        raise RuntimeError("invalid CUDA-event measurement")
    result = {"method": "CUDA events around repeated complete model calls", "iterations_per_sample": iterations, "samples_us": samples, "median_us": {name: statistics.median(values) for name, values in samples.items()}, "purpose": "environment feasibility on upstream development example only"}
    print(json.dumps(result, indent=2))
    Path("../cuda_event_timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
