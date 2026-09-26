"""A four-candidate GPU smoke test for the recording and analysis pipeline."""

from __future__ import annotations

import random
from statistics import median

import numpy as np
from numba import cuda


@cuda.jit
def plain(x, y):
    i = cuda.grid(1)
    if i < x.size:
        y[i] = 2.0 * x[i] + 1.0


@cuda.jit
def unroll_two(x, y):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    if i < x.size:
        y[i] = 2.0 * x[i] + 1.0
    j = i + stride
    if j < x.size:
        y[j] = 2.0 * x[j] + 1.0


def evaluate(manifest: dict) -> list[dict]:
    if not cuda.is_available():
        raise RuntimeError("Numba cannot access a CUDA GPU")
    workload = manifest["workload"]
    size = int(workload["size"])
    seed = int(workload["seed"])
    rng = np.random.default_rng(seed)
    x = rng.random(size, dtype=np.float32)
    expected = 2.0 * x + 1.0
    device_x = cuda.to_device(x)
    device_y = cuda.device_array_like(device_x)
    candidates = manifest["candidates"]
    timings = {item["id"]: [] for item in candidates}
    correct = {}

    def launch(item: dict) -> None:
        threads = int(item["block_size"])
        elements_per_thread = 2 if item["unroll_two"] else 1
        blocks = (size + threads * elements_per_thread - 1) // (threads * elements_per_thread)
        kernel = unroll_two if item["unroll_two"] else plain
        kernel[blocks, threads](device_x, device_y)

    for item in candidates:
        for _ in range(int(workload["warmups"])):
            launch(item)
        cuda.synchronize()
        correct[item["id"]] = bool(np.allclose(device_y.copy_to_host(), expected, rtol=1e-6, atol=1e-6))

    order = list(candidates)
    shuffler = random.Random(seed)
    for _ in range(int(workload["trials"])):
        shuffler.shuffle(order)
        for item in order:
            start = cuda.event(timing=True)
            end = cuda.event(timing=True)
            start.record()
            launch(item)
            end.record()
            end.synchronize()
            timings[item["id"]].append(float(cuda.event_elapsed_time(start, end)))

    return [
        {
            "id": item["id"],
            "correct": correct[item["id"]],
            "latencies_ms": timings[item["id"]],
            "median_ms": median(timings[item["id"]]),
        }
        for item in candidates
    ]
