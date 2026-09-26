"""Independent CUDA correctness and timing probe for development trials."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import statistics
from pathlib import Path

import torch


def _load_run(path: Path):
    spec = importlib.util.spec_from_file_location("trial_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def _paired_graph_benchmark(reference, candidate, value: torch.Tensor, repeats: int = 32, pairs: int = 20) -> dict:
    """Time GPU work inside graphs, alternating the two implementations."""
    def capture(fn):
        for _ in range(5):
            fn(value)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                fn(value)
        return graph

    graphs = {"reference": capture(reference), "candidate": capture(candidate)}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = {"reference": [], "candidate": []}
    order = ["reference_first" if index % 2 == 0 else "candidate_first" for index in range(pairs)]
    random.Random(1729).shuffle(order)
    for pair in order:
        names = ("reference", "candidate") if pair == "reference_first" else ("candidate", "reference")
        for name in names:
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / repeats)
    return {
        "benchmark_mode": "paired CUDA graph replays",
        "repeats_per_graph": repeats,
        "pair_order": order,
        "reference_samples_ms": samples["reference"],
        "candidate_samples_ms": samples["candidate"],
    }


def _run_with_triton_launch_count(fn, value: torch.Tensor):
    """Count actual Triton JIT launches, rather than accepting an unused kernel definition."""
    from triton.runtime.jit import JITFunction

    original = JITFunction.__getitem__
    launches = []

    def tracked_getitem(kernel, grid):
        launcher = original(kernel, grid)

        def tracked_launch(*args, **kwargs):
            result = launcher(*args, **kwargs)
            launches.append(kernel.__name__)
            return result

        return tracked_launch

    JITFunction.__getitem__ = tracked_getitem
    try:
        actual = fn(value)
    finally:
        JITFunction.__getitem__ = original
    return actual, launches


def evaluate(candidate_path: Path, reference, make_input, shapes: list[tuple[int, ...]]) -> dict:
    candidate_path = candidate_path.resolve()
    result = {
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "evaluator_common_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "correct": False,
        "error": None,
        "checks": [],
        "reference_samples_ms": [],
        "candidate_samples_ms": [],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    if not torch.cuda.is_available():
        result["error"] = "CUDA unavailable"
        return result
    try:
        candidate = _load_run(candidate_path)
        for index, shape in enumerate(shapes):
            torch.manual_seed(index + 271)
            value = make_input(shape)
            expected = reference(value)
            actual, launches = _run_with_triton_launch_count(candidate, value)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            if not launches:
                raise ValueError(f"No Triton kernel launched for shape {shape}")
            result["checks"].append({"shape": shape, "correct": True, "triton_launches": launches})
        result["correct"] = True
        torch.manual_seed(991)
        value = make_input(shapes[-1])
        result.update(_paired_graph_benchmark(reference, candidate, value))
        result["reference_median_ms"] = statistics.median(result["reference_samples_ms"])
        result["candidate_median_ms"] = statistics.median(result["candidate_samples_ms"])
        result["speedup"] = result["reference_median_ms"] / result["candidate_median_ms"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
