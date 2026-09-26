"""Independent CUDA correctness and timing probe for development trials."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


def _time_ms(fn, value: torch.Tensor, samples: int = 20) -> list[float]:
    for _ in range(5):
        fn(value)
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(value)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def evaluate(candidate_path: Path, reference, make_input, shapes: list[tuple[int, ...]]) -> dict:
    candidate_path = candidate_path.resolve()
    result = {
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
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
            actual = candidate(value)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            result["checks"].append({"shape": shape, "correct": True})
        result["correct"] = True
        torch.manual_seed(991)
        value = make_input(shapes[-1])
        result["reference_samples_ms"] = _time_ms(reference, value)
        result["candidate_samples_ms"] = _time_ms(candidate, value)
        result["reference_median_ms"] = statistics.median(result["reference_samples_ms"])
        result["candidate_median_ms"] = statistics.median(result["candidate_samples_ms"])
        result["speedup"] = result["reference_median_ms"] / result["candidate_median_ms"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
