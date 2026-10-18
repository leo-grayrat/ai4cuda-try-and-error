"""Reference-only GPU feasibility for frozen task shapes; no agent outputs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

from prepare_tasks import ROOT, catalog
from environment_probe import msvc_environment
from gpu_lock import gpu_lock


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("study_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(model, inputs, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = model(*inputs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def probe(task: dict, index: int, seed: int, validation_only: bool = False):
    module = load_reference(ROOT / "tasks" / task["id"] / "model.py")
    for key, value in task["cases"][index].items():
        setattr(module, key, tuple(value) if key == "input_shape" else value)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = module.Model(*module.get_init_inputs()).eval().cuda()
    # Non-default affine values detect kernels that silently assume unit weights.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.7, 1.3)
    cpu_inputs = module.get_inputs()
    inputs = [x.cuda() for x in cpu_inputs]
    del cpu_inputs
    compiled = torch.compile(model)
    with torch.no_grad():
        from verify_candidate import compare_output
        expected = model(*inputs)
        actual = compiled(*inputs)
        compare_output(actual, expected, task["id"])
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            stream_actual = compiled(*inputs)
        stream.synchronize()
        compare_output(stream_actual, expected, task["id"])
        if validation_only:
            return {"task": task["id"], "case_index": index, "model_sha256": hashlib.sha256((ROOT / "tasks" / task["id"] / "model.py").read_bytes()).hexdigest(), "globals": task["cases"][index], "correct_compiled_vs_eager": True, "nondefault_stream_correct": True, "validation_only": True, "softmax_distribution_checks": task["id"] == "softmax", "seconds": time.monotonic() - started}
        # Calibration selects equal iteration counts for both implementations.
        for _ in range(3):
            compiled(*inputs)
            model(*inputs)
        torch.cuda.synchronize()
        calibration_us = max(measure(model, inputs, 2), measure(compiled, inputs, 2))
        iterations = max(2, min(100, math.ceil(20000 / calibration_us)))
        samples = {"eager": [], "compiled": []}
        rng = random.Random(seed)
        for _ in range(7):
            order = ["eager", "compiled"]
            rng.shuffle(order)
            for name in order:
                samples[name].append(measure(model if name == "eager" else compiled, inputs, iterations))
    result = {"task": task["id"], "model_sha256": hashlib.sha256((ROOT / "tasks" / task["id"] / "model.py").read_bytes()).hexdigest(), "case_index": index, "split": task["split"], "globals": task["cases"][index], "seed": seed, "correct_compiled_vs_eager": True, "nondefault_stream_correct": True, "input_shapes": [list(x.shape) for x in inputs], "input_bytes": sum(x.numel() * x.element_size() for x in inputs), "parameters": sum(p.numel() for p in model.parameters()), "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "iterations_per_sample": iterations, "samples_us": samples, "median_us": {k: statistics.median(v) for k, v in samples.items()}, "seconds": time.monotonic() - started}
    if any(not all(math.isfinite(t) and t > 0 for t in values) for values in samples.values()):
        raise RuntimeError("invalid baseline timing")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        parser.error("name must be a directory name")
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=False)
    os.environ.update(msvc_environment())
    for key, value in [("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"), ("TEMP", "temp"), ("TMP", "temp")]:
        (run / value).mkdir(exist_ok=True)
        os.environ[key] = str(run / value)
    # Torch/Triton cache tempfile.gettempdir() during import. Configure the
    # writable locations before importing, as in the isolated agent subprocess.
    global torch
    import torch
    tasks = [t for t in catalog()["tasks"] if not args.tasks or t["id"] in args.tasks]
    rows = []
    for task in tasks:
        for index in range(len(task["cases"]) if args.all_cases else 1):
            print(f"baseline {task['id']} case {index}", flush=True)
            try:
                with gpu_lock(ROOT / "runs/gpu.lock") as waited:
                    result = probe(task, index, 20261019 + index, validation_only=args.validate_only)
                result["gpu_wait_seconds"] = waited
            except Exception as exc:
                result = {"task": task["id"], "case_index": index, "error": f"{type(exc).__name__}: {exc}", "correct_compiled_vs_eager": False}
            rows.append(result)
            (run / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in result.items() if k != "samples_us"}), flush=True)
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
    return 0 if all(row["correct_compiled_vs_eager"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
