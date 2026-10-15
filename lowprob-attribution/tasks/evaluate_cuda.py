"""Standalone evaluator for CUDA C++ kernel tasks.

Compiles and tests a candidate Model against the reference,
reports correctness + timing without KernelBench's full import chain.

Usage:
    python evaluate.py --reference reference.py --candidate candidate.py --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import tempfile
from pathlib import Path


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# correctness and timing each call the model on a fresh clone of the inputs, so
# the device holds the input set plus one clone at peak (plus small outputs).
PEAK_INPUT_COPIES = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--correct-trials", type=int, default=3)
    ap.add_argument("--perf-trials", type=int, default=20)
    ap.add_argument("--gpu-fraction", type=float, default=0.6,
                    help="max fraction of total VRAM the run may plan to occupy")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    ref_mod = load_module(args.reference, "ref_mod")
    cand_mod = load_module(args.candidate, "cand_mod")

    # --- GPU memory guard ---------------------------------------------------
    # get_inputs() builds tensors on the CPU first, so the workload can be sized
    # before anything touches the device. Peak device use is the input set plus
    # one clone per model call (correctness and timing each clone the inputs),
    # i.e. PEAK_INPUT_COPIES copies. On Windows/WDDM an oversized allocation does
    # not just OOM - it spills into system RAM and freezes the machine.
    raw_inputs = ref_mod.get_inputs()
    input_bytes = sum(t.numel() * t.element_size() for t in raw_inputs)
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    budget = total_bytes * args.gpu_fraction
    peak = input_bytes * PEAK_INPUT_COPIES
    if peak > budget:
        raise SystemExit(
            "refusing to run: estimated peak GPU use {:.2f} GB "
            "(inputs {:.2f} GB x {} copies) exceeds {:.0f}% of {:.2f} GB VRAM. "
            "Shrink the task size, or raise --gpu-fraction deliberately.".format(
                peak / 2**30, input_bytes / 2**30, PEAK_INPUT_COPIES,
                args.gpu_fraction * 100, total_bytes / 2**30,
            )
        )

    init_inputs = ref_mod.get_init_inputs()
    ref_model = ref_mod.Model(*init_inputs).cuda()
    cand_model = cand_mod.Model(*init_inputs).cuda()

    inputs = [x.cuda() for x in raw_inputs]
    torch.cuda.empty_cache()

    # --- correctness ---
    correct = True
    error_msg = None
    try:
        with torch.no_grad():
            ref_out = ref_model(*[x.clone() for x in inputs])
            cand_out = cand_model(*[x.clone() for x in inputs])
        for i in range(args.correct_trials):
            test_inputs = [x.clone() for x in inputs]
            ref_out = ref_model(*[x.clone() for x in test_inputs])
            cand_out = cand_model(*[x.clone() for x in test_inputs])
            if not torch.allclose(ref_out, cand_out, atol=1e-5, rtol=1e-3):
                correct = False
                error_msg = f"mismatch: max_diff={float((ref_out - cand_out).abs().max()):.6e}"
                break
    except Exception as exc:
        correct = False
        error_msg = str(exc)[:500]

    # --- timing ---
    latencies = []
    if correct:
        try:
            for _ in range(3):  # warmup
                with torch.no_grad():
                    _ = cand_model(*[x.clone() for x in inputs])
            torch.cuda.synchronize()
            for _ in range(args.perf_trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                with torch.no_grad():
                    _ = cand_model(*[x.clone() for x in inputs])
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
        except Exception:
            pass

    # --- reference timing ---
    ref_latencies = []
    try:
        torch.cuda.synchronize()
        for _ in range(args.perf_trials // 2):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            with torch.no_grad():
                _ = ref_model(*[x.clone() for x in inputs])
            end.record()
            torch.cuda.synchronize()
            ref_latencies.append(start.elapsed_time(end))
    except Exception:
        pass

    result = {
        "correct": correct,
        "error": error_msg,
        "cand_median_ms": statistics.median(latencies) if latencies else None,
        "ref_median_ms": statistics.median(ref_latencies) if ref_latencies else None,
        "speedup": (statistics.median(ref_latencies) / statistics.median(latencies))
                   if latencies and ref_latencies else None,
        "cand_latencies": latencies[:5],
        "ref_latencies": ref_latencies[:5],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
