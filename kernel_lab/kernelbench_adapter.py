"""Call KernelBench's own evaluator from a compatible KernelBench environment.

The parent lab process records candidate identity and compares runs. This module
leaves correctness and CUDA timing to upstream KernelBench.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correct-trials", type=int, default=5)
    parser.add_argument("--perf-trials", type=int, default=100)
    parser.add_argument("--backend", default="cuda", choices=["cuda", "triton"])
    args = parser.parse_args()

    import torch
    from kernelbench.eval import eval_kernel_against_ref
    from kernelbench import timing

    timing_batches = []
    original_stats = timing.get_timing_stats

    def capture_stats(elapsed_times, device=None):
        timing_batches.append([float(value) for value in elapsed_times])
        return original_stats(elapsed_times, device=device)

    timing.get_timing_stats = capture_stats
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = eval_kernel_against_ref(
                original_model_src=Path(args.reference).read_text(encoding="utf-8"),
                custom_model_src=Path(args.candidate).read_text(encoding="utf-8"),
                seed_num=args.seed,
                num_correct_trials=args.correct_trials,
                num_perf_trials=args.perf_trials,
                measure_performance=True,
                timing_method="cuda_event",
                device=torch.device("cuda:0"),
                backend=args.backend,
                check_for_excessive_speedup=True,
            )
    finally:
        timing.get_timing_stats = original_stats
    # The first timing batch belongs to ModelNew. Upstream's summary rounds its mean.
    samples = timing_batches[0] if result.correctness and timing_batches else []
    print(json.dumps({
        "correct": bool(result.correctness),
        "compiled": bool(result.compiled),
        "runtime_versions": {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda},
        "latencies_ms": samples,
        "median_ms": median(samples) if samples else None,
        "sample_source": "KernelBench timing function before summary rounding" if samples else None,
        "upstream_runtime_stats": result.runtime_stats,
        "upstream_metadata": {key: str(value) for key, value in result.metadata.items()},
    }))


if __name__ == "__main__":
    main()
