"""One warmed model call bracketed by cudaProfilerStart/Stop for Nsight CLI.

Run only as a child of trace_formal.py after the serial formal campaign ends.
This records activity, not the primary performance outcome.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--role", choices=["eager", "compiled", "candidate"], required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.work.resolve()))
    sys.path.insert(0, str(args.control.resolve()))
    import torch
    from utils.profiling import initialize_models
    from verify_candidate import native_forward, compare_output

    def precision():
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False

    torch.manual_seed(20261019)
    precision()
    eager, candidate, inputs, candidate_inputs = initialize_models()
    if args.role == "compiled":
        target = torch.compile(eager)
    elif args.role == "candidate":
        target = lambda *values: native_forward(candidate, values)
    else:
        target = eager
    values = candidate_inputs if args.role == "candidate" else inputs
    with torch.no_grad():
        precision()
        expected = eager(*inputs)
        for _ in range(10):
            precision()
            actual = target(*values)
        torch.cuda.synchronize()
        spec = json.loads((args.control / "task_spec.json").read_text(encoding="utf-8"))
        compare_output(actual, expected, spec["id"])
        # Initialization, comparisons, allocation of inputs, and tuning all
        # precede this range. Keep the returned tensor alive until after Stop.
        precision()
        torch.cuda.profiler.start()
        try:
            actual = target(*values)
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
        compare_output(actual, expected, spec["id"])
    print(json.dumps({"trace_worker": "complete", "role": args.role,
                      "warmup_calls": 10, "profiled_calls": 1,
                      "nominal_output_matches": True}), flush=True)


if __name__ == "__main__":
    main()
