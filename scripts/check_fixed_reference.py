"""Check a KernelBench candidate against a reference with TF32 held off."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    reference = load_module(args.reference, "fixed_reference")
    candidate = load_module(args.candidate, "fixed_candidate")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    init_inputs = reference.get_init_inputs()
    with torch.no_grad():
        torch.manual_seed(args.seed)
        model = reference.Model(*init_inputs).to(device)
        torch.manual_seed(args.seed)
        model_new = candidate.ModelNew(*init_inputs).to(device)
        results = []
        for trial in range(args.trials):
            torch.manual_seed(args.seed + trial + 1)
            inputs = [value.to(device) for value in reference.get_inputs()]
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            expected = model(*inputs)
            torch.cuda.synchronize(device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            actual = model_new(*inputs)
            torch.cuda.synchronize(device)
            difference = (expected - actual).abs()
            results.append({
                "allclose_atol_rtol_1e-2": bool(torch.allclose(expected, actual, atol=1e-2, rtol=1e-2)),
                "max_abs_diff": float(difference.max()),
                "mean_abs_diff": float(difference.mean()),
            })
    print(json.dumps({"reference": str(args.reference), "candidate": str(args.candidate),
                      "reference_tf32": False, "candidate_tf32": True, "trials": results}, indent=2))


if __name__ == "__main__":
    main()
