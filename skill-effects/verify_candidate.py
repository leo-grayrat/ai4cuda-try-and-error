"""Independent execution against immutable reference code and declared cases."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class NativeComputeOnly(TorchDispatchMode):
    """Allow tensor creation/metadata while detecting dispatched tensor math.

    This checks observed dispatch during forward; it is not a whole-program
    security sandbox. It covers torch operations hidden behind nn.Module calls
    which the small AST checker cannot reliably resolve.
    """
    allowed = {"aten::empty", "aten::empty_like", "aten::empty_strided", "aten::zeros", "aten::zeros_like", "aten::ones", "aten::ones_like", "aten::full", "aten::full_like", "aten::rand", "aten::rand_like", "aten::randn", "aten::randn_like", "aten::lift_fresh", "aten::lift_fresh_copy", "aten::detach", "aten::_to_copy", "aten::copy_", "aten::zero_", "aten::fill_"}

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        name = function._schema.name
        if name not in self.allowed:
            raise RuntimeError(f"PyTorch computation during candidate forward is not permitted: {name}")
        return function(*args, **(kwargs or {}))


def native_forward(candidate, inputs):
    with NativeComputeOnly():
        return candidate(*inputs)


def compare_output(actual, expected, task):
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    if task == "softmax":
        # Elementwise atol=1e-2 alone accepts zero/sparse fake probabilities for
        # very wide rows. Also compare the distribution's total mass and L1.
        torch.testing.assert_close(actual.sum(dim=1), torch.ones_like(expected.sum(dim=1)), atol=1e-4, rtol=1e-4)
        if not bool(torch.all((actual - expected).abs().sum(dim=1) <= 1e-3)):
            raise AssertionError("softmax per-row L1 error exceeds 1e-3")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_case(reference, candidate_type, task, case, seed):
    torch.manual_seed(seed)
    if task == "axpby":
        init = [case["alpha"]]
        inputs = [torch.randn(case["shape"], device="cuda") for _ in range(2)]
    else:
        for key, value in case.items():
            setattr(reference, key, tuple(value) if key == "input_shape" else value)
        init = reference.get_init_inputs()
        inputs = [x.cuda() for x in reference.get_inputs()]
        if seed % 2:
            inputs = [x * 3 - 1.7 for x in inputs]
    original = reference.Model(*init).eval().cuda()
    candidate = candidate_type(*init).eval().cuda()
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.uniform_(-0.7, 1.3)
        candidate.load_state_dict(original.state_dict(), strict=True)
        expected = original(*inputs)
        saved_inputs = [x.clone() for x in inputs]
        actual = native_forward(candidate, inputs)
        compare_output(actual, expected, task)
        for actual_input, before in zip(inputs, saved_inputs):
            torch.testing.assert_close(actual_input, before, atol=0, rtol=0)
        saved_output = actual.clone()
        next_inputs = [x + 0.31 for x in saved_inputs]
        next_expected = original(*next_inputs)
        next_actual = native_forward(candidate, next_inputs)
        compare_output(next_actual, next_expected, task)
        torch.testing.assert_close(actual, saved_output, atol=0, rtol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            stream_actual = native_forward(candidate, saved_inputs)
        stream.synchronize()
        compare_output(stream_actual, expected, task)
        torch.cuda.synchronize()
    return {"seed": seed, "input_shapes": [list(x.shape) for x in inputs], "correct": True, "input_preserved": True, "previous_output_preserved": True, "nondefault_stream_correct": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    work, control = args.work.resolve(), args.control.resolve()
    spec = json.loads((control / "task_spec.json").read_text(encoding="utf-8"))
    reference = load("immutable_reference", control / "model.py")
    sys.path.insert(0, str(work))
    candidate = load("tested_candidate", work / "model_new.py").ModelNew
    rows = []
    result_path = work.parent / "independent_correctness.json"
    for index, case in enumerate(spec["cases"]):
        for seed in [20261019, 20261020]:
            try:
                result = verify_case(reference, candidate, spec["id"], case, seed)
                rows.append({"case_index": index, **result})
                print(f"[PASS] case={index} seed={seed}: values, parameters, repeated calls, non-default stream", flush=True)
            except Exception as exc:
                rows.append({"case_index": index, "seed": seed, "correct": False, "error": f"{type(exc).__name__}: {exc}"})
                result_path.write_text(json.dumps({"correct": False, "planned_checks": len(spec["cases"]) * 2, "checks": rows}, indent=2), encoding="utf-8")
                print(f"[FAIL] case={index} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
                return 1
    result_path.write_text(json.dumps({"correct": True, "planned_checks": len(spec["cases"]) * 2, "checks": rows}, indent=2), encoding="utf-8")
    print("[PASS] independent verification complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
