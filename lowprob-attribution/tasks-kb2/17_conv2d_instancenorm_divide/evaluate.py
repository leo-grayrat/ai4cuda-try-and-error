"""Independent evaluator: correctness on three shapes, then timing."""
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TOL = 1e-4


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # so `from reference import MODEL` works
    spec.loader.exec_module(module)
    return module


def main():
    candidate_path = Path(sys.argv[1])
    reference = _load(HERE / "reference.py", "reference")
    candidate = _load(candidate_path, "candidate")

    x = reference._MODEL  # noqa: F841  (keeps the model alive for both sides)
    probe = reference.get_inputs()[0]
    shapes = [tuple(probe.shape)]
    if len(probe.shape) >= 2:
        s = list(probe.shape)
        s[0] = max(1, s[0] // 2)
        shapes.append(tuple(s))

    results = []
    for shape in shapes:
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            expected = reference.run(x)
            try:
                actual = candidate.run(x)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"correct": False, "error": f"{type(exc).__name__}: {exc}"}))
                return
        if actual.shape != expected.shape:
            print(json.dumps({"correct": False,
                              "error": f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"}))
            return
        if not torch.allclose(actual, expected, rtol=TOL, atol=TOL):
            diff = (actual - expected).abs().max().item()
            print(json.dumps({"correct": False, "error": f"max abs diff {diff:.3e}"}))
            return
        results.append(shape)

    print(json.dumps({"correct": True, "shapes": results}))


if __name__ == "__main__":
    main()
