"""Turn KernelBench level-2 problems into this repo's run(x) task format.

KernelBench gives a `Model` class plus `get_inputs()` / `get_init_inputs()`. The
driver here wants `reference.py` exposing `run(x)`. The wrapper instantiates the
model once with the declared init inputs and closes over it, so the shape of the
task the model sees is "one tensor in, one tensor out" -- the same shape as the
four reduction tasks, but the computation behind it is a chain of 3-6 operators
that has to be fused.

Level 2 is the point: fusion means the implementer has to decide what to fuse,
whether to materialise intermediates, and how to tile, which the reduction tasks
had almost none of.

Usage:
    python make_kb2_tasks.py --out ../tasks-kb2 --limit 6
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

KB = Path(__file__).resolve().parents[2] / ".cache" / "KernelBench" / "KernelBench" / "level2"

REFERENCE = '''"""Reference for {name}.

Auto-generated from KernelBench level 2. The model is instantiated once with the
declared init inputs; `run` is the single-tensor entry point the driver uses.
"""
import torch
import torch.nn as nn


{model_source}


torch.manual_seed(0)

_INIT = get_init_inputs()
_MODEL = Model(*_INIT).cuda().eval()
MODEL = _MODEL          # exported so a candidate can read the same weights


def run(x):
    with torch.no_grad():
        return _MODEL(x)
'''

CANDIDATE = '''import torch
import triton
import triton.language as tl

from reference import MODEL   # the weights, fixed by seed


def run(x):
    """Replace this with a Triton implementation. x is a CUDA tensor."""
    raise NotImplementedError
'''

TASK_MD = '''Implement `run(x)` in `candidate.py` for the computation in `reference.py`.

The reference is a chain of {n_ops} operators: {chain}. It is currently implemented as
a sequence of PyTorch ops, so each intermediate result is materialised to global
memory. A single fused Triton kernel, or a small number of them, should be able
to do the same work without those round trips.

The weights are available as `MODEL` (import it from `reference`); the seed is
fixed, so both sides see the same parameters.

Use a Triton GPU kernel (`@triton.jit` with `tl.load` / `tl.store`). Calling the
PyTorch ops directly, or any torch/numpy equivalent, is not acceptable.
'''

EVALUATOR = '''"""Independent evaluator: correctness on three shapes, then timing."""
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
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("../tasks-kb2"))
    ap.add_argument("--names", nargs="*", default=None,
                    help="explicit KernelBench filenames (without .py)")
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.names:
        files = [KB / f"{n}.py" for n in args.names]
    else:
        files = sorted(KB.glob("*.py"))[: args.limit]

    for path in files:
        source = path.read_text(encoding="utf-8")
        # keep everything up to and including get_init_inputs / get_inputs
        model_part = source
        name = path.stem
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        target = args.out / slug
        target.mkdir(parents=True, exist_ok=True)

        (target / "reference.py").write_text(
            REFERENCE.format(name=name, model_source=model_part), encoding="utf-8")
        (target / "candidate.py").write_text(CANDIDATE, encoding="utf-8")
        (target / "evaluate.py").write_text(EVALUATOR, encoding="utf-8")

        chain = " + ".join(w for w in name.split("_")[1:] if not w.isdigit())
        (target / "task.md").write_text(
            TASK_MD.format(n_ops=len(name.split("_")) - 1, chain=chain or name),
            encoding="utf-8")
        print(f"  {slug}")


if __name__ == "__main__":
    main()
