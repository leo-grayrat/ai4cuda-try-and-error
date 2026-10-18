"""Shared experimental tool boundary and conservative source checks."""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiler_feedback(output: str, limit: int = 6000) -> str:
    if len(output) <= limit:
        return output
    lines = output.splitlines()
    indices = set()
    for index, line in enumerate(lines):
        if re.search(r"\berror\b|\bfatal\b|^FAILED:", line, flags=re.I):
            indices.update(range(max(0, index - 1), min(len(lines), index + 3)))
    diagnostic = "\n".join(lines[i] for i in sorted(indices))
    return (output[:1000] + "\n[diagnostic excerpts; complete raw log retained]\n" + diagnostic[:3000] + "\n[last log lines]\n" + output[-1800:])[:limit]


def summarize_usage(rows: list[dict]) -> dict:
    result = dict.fromkeys(["input_tokens", "uncached_input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"], 0)
    reasoning_available = True
    for row in rows:
        prompt, completion = row["prompt_tokens"], row["completion_tokens"]
        if row["total_tokens"] != prompt + completion:
            raise ValueError("API token total does not match prompt + completion")
        hit, miss = row["prompt_cache_hit_tokens"], row["prompt_cache_miss_tokens"]
        if hit + miss != prompt:
            raise ValueError("API cache breakdown does not match prompt tokens")
        reasoning = row.get("completion_tokens_details", {}).get("reasoning_tokens")
        if reasoning is None:
            reasoning_available = False
        elif not 0 <= reasoning <= completion:
            raise ValueError("reasoning tokens must be included in completion")
        result["input_tokens"] += prompt
        result["uncached_input_tokens"] += miss
        result["cached_input_tokens"] += hit
        result["output_tokens"] += completion
        result["total_tokens"] += row["total_tokens"]
        result["reasoning_tokens"] += reasoning or 0
    if not reasoning_available:
        result["reasoning_tokens"] = None
    return result


class FileTools:
    """Trial-local paths in all arms; candidate-only writes in permission arms.

    This is tool permission enforcement, not an OS sandbox for arbitrary native
    code. Native code execution is separately checked against frozen files.
    """
    def __init__(self, root: Path, permissions: bool):
        self.root = root.resolve()
        self.permissions = permissions

    def path(self, name: str, write: bool = False) -> Path:
        name = name.replace("\\", "/")
        parts = PurePosixPath(name).parts
        reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}
        if not parts or name.startswith("/") or ":" in name or any(p in ("..", ".") or p.rstrip(" .") != p or p.split(".")[0].upper() in reserved for p in parts):
            raise ValueError("invalid trial-local path")
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise PermissionError("path leaves trial workspace")
        if write and self.permissions:
            allowed = name == "model_new.py" or (parts[0] == "kernels" and path.suffix in {".cu", ".cpp", ".h", ".cuh"})
            if not allowed:
                raise PermissionError("write denied: candidate files are model_new.py and kernels/ sources")
        return path

    def read(self, name: str) -> str:
        return self.path(name).read_text(encoding="utf-8")

    def write(self, name: str, content: str):
        path = self.path(name, write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def delete(self, name: str):
        self.path(name, write=True).unlink()

    def list(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)).replace("\\", "/") for p in self.root.rglob("*") if p.is_file() and p.suffix in {".py", ".cu", ".cpp", ".h", ".cuh", ".md", ".sh"} and not any(part in {"build", "__pycache__"} for part in p.relative_to(self.root).parts))


def source_checks(work: Path) -> dict[str, str]:
    """Return definite findings only; numerical behavior requires execution.

    The Python check covers direct torch/functional calls and tensor arithmetic
    in forward, not arbitrary whole-program equivalence or all possible aliases.
    Findings are scoped to this checker, never a universal conformance score.
    """
    findings = {}
    kernels = list((work / "kernels").glob("*.cu"))
    if not kernels:
        findings["missing_cuda_kernel"] = "kernels/ has no .cu source"
    if not list((work / "kernels").glob("*_binding.cpp")):
        findings["missing_binding"] = "kernels/ has no *_binding.cpp source"
    model = work / "model_new.py"
    if not model.exists():
        findings["missing_model_new"] = "model_new.py is absent"
        return findings
    try:
        tree = ast.parse(model.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        findings["python_syntax"] = str(exc)
        return findings
    torch_aliases = {"torch", "F", "nn"}
    imported_functions = {}
    known_compute = {"relu", "gelu", "sigmoid", "softmax", "log_softmax", "layer_norm", "rms_norm", "batch_norm", "group_norm", "linear", "matmul", "mm", "bmm", "addmm", "einsum", "conv1d", "conv2d", "conv3d", "convolution", "sum", "mean", "var", "std", "cumsum", "max", "min", "add", "sub", "mul", "div", "pow", "sqrt", "rsqrt", "exp", "log", "sin", "cos", "tanh", "clamp", "norm", "normalize", "reshape", "view", "transpose", "permute"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    torch_aliases.add(alias.asname or alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("torch"):
            torch_aliases.update(a.asname or a.name for a in node.names)
            imported_functions.update({a.asname or a.name: a.name for a in node.names})
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward":
            tensor_args = {arg.arg for arg in node.args.args if arg.arg != "self"}
            for child in ast.walk(node):
                if isinstance(child, ast.BinOp) and any(isinstance(operand, ast.Name) and operand.id in tensor_args for operand in [child.left, child.right]):
                    findings["python_compute"] = f"direct forward-input arithmetic at line {child.lineno}; applies to the study's tensor-only forward inputs"
                if isinstance(child, ast.Call):
                    text = ast.unparse(child.func)
                    name = imported_functions.get(text, text.split(".")[-1])
                    tensor_method = isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name) and child.func.value.id in tensor_args
                    if name in known_compute and (text.split(".")[0] in torch_aliases or tensor_method):
                        findings["python_compute"] = f"known tensor computation in forward at line {child.lineno}: {text}"
    stale = [p.name for p in (work / "kernels").glob("*") if re.search(r"(?:_old|_test|_v\d+)\.(?:cu|cpp)$|\.bak$", p.name)]
    if stale:
        findings["intermediate_files"] = ", ".join(stale)
    return findings
