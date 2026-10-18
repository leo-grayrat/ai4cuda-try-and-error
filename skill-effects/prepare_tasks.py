"""Freeze real KernelBench Model code with explicit input-size adapters."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def model_ast(source: str) -> str:
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Model")
    return ast.dump(node, include_attributes=False)


def render_model(source: str, overrides: dict) -> str:
    # Keep the full original text as a prefix. Model/get_inputs/get_init_inputs
    # are unchanged; these globals are looked up when the factory is called.
    for key in overrides:
        if not key.isidentifier():
            raise ValueError("invalid override name")
    addition = "\n\n# Study input-size adapter; original Model implementation above is unchanged.\n"
    for key, value in overrides.items():
        if key == "input_shape":
            value = tuple(value)
        addition += f"{key} = {value!r}\n"
    rendered = source + (addition if overrides else "")
    if model_ast(source) != model_ast(rendered):
        raise ValueError("Model semantics changed")
    return rendered


def catalog():
    return json.loads((ROOT / "task_catalog.json").read_text(encoding="utf-8"))


def main():
    spec = catalog()
    cache = ROOT.parent / ".cache/KernelBench"
    git = ["git", "-c", f"safe.directory={cache.as_posix()}", "-C", str(cache)]
    source_dir = ROOT / "sources/KernelBench"
    source_dir.mkdir(exist_ok=True)
    (source_dir / "LICENSE").write_bytes(subprocess.check_output(git + ["show", f"{spec['source_commit']}:LICENSE"]))
    manifest = []
    for task in spec["tasks"]:
        data = subprocess.check_output(git + ["show", f"{spec['source_commit']}:KernelBench/{task['source']}"])
        source_path = source_dir / task["source"]
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(data)
        rendered = render_model(data.decode("utf-8"), task["overrides"])
        directory = ROOT / "tasks" / task["id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.py").write_text(rendered, encoding="utf-8")
        (directory / "contract.md").write_text(f"# {task['id']}\n\nImplement the Model in model.py as ModelNew. Support contiguous float32 CUDA inputs, constructor settings and shapes in the following configurations, including different random seeds, non-default streams and repeated calls. Preserve state_dict compatibility, including randomized affine/linear parameters. Do not modify inputs; previously returned outputs must remain unchanged after later calls.\n\n```json\n{json.dumps(task['cases'], indent=2)}\n```\n", encoding="utf-8")
        if task.get("operator_checks"):
            with (directory / "contract.md").open("a", encoding="utf-8") as stream:
                stream.write("\nAdditional softmax distribution checks: row sums must match 1 with atol=rtol=1e-4, and each row's L1 error versus the reference must be <=1e-3. Elementwise atol=rtol=1e-2 alone is insufficient for very wide probability vectors.\n")
        manifest.append({**task, "source_sha256": hashlib.sha256(data).hexdigest(), "model_sha256": hashlib.sha256((directory / "model.py").read_bytes()).hexdigest(), "normalized_model_text_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "model_ast_unchanged": True})
    (ROOT / "tasks/manifest.json").write_text(json.dumps({"repository": spec["source_repository"], "commit": spec["source_commit"], "status": spec["status"], "tasks": manifest}, indent=2), encoding="utf-8")
    print(json.dumps({"prepared": [r["id"] for r in manifest], "unchanged_Model_AST": True}))


if __name__ == "__main__":
    main()
