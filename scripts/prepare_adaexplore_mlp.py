"""Prepare a measured 2x2 from AdaExplore MLP nodes 7 -> 8 -> 9."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tarfile
from pathlib import Path


ARCHIVE_SHA256 = "602732f6132b7a93f9e4a9ce572b303321e60e89542591436003f0dbd0e66ee5"
PREFIX = "MCTS_0120_KB-l3_100/3_1/"
INSERTION = (
    "\n"
    "        torch.backends.cuda.matmul.allow_tf32 = True\n"
    "        torch.backends.cudnn.allow_tf32 = True\n"
)
ANCHOR = "        super(ModelNew, self).__init__()\n"


def executable_ast(source: str) -> str:
    tree = ast.parse(source)

    class RemoveDocstrings(ast.NodeTransformer):
        def visit_Module(self, node):
            self.generic_visit(node)
            node.body = [statement for statement in node.body if not isinstance(statement, ast.Expr)
                         or not isinstance(statement.value, ast.Constant)
                         or not isinstance(statement.value.value, str)]
            return node

        def visit_ClassDef(self, node):
            self.generic_visit(node)
            node.body = [statement for statement in node.body if not isinstance(statement, ast.Expr)
                         or not isinstance(statement.value, ast.Constant)
                         or not isinstance(statement.value.value, str)]
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.body = [statement for statement in node.body if not isinstance(statement, ast.Expr)
                         or not isinstance(statement.value, ast.Constant)
                         or not isinstance(statement.value.value, str)]
            return node

    return ast.dump(RemoveDocstrings().visit(tree), include_attributes=False)


def prepare(archive_path: Path, reference_path: Path, output: Path) -> dict:
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("AdaExplore Level 3 archive checksum differs")
    source = {}
    ancestry = {}
    original_metrics = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for index in (7, 8, 9):
            raw = archive.extractfile(PREFIX + f"step_{index}.py")
            log = archive.extractfile(PREFIX + f"step_{index}_log.json")
            metrics = archive.extractfile(PREFIX + f"step_{index}_metrics.json")
            if raw is None or log is None or metrics is None:
                raise ValueError(f"Missing step {index} artifact")
            source[index] = raw.read().decode("utf-8")
            ancestry[index] = json.load(log)["parent_node_id"]
            original_metrics[index] = json.load(metrics)
    if ancestry[8] != 7 or ancestry[9] != 8:
        raise ValueError("Selected nodes do not form the expected parent chain")
    if not all(original_metrics[index]["correctness"] for index in (7, 8, 9)):
        raise ValueError("Selected archive candidates were not all correct")
    if source[7].count(ANCHOR) != 1 or source[8].count(ANCHOR) != 1:
        raise ValueError("ModelNew constructor anchor changed")
    base, only_a, both = source[7], source[8], source[9]
    only_b = base.replace(ANCHOR, ANCHOR + INSERTION, 1)
    expected_both = only_a.replace(ANCHOR, ANCHOR + INSERTION, 1)
    if executable_ast(expected_both) != executable_ast(both):
        raise ValueError("Archive step 8 -> 9 contains executable changes beyond TF32 flags")

    output.mkdir(parents=True, exist_ok=True)
    for name, code in (("base", base), ("A", only_a), ("B", only_b), ("AB", both)):
        (output / f"{name}.py").write_text(code, encoding="utf-8")
    (output / "reference.py").write_bytes(reference_path.read_bytes())
    manifest = {
        "schema_version": 1,
        "experiment_id": "adaexplore-level3-mlp-7-8-9-tf32-relu",
        "description": "Real AdaExplore parent chain 7->8->9; B is the independently constructed TF32-only counterfactual.",
        "evaluator": "kernelbench",
        "reference_path": "reference.py",
        "workload": {"seed": 42, "correct_trials": 5, "perf_trials": 30, "backend": "triton"},
        "factors": ["torch_inplace_relu", "tf32_flags"],
        "candidates": [
            {"id": "base", "changes": [], "source_path": "base.py"},
            {"id": "A", "changes": ["torch_inplace_relu"], "source_path": "A.py"},
            {"id": "B", "changes": ["tf32_flags"], "source_path": "B.py"},
            {"id": "AB", "changes": ["torch_inplace_relu", "tf32_flags"], "source_path": "AB.py"},
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "source": "https://huggingface.co/datasets/VanishD/AdaExplore_Traces",
        "archive_sha256": ARCHIVE_SHA256,
        "task": "3_1",
        "archive_nodes": {str(index): {"parent": ancestry[index],
                                       "source_sha256": hashlib.sha256(source[index].encode()).hexdigest(),
                                       "archived_runtime": original_metrics[index]["runtime"],
                                       "archived_device": original_metrics[index]["metadata"].get("device")}
                          for index in (7, 8, 9)},
        "counterfactual_B_sha256": hashlib.sha256(only_b.encode()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.reference, args.output), indent=2))


if __name__ == "__main__":
    main()
