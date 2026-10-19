"""Observe the pinned FlashRT executor's declared-state enforcement boundary.

This is a CPU mechanism probe of original code, not an app benchmark or a
candidate implementation. Does not modify the downloaded repository.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    sys.path.insert(0, str(repository))
    from flashrt.ir import (IRGraph, IROperator, OpType, SequentialExecutor,
                            StateObject, analyze_cross_chunk_dependencies)

    rows = []
    for mode, reads, writes in [
        ("undeclared_get", [], []), ("undeclared_set", ["counter"], []),
        ("declared_set", [], ["counter"]), ("read_declared_inplace_mutation", ["counter"], []),
    ]:
        class ProbeOperator(IROperator):
            def execute(self, inputs, context, state):
                if mode == "undeclared_get":
                    state.get("counter")
                elif mode in {"undeclared_set", "declared_set"}:
                    state.set("counter", [1])
                else:
                    state.get("counter").append(1)
                return {}

        op = ProbeOperator(name="probe", op_type=OpType.EXPOSED,
                           state_reads=reads, state_writes=writes)
        graph = IRGraph(name=mode, operators={"probe": op},
                        state_objects={"counter": StateObject(name="counter")}, chunk_ops=["probe"])
        executor = SequentialExecutor(graph)
        executor.init_state({"counter": []})
        row = {"case": mode, "declared_reads": reads, "declared_writes": writes,
               "graph_validation_errors": graph.validate(),
               "cross_chunk_analysis": asdict(analyze_cross_chunk_dependencies(graph))}
        try:
            executor.run_chunk({})
            row["outcome"] = "executed"
        except RuntimeError as exc:
            row.update(outcome="rejected", error=str(exc))
        row["counter_after"] = list(executor.state["counter"])
        rows.append(row)
    assert [row["outcome"] for row in rows] == ["rejected", "rejected", "executed", "executed"]
    assert rows[2]["counter_after"] == [1] and rows[3]["counter_after"] == [1]
    source_hashes = {name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
                     for name in ["AGENTS.md", "flashrt/ir/graph.py", "flashrt/ir/executor.py", "flashrt/ir/analysis.py"]}
    result = {"scope": "CPU probe of original IR executor; not end-to-end application validation or performance evidence",
              "repository": str(repository), "source_sha256": source_hashes,
              "python": sys.executable, "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "outcomes": [row["outcome"] for row in rows],
                      "read_declared_inplace_mutation_changed_state": rows[3]["counter_after"] == [1]}))


if __name__ == "__main__":
    main()
