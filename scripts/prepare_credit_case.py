"""Prepare a code-first attribution case for the existing KernelBench runner."""

from __future__ import annotations

import argparse
import ast
import json
import tarfile
from pathlib import Path

from kernel_lab.code_credit import locate_actions, revert_action
from kernel_lab.core import source_hash


def load_ada_edge(archive_path: Path, task: str, parent_node: int, child_node: int) -> tuple[str, str, dict]:
    """Read one explicitly parent-linked AdaExplore edge without extraction."""
    with tarfile.open(archive_path, "r:gz") as archive:
        names = {member.name for member in archive if member.isfile()}

        def read(suffix: str) -> bytes:
            matches = [name for name in names if name.endswith(f"/{task}/{suffix}")]
            if len(matches) != 1:
                raise ValueError(f"Expected one archive member for {task}/{suffix}, found {len(matches)}")
            stream = archive.extractfile(matches[0])
            if stream is None:
                raise ValueError(f"Cannot read {matches[0]}")
            return stream.read()

        child_log = json.loads(read(f"step_{child_node}_log.json"))
        parent_metrics = json.loads(read(f"step_{parent_node}_metrics.json"))
        child_metrics = json.loads(read(f"step_{child_node}_metrics.json"))
        if child_log.get("parent_node_id") != parent_node:
            raise ValueError("Selected nodes are not an explicit parent-child edge")
        if parent_metrics.get("correctness") is not True or child_metrics.get("correctness") is not True:
            raise ValueError("Both archived candidates must be marked correct")
        parent_hardware = parent_metrics.get("metadata") or {}
        child_hardware = child_metrics.get("metadata") or {}
        if (parent_hardware.get("hardware"), parent_hardware.get("device")) != (
            child_hardware.get("hardware"), child_hardware.get("device")
        ):
            raise ValueError("Archived parent and child used different devices")
        parent = read(f"step_{parent_node}.py").decode("utf-8")
        child = read(f"step_{child_node}.py").decode("utf-8")
        if source_hash(parent.encode()) == source_hash(child.encode()):
            raise ValueError("Parent and child source code is identical")
        return parent, child, {
            "archive": archive_path.name,
            "task": task,
            "parent_node": parent_node,
            "child_node": child_node,
            "archived_hardware": parent_hardware,
            "archived_runtime": {"parent": parent_metrics.get("runtime"), "child": child_metrics.get("runtime")},
        }


def prepare_case(
    *, parent_source: str, child_source: str, reference_source: str,
    out_dir: Path, experiment_id: str, provenance: dict,
) -> dict:
    """Write sources, selected reversions, and a trace with no invented token evidence."""
    if out_dir.exists():
        raise FileExistsError(out_dir)
    actions = locate_actions(parent_source, child_source, language="python")
    selected = [action for action in actions if action["score"] > 0][:3]
    control = next((action for action in actions if action["score"] == 0), None)
    if control is not None:
        selected.append(control)
    out_dir.mkdir(parents=True)
    (out_dir / "parent.py").write_text(parent_source, encoding="utf-8")
    (out_dir / "child.py").write_text(child_source, encoding="utf-8")
    (out_dir / "reference.py").write_text(reference_source, encoding="utf-8")
    candidates = [
        {"id": "parent", "source_path": "parent.py", "changes": []},
        {"id": "child", "source_path": "child.py", "changes": ["all"]},
    ]
    candidate_sha256 = {
        "parent": source_hash(parent_source.encode()),
        "child": source_hash(child_source.encode()),
    }
    trace_actions = []
    for action in actions:
        record = {**action, "candidate_id": None, "probe_status": "not_selected", "evidence": None}
        if action in selected:
            candidate = revert_action(parent_source, child_source, action)
            if candidate == parent_source:
                record["candidate_id"] = "parent"
                record["probe_status"] = "endpoint_parent"
            elif candidate == child_source:
                record["probe_status"] = "duplicate_child"
            else:
                try:
                    ast.parse(candidate)
                except SyntaxError:
                    record["probe_status"] = "invalid_syntax"
                else:
                    candidate_id = f"without-{action['id']}"
                    filename = f"{candidate_id}.py"
                    (out_dir / filename).write_text(candidate, encoding="utf-8")
                    candidates.append({"id": candidate_id, "source_path": filename, "changes": ["all_except", action["id"]]})
                    candidate_sha256[candidate_id] = source_hash(candidate.encode())
                    record["candidate_id"] = candidate_id
                    record["probe_status"] = "ready_for_verification"
        trace_actions.append(record)
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "evaluator": "kernelbench",
        "reference_path": "reference.py",
        "workload": {
            "seed": 42, "correct_trials": 5, "perf_trials": 30,
            "backend": "triton" if "@triton.jit" in parent_source + child_source else "cuda",
        },
        "candidates": candidates,
    }
    trace = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "provenance": provenance,
        "source_sha256": {
            "parent": source_hash(parent_source.encode()),
            "child": source_hash(child_source.encode()),
            "reference": source_hash(reference_source.encode()),
        },
        "candidate_sha256": candidate_sha256,
        "actions": trace_actions,
        "raw_response": None,
        "response_tokens": None,
        "source_to_response": None,
        "measurement_records": [],
        "control_status": "selected" if control is not None else "no_changed_boilerplate",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    screen_manifest = {**manifest, "candidates": candidates[:2]}
    (out_dir / "screen-manifest.json").write_text(
        json.dumps(screen_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("task")
    parser.add_argument("parent_node", type=int)
    parser.add_argument("child_node", type=int)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    parent, child, provenance = load_ada_edge(args.archive, args.task, args.parent_node, args.child_node)
    trace = prepare_case(
        parent_source=parent, child_source=child,
        reference_source=args.reference.read_text(encoding="utf-8"),
        out_dir=args.output,
        experiment_id=f"ada-{args.task}-{args.parent_node}-{args.child_node}",
        provenance=provenance,
    )
    print(json.dumps({"experiment_id": trace["experiment_id"], "actions": len(trace["actions"]),
                      "prepared_candidates": sum(action["candidate_id"] is not None for action in trace["actions"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
