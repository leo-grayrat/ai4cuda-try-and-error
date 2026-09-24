"""Build a deterministic, read-only index of archive parent-child edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_adaexplore import CODE, STEP


def index_edges(path: Path) -> dict:
    """Include every parent edge and why it is or is not eligible for screening."""
    archive_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    nodes: dict[str, dict[int, dict]] = defaultdict(dict)
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            code_match = CODE.fullmatch(member.name)
            step_match = STEP.fullmatch(member.name)
            if code_match is None and step_match is None:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read {member.name}")
            if code_match is not None:
                task, number = code_match.group(1), int(code_match.group(2))
                nodes[task].setdefault(number, {})["code_hash"] = hashlib.sha256(stream.read()).hexdigest()
            else:
                task, number, kind = step_match.group(1), int(step_match.group(2)), step_match.group(3)
                nodes[task].setdefault(number, {})[kind] = json.load(stream)

    rows = []
    counts = Counter()
    for task in sorted(nodes, key=lambda value: tuple(int(part) for part in value.split("_"))):
        task_nodes = nodes[task]
        for child_number in sorted(task_nodes):
            if child_number == 0:
                continue
            child = task_nodes[child_number]
            child_log, child_metrics = child.get("log"), child.get("metrics")
            if child_log is None or child_metrics is None:
                raise ValueError(f"{task} step {child_number} lacks log or metrics")
            if child_log.get("node_id") != child_number:
                raise ValueError(f"{task} step {child_number} has mismatched node_id")
            parent_number = child_log.get("parent_node_id")
            if (not isinstance(parent_number, int) or parent_number >= child_number
                    or parent_number not in task_nodes):
                raise ValueError(f"{task} step {child_number} has invalid parent {parent_number}")
            parent = task_nodes[parent_number]
            parent_metrics = parent.get("metrics")
            if parent_metrics is None:
                raise ValueError(f"{task} step {parent_number} lacks metrics")
            child_hardware = child_metrics.get("metadata") or {}
            parent_hardware = parent_metrics.get("metadata") or {}
            if parent_number == 0:
                status = "root_parent"
            elif not parent.get("code_hash") or not child.get("code_hash"):
                status = "missing_source"
            elif parent_metrics.get("correctness") is not True or child_metrics.get("correctness") is not True:
                status = "archived_incorrect"
            elif (not child_hardware.get("device") or
                  (parent_hardware.get("hardware"), parent_hardware.get("device")) !=
                  (child_hardware.get("hardware"), child_hardware.get("device"))):
                status = "different_device"
            elif parent["code_hash"] == child["code_hash"]:
                status = "identical_source"
            else:
                status = "eligible"
            counts[status] += 1
            rows.append({
                "task": task,
                "parent_node": parent_number,
                "child_node": child_number,
                "status": status,
                "parent_code_sha256": parent.get("code_hash"),
                "child_code_sha256": child.get("code_hash"),
            })
    return {
        "schema_version": 1,
        "archive": path.name,
        "archive_sha256": archive_sha256,
        "order": "numeric task, child node, parent node; archive runtime unused",
        "counts": dict(counts),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    indexed = index_edges(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(indexed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive_sha256": indexed["archive_sha256"], "counts": indexed["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
