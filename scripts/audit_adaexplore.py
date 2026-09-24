"""Audit explicit parent links in an AdaExplore run archive without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


STEP = re.compile(r"^[^/]+/([23]_\d+)/step_(\d+)_(log|metrics)\.json$")
CODE = re.compile(r"^[^/]+/([23]_\d+)/step_(\d+)\.py$")


def audit(path: Path) -> dict:
    tasks: dict[str, dict[int, dict]] = defaultdict(dict)
    codes: set[tuple[str, int]] = set()
    code_hashes: dict[tuple[str, int], str] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            code_match = CODE.fullmatch(member.name)
            if code_match:
                key = code_match.group(1), int(code_match.group(2))
                codes.add(key)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Missing archive member content: {member.name}")
                code_hashes[key] = hashlib.sha256(stream.read()).hexdigest()
                continue
            match = STEP.fullmatch(member.name)
            if not match:
                continue
            task, index, kind = match.group(1), int(match.group(2)), match.group(3)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing archive member content: {member.name}")
            data = json.load(stream)
            tasks[task].setdefault(index, {})[kind] = data

    counts = Counter()
    examples = []
    two_step_chains = []
    identical_code_edges = []
    hardware = Counter()
    for task, nodes in sorted(tasks.items()):
        counts["tasks"] += 1
        if 0 not in nodes:
            raise ValueError(f"{task} has no root node")
        for index, entry in sorted(nodes.items()):
            log = entry.get("log")
            metrics = entry.get("metrics")
            if log is None or metrics is None:
                raise ValueError(f"{task} step {index} lacks log or metrics")
            if log["node_id"] != index:
                raise ValueError(f"{task} step {index} has mismatched node_id")
            if index == 0:
                if log["parent_node_id"] is not None:
                    raise ValueError(f"{task} root has a parent")
                continue
            counts["candidate_nodes"] += 1
            if (task, index) in codes:
                counts["candidate_code_files"] += 1
            parent_id = log["parent_node_id"]
            if not isinstance(parent_id, int) or parent_id >= index or parent_id not in nodes:
                raise ValueError(f"{task} step {index} has invalid parent {parent_id}")
            counts["explicit_parent_edges"] += 1
            child_meta = metrics.get("metadata") or {}
            if child_meta.get("hardware"):
                hardware[child_meta["hardware"]] += 1
            child_runtime = metrics.get("runtime")
            child_valid = (metrics.get("correctness") is True and isinstance(child_runtime, (int, float))
                           and math.isfinite(child_runtime) and child_runtime > 0)
            if child_valid:
                counts["correct_timed_nodes"] += 1
            if parent_id == 0:
                continue
            parent_metrics = nodes[parent_id]["metrics"]
            parent_runtime = parent_metrics.get("runtime")
            parent_valid = (parent_metrics.get("correctness") is True and isinstance(parent_runtime, (int, float))
                            and math.isfinite(parent_runtime) and parent_runtime > 0)
            if not (child_valid and parent_valid):
                continue
            counts["correct_timed_parent_child_edges"] += 1
            parent_meta = parent_metrics.get("metadata") or {}
            same_device = (child_meta.get("hardware") == parent_meta.get("hardware")
                           and child_meta.get("device") == parent_meta.get("device")
                           and child_meta.get("device") is not None)
            if same_device:
                counts["same_device_edges"] += 1
                if len(examples) < 12:
                    examples.append({"task": task, "parent": parent_id, "child": index,
                                     "parent_runtime": parent_runtime, "child_runtime": child_runtime,
                                     "device": child_meta["device"]})
            if ((task, parent_id) in code_hashes and (task, index) in code_hashes
                    and code_hashes[(task, parent_id)] == code_hashes[(task, index)]):
                counts["identical_code_correct_edges"] += 1
                if same_device:
                    counts["same_device_identical_code_edges"] += 1
                    identical_code_edges.append({"task": task, "parent": parent_id, "child": index,
                                                 "parent_runtime": parent_runtime, "child_runtime": child_runtime,
                                                 "multiplicative_difference": max(parent_runtime / child_runtime,
                                                                                  child_runtime / parent_runtime),
                                                 "device": child_meta["device"]})
            grandparent_id = nodes[parent_id]["log"]["parent_node_id"]
            if not isinstance(grandparent_id, int) or grandparent_id == 0:
                continue
            grandparent_metrics = nodes[grandparent_id]["metrics"]
            grandparent_runtime = grandparent_metrics.get("runtime")
            grandparent_meta = grandparent_metrics.get("metadata") or {}
            if (same_device and grandparent_metrics.get("correctness") is True
                    and isinstance(grandparent_runtime, (int, float))
                    and math.isfinite(grandparent_runtime) and grandparent_runtime > 0
                    and grandparent_meta.get("hardware") == child_meta.get("hardware")
                    and grandparent_meta.get("device") == child_meta.get("device")):
                counts["same_device_correct_two_step_chains"] += 1
                if (code_hashes[(task, grandparent_id)] != code_hashes[(task, parent_id)]
                        and code_hashes[(task, parent_id)] != code_hashes[(task, index)]):
                    counts["two_step_chains_with_two_code_changes"] += 1
                    two_step_chains.append({"task": task, "nodes": [grandparent_id, parent_id, index],
                                            "runtimes": [grandparent_runtime, parent_runtime, child_runtime],
                                            "device": child_meta["device"],
                                            "speedup_first_to_last": grandparent_runtime / child_runtime})
    two_step_chains.sort(key=lambda chain: chain["speedup_first_to_last"], reverse=True)
    identical_code_edges.sort(key=lambda edge: edge["multiplicative_difference"], reverse=True)
    review_candidates = [chain for chain in two_step_chains
                         if 1.02 < chain["runtimes"][0] / chain["runtimes"][1] < 4
                         and 1.02 < chain["runtimes"][1] / chain["runtimes"][2] < 4]
    return {"archive": path.name, "counts": dict(counts), "hardware_labels": dict(hardware),
            "same_device_edge_examples": examples,
            "largest_same_device_identical_code_differences": identical_code_edges[:12],
            "top_same_device_two_step_chains": two_step_chains[:12],
            "two_step_review_candidates": review_candidates[:12],
            "interpretation": "Parent links are explicit. Reported runtimes are archived summaries, not local remeasurements or independent 2x2 interventions."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.archive)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
