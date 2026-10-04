#!/usr/bin/env python3
"""Stage 1.5 step 1-2: select successful parent->child edges from AdaExplore MCTS
trajectories and dump their unified diffs.

What counts as an edge (SPEC.md section 6):
  - parent and child both have correctness == true (and compiled == true)
  - child runtime improved by > 5% relative to parent
  - parent/child relation is taken from the explicit MCTS field
    `parent_node_id` in step_N_log.json (NOT from adjacent step numbers)

Outputs:
  data/edges/edge_index.json      - all candidate edges with metadata
  data/edges/selected_edges.json  - the stratified sample actually annotated
  data/edges/<edge_id>.diff       - unified diff text

This script computes structural diff statistics only (hunk count, changed lines,
regions *by line offset of the parent file*, and the parent's top-level construct
names). It does NOT classify what optimization was performed: transformation
labels are produced separately by reading diffs, per the methodology red line
("no classifier / locator / AST pipeline").
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "adaexplore"
OUT = ROOT / "data" / "edges"

IMPROVE_THRESHOLD = 0.95  # child_runtime < parent_runtime * 0.95  => >5% faster
TARGET_EDGES = 55
MAX_EDGES_PER_TASK = 3
L3_TASK_SAMPLE = 15
RNG_SEED = 20261002

RUNTIME_KEYS = ("mean", "median", "value")


def read_metrics(task_dir: Path, node: int) -> dict | None:
    p = task_dir / f"step_{node}_metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_log(task_dir: Path, node: int) -> dict | None:
    p = task_dir / f"step_{node}_log.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def get_runtime(m: dict) -> float | None:
    if not m.get("correctness"):
        return None
    rs = m.get("runtime_stats") or {}
    for k in RUNTIME_KEYS:
        v = rs.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    v = m.get("runtime")
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def get_fast_p(m: dict) -> float | None:
    rs = m.get("runtime_stats") or {}
    v = rs.get("fast_p")
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def parent_constructs(src: list[str]) -> list[dict]:
    """Top-level def/class names with line offsets. Purely structural: used to
    report *where* hunks land (kernel body vs host wrapper), not what changed."""
    out = []
    for i, ln in enumerate(src):
        m = re.match(r"^(?:def|class|@triton\.jit|async def)\s+([\w\.]+)", ln.strip())
        if m:
            out.append({"name": m.group(1), "line": i})
    if not out:
        out = [{"name": "<module>", "line": 0}]
    return out


def hunk_regions(diff_lines: list[str], constructs: list[dict]) -> list[str]:
    regions = []
    for ln in diff_lines:
        m = re.match(r"^@@ -(\d+)", ln)
        if not m:
            continue
        start = int(m.group(1)) - 1
        owner = constructs[0]["name"]
        for c in constructs:
            if c["line"] <= start:
                owner = c["name"]
            else:
                break
        regions.append(owner)
    return regions


def _unified_from_opcodes(a: list[str], b: list[str], context: int = 3):
    """Unified diff via SequenceMatcher.get_grouped_opcodes with autojunk off.

    difflib.unified_diff() does not expose autojunk; leaving it on can silently
    merge distant hunks on files with many repeated lines, which would corrupt
    the hunk-count statistic used later.
    """
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out: list[str] = []
    for group in sm.get_grouped_opcodes(context):
        first, last = group[0], group[-1]
        i1, j1 = first[1], first[3]
        i2, j2 = last[2], last[4]
        out.append(
            f"@@ -{i1 + 1},{i2 - i1} +{j1 + 1},{j2 - j1} @@"
        )
        for tag, x1, x2, y1, y2 in group:
            if tag == "equal":
                for k in range(x1, x2):
                    out.append(" " + a[k])
                continue
            if tag in ("delete", "replace"):
                for k in range(x1, x2):
                    out.append("-" + a[k])
            if tag in ("insert", "replace"):
                for k in range(y1, y2):
                    out.append("+" + b[k])
    return out


def diff_stats(a_text: str, b_text: str) -> dict:
    a = a_text.splitlines()
    b = b_text.splitlines()
    diff = list(_unified_from_opcodes(a, b))
    plus = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    minus = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    hunks = [l for l in diff if l.startswith("@@")]
    constructs = parent_constructs(a)
    regions = hunk_regions(hunks, constructs)
    # fingerprint: normalized changed-line content, order-insensitive.
    body = sorted(
        l[1:].strip()
        for l in diff
        if l[:1] in "+-" and l[:3] not in ("+++", "---") and l[1:].strip()
    )
    fp = hashlib.sha1("\n".join(body).encode()).hexdigest()[:12] if body else ""
    return {
        "diff_lines": len(diff),
        "hunks": len(hunks),
        "added": plus,
        "removed": minus,
        "changed_total": plus + minus,
        "hunk_regions": regions,
        "n_distinct_regions": len(set(regions)),
        "diff_fingerprint": fp,
        "diff_text": "\n".join(diff),
    }


def collect_candidates() -> list[dict]:
    edges: list[dict] = []
    runs = sorted(p for p in DATA.iterdir() if p.is_dir())
    for run in runs:
        level = "L2" if "l2" in run.name.lower() else "L3"
        for task in sorted(p for p in run.iterdir() if p.is_dir()):
            nodes = sorted(
                int(m.group(1))
                for m in (re.match(r"step_(\d+)_log\.json$", p.name)
                          for p in task.glob("step_*_log.json"))
                if m
            )
            cache: dict[int, dict | None] = {}

            def met(n: int):
                if n not in cache:
                    cache[n] = read_metrics(task, n)
                return cache[n]

            for n in nodes:
                log = read_log(task, n)
                if not log:
                    continue
                pid = log.get("parent_node_id")
                if pid is None:
                    continue
                pm, cm = met(pid), met(n)
                if not pm or not cm:
                    continue
                pr, cr = get_runtime(pm), get_runtime(cm)
                if pr is None or cr is None:
                    continue
                if cr >= pr * IMPROVE_THRESHOLD:
                    continue
                pf = Path(task / f"step_{pid}.py")
                cf = Path(task / f"step_{n}.py")
                if not (pf.exists() and cf.exists()):
                    continue
                a = pf.read_text(encoding="utf-8", errors="replace")
                b = cf.read_text(encoding="utf-8", errors="replace")
                if a == b:
                    continue
                st = diff_stats(a, b)
                if st["changed_total"] == 0:
                    continue
                edges.append(
                    {
                        "edge_id": f"{level}_{task.name}_{pid}_{n}",
                        "level": level,
                        "run": run.name,
                        "task_id": task.name,
                        "parent_node": pid,
                        "child_node": n,
                        "depth": log.get("depth"),
                        "created_by": log.get("created_by"),
                        "parent_runtime_ms": round(pr, 4),
                        "child_runtime_ms": round(cr, 4),
                        "speedup_vs_parent": round(pr / cr, 4),
                        "parent_fast_p": get_fast_p(pm),
                        "child_fast_p": get_fast_p(cm),
                        **{k: v for k, v in st.items() if k != "diff_text"},
                    }
                )
    return edges


def stratify(edges: list[dict]) -> list[dict]:
    """Sample TARGET_EDGES edges: cap per task, prefer large improvements, and
    keep a spread across hunk-count so that single-hunk and multi-location edges
    are both represented."""
    rng = random.Random(RNG_SEED)
    l3_tasks = sorted({e["task_id"] for e in edges if e["level"] == "L3"})
    rng.shuffle(l3_tasks)
    keep_l3 = set(l3_tasks[:L3_TASK_SAMPLE])

    pool = [e for e in edges if e["level"] == "L2" or e["task_id"] in keep_l3]
    by_task: dict[str, list[dict]] = {}
    for e in pool:
        by_task.setdefault(f'{e["level"]}/{e["task_id"]}', []).append(e)

    picked: list[dict] = []
    for key, lst in by_task.items():
        lst.sort(key=lambda e: -e["speedup_vs_parent"])
        picked.extend(lst[:MAX_EDGES_PER_TASK])

    # diversity pass: ensure coverage of 1-hunk vs multi-hunk edges
    picked.sort(key=lambda e: -e["speedup_vs_parent"])
    single = [e for e in picked if e["hunks"] == 1]
    multi = [e for e in picked if e["hunks"] > 1]
    rng.shuffle(single)
    rng.shuffle(multi)
    want_multi = min(len(multi), int(TARGET_EDGES * 0.6))
    sel = multi[:want_multi] + single[: TARGET_EDGES - want_multi]
    if len(sel) < TARGET_EDGES:
        rest = [e for e in picked if e not in sel]
        sel.extend(rest[: TARGET_EDGES - len(sel)])
    sel.sort(key=lambda e: (e["level"], e["task_id"], e["parent_node"]))
    return sel


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = collect_candidates()
    if not cands:
        print("no qualifying edges found", file=sys.stderr)
        return 1

    (OUT / "edge_index.json").write_text(
        json.dumps(cands, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sel = stratify(cands)
    for e in sel:
        tdir = DATA / e["run"] / e["task_id"]
        a = (tdir / f'step_{e["parent_node"]}.py').read_text(
            encoding="utf-8", errors="replace"
        )
        b = (tdir / f'step_{e["child_node"]}.py').read_text(
            encoding="utf-8", errors="replace"
        )
        st = diff_stats(a, b)
        (OUT / f'{e["edge_id"]}.diff').write_text(st["diff_text"], encoding="utf-8")

    (OUT / "selected_edges.json").write_text(
        json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lv = {}
    for e in cands:
        lv[e["level"]] = lv.get(e["level"], 0) + 1
    print(
        f"candidates={len(cands)} by_level={lv} "
        f"tasks_with_edges={len({(e['level'], e['task_id']) for e in cands})}"
    )
    print(
        f"selected={len(sel)} "
        f"L2={sum(1 for e in sel if e['level']=='L2')} "
        f"L3={sum(1 for e in sel if e['level']=='L3')} "
        f"multi_hunk={sum(1 for e in sel if e['hunks']>1)} "
        f"distinct_diff_fingerprints={len({e['diff_fingerprint'] for e in sel})}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
