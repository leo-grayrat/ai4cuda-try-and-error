#!/usr/bin/env python3
"""Stage 1.5 step 4: merge AI annotations and compute vocabulary statistics.

Computes, for the 55 annotated successful edges:
  - vocabulary reuse rate |V|/|E| and label coverage
  - Shannon entropy of the transformation-label distribution
  - Shannon entropy of diff-structure descriptors (hunk count, changed lines,
    and the per-edge diff fingerprint) -> the "compression" comparison
  - within-label diff diversity (many-to-many evidence)
  - cross-task label recurrence (transferability evidence, within-corpus)
  - red-line checks: any class > 40%? Other > 25%?

Outputs data/annotations.json and figures/*.png.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGES = ROOT / "data" / "edges"
FIG = ROOT / "figures"

VOCAB = [
    "Fuse", "EliminateMaterialization", "Tile", "IncreaseWorkPerThread",
    "Vectorize", "ChangeLaunchConfig", "WarpReduce", "SharedMemoryTransform",
    "PrecisionChange", "LoopTransform", "HostDeviceOverlap", "Other",
]


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h


def load_annotations() -> tuple[list[dict], dict[str, dict]]:
    recs: list[dict] = []
    for k in range(5):
        p = ROOT / "data" / f"annotation_batch_{k}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
    sel = json.loads((EDGES / "selected_edges.json").read_text(encoding="utf-8"))
    by_id = {e["edge_id"]: e for e in sel}
    ids = [r["edge_id"] for r in recs]
    assert len(recs) == len(sel), f"annotation count {len(recs)} != edges {len(sel)}"
    assert set(ids) == set(by_id), "edge_id mismatch between annotations and index"
    for r in recs:
        assert r["primary"] in VOCAB, f"bad primary {r['primary']}"
        assert set(r.get("secondary", [])) <= set(VOCAB)
        assert set(r["sites"]) <= set(by_id[r["edge_id"]]["hunk_regions"]), (
            f"sites not subset of regions for {r['edge_id']}"
        )
        assert r["spans_multiple_loci"] == (len(r["sites"]) >= 2), (
            f"spans_multiple_loci inconsistent for {r['edge_id']}"
        )
        assert r.get("reasoning", "").strip(), f"empty reasoning {r['edge_id']}"
    return recs, by_id


def analyse(recs: list[dict], by_id: dict[str, dict]) -> dict:
    n = len(recs)
    primary = Counter(r["primary"] for r in recs)
    all_labels = Counter()
    for r in recs:
        for lab in {r["primary"], *r.get("secondary", [])}:
            all_labels[lab] += 1

    prim_counts = Counter({k: v for k, v in primary.items() if v})
    used = [v for v in VOCAB if primary.get(v)]
    h_primary = entropy(prim_counts)
    h_max_possible = math.log2(n)

    all_c = Counter({k: v for k, v in all_labels.items() if v})
    h_all = entropy(all_c)
    tokens = sum(all_labels.values())
    h_all_max = math.log2(tokens) if tokens > 1 else 0.0

    # ---- diff-structure descriptors -------------------------------------
    hunk_c = Counter(by_id[r["edge_id"]]["hunks"] for r in recs)
    chg_bucket = Counter(
        min(int(by_id[r["edge_id"]]["changed_total"] // 50), 6) for r in recs
    )
    fp_c = Counter(by_id[r["edge_id"]]["diff_fingerprint"] for r in recs)
    h_hunks = entropy(hunk_c)
    h_chg = entropy(chg_bucket)
    h_fp = entropy(fp_c)

    # ---- many-to-many: within-label diff diversity ----------------------
    by_label: dict[str, set] = defaultdict(set)
    by_label_tasks: dict[str, set] = defaultdict(set)
    by_label_regionsets: dict[str, set] = defaultdict(set)
    for r in recs:
        e = by_id[r["edge_id"]]
        for lab in {r["primary"], *r.get("secondary", [])}:
            by_label[lab].add(e["diff_fingerprint"])
            by_label_tasks[lab].add(f'{e["level"]}/{e["task_id"]}')
            by_label_regionsets[lab].add(tuple(sorted(set(e["hunk_regions"]))))

    diversity = {
        lab: {
            "occurrences": all_labels[lab],
            "distinct_diffs": len(v),
            "distinct_tasks": len(by_label_tasks[lab]),
            "distinct_region_sets": len(by_label_regionsets[lab]),
        }
        for lab, v in by_label.items()
    }

    multi = sum(1 for r in recs if r["spans_multiple_loci"])
    conf = Counter(r.get("confidence", "medium") for r in recs)
    autotuner = sum(1 for r in recs if "autotun" in (r.get("notes", "") or "").lower())
    suspicious = sum(
        1 for r in recs if "suspicious_parent_speedup" in (r.get("notes", "") or "")
    )

    other_share = primary["Other"] / n
    top_lab, top_share = primary.most_common(1)[0] if primary else (None, 0.0)
    top_share = top_share / n if n else 0.0

    return {
        "n_edges": n,
        "n_tasks": len({f'{e["level"]}/{e["task_id"]}' for e in by_id.values()}),
        "vocabulary_size": len(VOCAB),
        "vocabulary_used": len(used),
        "reuse_rate_V_over_E": round(len(used) / n, 4),
        "coverage_pct": round(100 * (1 - other_share), 1),
        "entropy_primary_bits": round(h_primary, 4),
        "entropy_primary_max_bits": round(h_max_possible, 4),
        "entropy_all_labels_bits": round(h_all, 4),
        "entropy_all_labels_max_bits": round(h_all_max, 4),
        "entropy_diff_fingerprint_bits": round(h_fp, 4),
        "entropy_hunkcount_bits": round(h_hunks, 4),
        "entropy_changedline_bucket_bits": round(h_chg, 4),
        "distinct_diff_fingerprints": len(fp_c),
        "label_counts_primary": dict(primary),
        "label_counts_any": dict(all_labels),
        "redline_class_over_40pct": bool(top_share > 0.40),
        "redline_top_class": top_lab,
        "redline_top_share": round(top_share, 4),
        "redline_other_over_25pct": bool(other_share > 0.25),
        "redline_other_share": round(other_share, 4),
        "edges_spanning_multiple_loci": multi,
        "edges_spanning_multiple_loci_pct": round(100 * multi / n, 1),
        "confidence_dist": dict(conf),
        "edges_with_autotuner_change": autotuner,
        "edges_with_suspicious_parent_speedup": suspicious,
        "within_label_diversity": diversity,
        "caveat_within_label_diversity": (
            "distinct_diffs equals occurrences by identity whenever every edge diff "
            "is unique; see data/informativeness_calibration.json identity_audit. The "
            "distinct_tasks column must be read against the label-permutation null in "
            "that file, not as raw evidence of cross-task reuse."
        ),
        # Outcome-informativeness is deliberately NOT computed here. It is owned by
        # calibrate_informativeness.py, which estimates mutual information with a
        # correct bias term and calibrates every descriptor against its own null.
        # An earlier version of this file duplicated the quartile binning here and
        # produced a headline conclusion that did not survive review.
    }


def figures(recs: list[dict], by_id: dict[str, dict], stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG.mkdir(parents=True, exist_ok=True)

    # Fig 1: description-length comparison, with the identity explicitly marked
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    names = [
        "transformation\nlabel (primary)",
        "transformation\nlabel (any)",
        "diff hunk count",
        "changed-line\nbucket",
        "coarse structure\n(hunks x lines)",
        "diff fingerprint\n(IDENTITY)",
    ]
    coarse_of = Counter(
        f"{by_id[r['edge_id']]['hunks']}|{min(by_id[r['edge_id']]['changed_total'] // 50, 6)}"
        for r in recs
    )
    vals = [
        stats["entropy_primary_bits"],
        stats["entropy_all_labels_bits"],
        stats["entropy_hunkcount_bits"],
        stats["entropy_changedline_bucket_bits"],
        entropy(coarse_of),
        stats["entropy_diff_fingerprint_bits"],
    ]
    colors = ["#2f6f4f", "#4a8f6a", "#9a9a9a", "#b0b0b0", "#7f7f7f", "#c9564a"]
    ax.bar(names, vals, color=colors)
    ax.patches[len(names) - 1].set_hatch("//")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title(
        "Description length per edge\n"
        f"n={stats['n_edges']} edges, {stats['n_tasks']} tasks; "
        "the rightmost bar is identically log2(n) and is NOT evidence"
    )
    ax.tick_params(axis="x", labelsize=7.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_entropy_comparison.png", dpi=160)
    plt.close(fig)

    # Fig 2: label distribution
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labs = [v for v in VOCAB if stats["label_counts_any"].get(v)]
    prim = [stats["label_counts_primary"].get(l, 0) for l in labs]
    sec = [stats["label_counts_any"].get(l, 0) - p for l, p in zip(labs, prim)]
    y = np.arange(len(labs))
    ax.barh(y, prim, color="#2f6f4f", label="primary")
    ax.barh(y, sec, left=prim, color="#9fc7ae", label="secondary only")
    ax.set_yticks(y, labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("edge count")
    ax.axvline(0.40 * stats["n_edges"], ls="--", c="crimson", lw=1,
               label="40% red line")
    ax.legend(fontsize=8)
    ax.set_title("Frozen 12-class vocabulary: observed distribution")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_label_distribution.png", dpi=160)
    plt.close(fig)

    # Fig 3: many-to-many evidence
    fig, ax = plt.subplots(figsize=(7, 4.2))
    d = stats["within_label_diversity"]
    labs2 = sorted(d, key=lambda l: -d[l]["occurrences"])
    occ = [d[l]["occurrences"] for l in labs2]
    dd = [d[l]["distinct_diffs"] for l in labs2]
    x = np.arange(len(labs2))
    ax.bar(x - 0.2, occ, width=0.4, label="edges carrying label", color="#31688f")
    ax.bar(x + 0.2, dd, width=0.4, label="distinct diff fingerprints", color="#c0843a")
    ax.set_xticks(x, labs2, fontsize=7, rotation=40, ha="right")
    ax.legend(fontsize=8)
    ax.set_title(
        "One transformation -> many distinct diffs "
        f"(overall {stats['distinct_diff_fingerprints']}/{stats['n_edges']} unique)"
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig3_many_to_many.png", dpi=160)
    plt.close(fig)

    # Fig 4: speedup vs label
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bylab: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        bylab[r["primary"]].append(by_id[r["edge_id"]]["speedup_vs_parent"])
    ks = sorted(bylab, key=lambda k: -max(bylab[k]))
    ax.boxplot([bylab[k] for k in ks], labels=ks, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("speedup vs parent (log)")
    ax.tick_params(axis="x", labelsize=7, rotation=40)
    ax.set_title("Realised speedup per primary transformation")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_speedup_by_label.png", dpi=160)
    plt.close(fig)


def main() -> int:
    recs, by_id = load_annotations()
    stats = analyse(recs, by_id)
    merged = ROOT / "data" / "annotations.json"
    merged.write_text(
        json.dumps(
            {
                "meta": {
                    "annotator": "AI (Qwen session), batched subagents",
                    "protocol": "ANNOTATION_GUIDE.md + VOCABULARY_FROZEN.md",
                    "vocabulary_frozen_before_any_diff_read": True,
                    "edge_selection": "MCTS parent_node_id, both correct, >5% faster",
                    "n_candidate_edges_in_corpus": 2510,
                },
                "statistics": stats,
                "annotations": recs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    figures(recs, by_id, stats)

    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("within_label_diversity",)}, indent=2))
    print("\n=== within-label diversity ===")
    for lab, d in sorted(
        stats["within_label_diversity"].items(), key=lambda x: -x[1]["occurrences"]
    ):
        print(f"{lab:<24} occ={d['occurrences']:>3} distinct_diffs={d['distinct_diffs']:>3} "
              f"tasks={d['distinct_tasks']:>3} region_sets={d['distinct_region_sets']}")
    print(f"\nwrote {merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
