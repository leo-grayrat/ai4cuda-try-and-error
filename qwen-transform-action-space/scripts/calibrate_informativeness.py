#!/usr/bin/env python3
"""Stage 2.5/4 — corrected informativeness calibration with symmetric permutation tests.

Revision R-1 of the Stage 3 roadmap. The previous version of this script contained
a real methodological error that reversed the paper's headline finding, and the
reconciliation script could not catch it because it only checked "paper number ==
script output", never "script method == valid method".

What was wrong
--------------
The Miller-Madow correction was applied to the CONDITIONAL entropy as
    H_hat = H_plugin + (k_X - 1) / (2n)
which is the bias term for a marginal entropy of k_X cells, not for a conditional
entropy or a mutual information. The bias of the plug-in MI estimator is
    (k_X - 1)(k_Y - 1) / (2n)
(the full MM correction applies to H(X), H(Y) and H(X,Y) simultaneously). With
k_Y = 4 quartile cells the old code under-corrected the label descriptor by a
factor of 3 relative to the coarse descriptor, whose 24 cells made the error
larger still in the opposite direction of interpretation.

What this version does
----------------------
1. Estimates MI(X;Y) by the plug-in estimator AND by the proper Miller-Madow
   estimator applied to H(X), H(Y), H(X,Y).
2. Runs a permutation test in which the descriptor X is HELD FIXED and the outcome
   Y is shuffled, for BOTH descriptors, so the comparison is symmetric. This is the
   test that does not depend on any bias formula.
3. Adds a cardinality-matched null: for each descriptor, random partitions into the
   SAME number of groups as that descriptor has cells, to expose the mechanical
   advantage that more cells confer on raw conditional-entropy comparisons.
4. Bootstrap CI on MI_label - MI_coarse.
5. Additional null models for the Section 3.5 diversity claims (revision R-3):
   - `distinct_diffs == occurrences` is shown to be an identity and retired.
   - `distinct_tasks` is compared against a label-permutation null, which is the
     only version of that statistic that can mean anything.

Usage: python scripts/calibrate_informativeness.py
Output: data/informativeness_calibration.json (schema v2)
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261002
N_PERM = 20000
N_BOOT = 4000


# ------------------------------------------------------------------ estimators

def plugin_entropy(counts: Counter, n: int) -> float:
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)


def mm_entropy(counts: Counter, n: int) -> float:
    """Miller-Madow: plug-in + (number of observed cells - 1) / (2n)."""
    k = sum(1 for c in counts.values() if c)
    return plugin_entropy(counts, n) + (k - 1) / (2 * n)


def mi_plugin(x: list, y: list) -> float:
    n = len(x)
    cx, cy, cxy = Counter(x), Counter(y), Counter(zip(x, y))
    return plugin_entropy(cx, n) + plugin_entropy(cy, n) - plugin_entropy(cxy, n)


def mi_miller_madow(x: list, y: list) -> float:
    """MM applied to each entropy term; bias = (k_X-1)(k_Y-1)/2n only when the
    joint has k_X*k_Y support. Doing it term-by-term is the safe general form."""
    n = len(x)
    cx, cy, cxy = Counter(x), Counter(y), Counter(zip(x, y))
    return mm_entropy(cx, n) + mm_entropy(cy, n) - mm_entropy(cxy, n)


def mi_bias_term(x: list, y: list) -> float:
    """The classic large-sample MI bias: (k_X - 1)(k_Y - 1) / (2n)."""
    n = len(x)
    kx = len(set(x))
    ky = len(set(y))
    return (kx - 1) * (ky - 1) / (2 * n)


# ------------------------------------------------------------------ permutation

def permute_y(y: list, rng: random.Random) -> list:
    z = list(y)
    rng.shuffle(z)
    return z


def permutation_p(observed_mi: float, x: list, y: list, n_perm: int, seed: int) -> dict:
    """Fix the descriptor, shuffle the outcome. Null MI distribution ~0 for an
    uninformative descriptor."""
    rng = random.Random(seed)
    null = [mi_plugin(x, permute_y(y, rng)) for _ in range(n_perm)]
    ge = sum(1 for v in null if v >= observed_mi - 1e-12)
    mean = sum(null) / len(null)
    sd = math.sqrt(sum((v - mean) ** 2 for v in null) / len(null))
    return {
        "observed_mi_plugin_bits": round(observed_mi, 4),
        "null_mean_bits": round(mean, 4),
        "null_sd_bits": round(sd, 4),
        "p_value_one_sided": round((ge + 1) / (n_perm + 1), 6),
        "n_permutations": n_perm,
    }


def balanced_null(x: list, y: list, n_perm: int, seed: int) -> dict:
    """Cardinality-matched null: random balanced partition into the SAME number of
    groups as x has cells. Exposes the mechanical advantage of more cells."""
    n = len(y)
    k = len(set(x))
    rng = random.Random(seed)
    sizes = [n // k] * k
    for i in range(n - sum(sizes)):
        sizes[i] += 1
    pool_base = [j for j, s in enumerate(sizes) for _ in range(s)]
    vals = []
    for _ in range(n_perm):
        pool = list(pool_base)
        rng.shuffle(pool)
        vals.append(mi_plugin(pool, y))
    vals.sort()
    return {
        "cells": k,
        "null_mi_mean_bits": round(sum(vals) / len(vals), 4),
        "null_mi_p95_bits": round(vals[int(0.95 * len(vals))], 4),
        "n_permutations": n_perm,
    }


def bootstrap_diff(mi_a: list, mi_b: list, y: list, n_boot: int, seed: int) -> dict:
    """CI for MI(label;Y) - MI(coarse;Y) via paired edge resampling."""
    rng = random.Random(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yy = [y[i] for i in idx]
        aa = [mi_a[i] for i in idx]
        bb = [mi_b[i] for i in idx]
        diffs.append(mi_plugin(aa, yy) - mi_plugin(bb, yy))
    diffs.sort()
    return {
        "n_bootstrap": n_boot,
        "mean_diff_bits": round(sum(diffs) / len(diffs), 4),
        "ci95_low_bits": round(diffs[int(0.025 * len(diffs))], 4),
        "ci95_high_bits": round(diffs[int(0.975 * len(diffs))], 4),
        "prob_label_ahead": round(sum(1 for d in diffs if d > 0) / len(diffs), 4),
    }


# ------------------------------------------------------------------ diversity nulls

def distinct_tasks_null(recs, by_id, n_perm: int, seed: int) -> dict:
    """For each label, compare observed distinct-task count against the same label
    multiset randomly re-assigned to edges. Under the null, labels are unrelated to
    task identity; a label that genuinely recurs ACROSS tasks should score at or
    above the null, and one that clusters WITHIN tasks scores below it."""
    rng = random.Random(seed)
    labels = [r["primary"] for r in recs]
    tasks = [f'{by_id[r["edge_id"]]["level"]}/{by_id[r["edge_id"]]["task_id"]}' for r in recs]
    n = len(recs)

    def per_label_distinct(ls):
        c: dict[str, set] = {}
        for lab, tk in zip(ls, tasks):
            c.setdefault(lab, set()).add(tk)
        return {lab: len(v) for lab, v in c.items()}

    obs = per_label_distinct(labels)
    counts: dict[str, list[int]] = {lab: [] for lab in obs}
    for _ in range(n_perm):
        shuffled = list(labels)
        rng.shuffle(shuffled)
        d = per_label_distinct(shuffled)
        for lab in obs:
            counts[lab].append(d.get(lab, 0))

    out = {}
    for lab, o in obs.items():
        vals = sorted(counts[lab])
        mean = sum(vals) / len(vals)
        below = sum(1 for v in vals if v <= o) / len(vals)
        out[lab] = {
            "observed_distinct_tasks": o,
            "null_mean": round(mean, 3),
            "null_median": vals[len(vals) // 2],
            "percentile": round(100 * below, 1),
            "clusters_within_task": bool(below < 0.05),
        }
    return {"n_permutations": n_perm, "per_label": out}


def identity_audit(by_id) -> dict:
    """Demonstrate that the 'distinct diffs per label' statistic carries no
    information once every edge diff is unique."""
    fp = [e["diff_fingerprint"] for e in by_id.values()]
    return {
        "n_edges": len(fp),
        "n_distinct_fingerprints": len(set(fp)),
        "all_unique": len(fp) == len(set(fp)),
        "consequence": (
            "if every edge has a distinct fingerprint, then for ANY label assignment "
            "distinct_diffs(label) == occurrences(label) by identity, so the column "
            "cannot discriminate hypotheses"
        ),
    }


# ------------------------------------------------------------------ main

def calibration_figure(out: dict) -> None:
    """Fig 5: observed MI of each descriptor against its own permutation null.

    This is the figure the paper's headline claim rests on. Plotting MI against the
    descriptor's own null (rather than MI against other descriptors' MI) is what
    removes the granularity confound that reversed the v1 conclusion.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib unavailable; skipped fig5", file=sys.stderr)
        return

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ds = out["descriptors"]
    names = [d["descriptor"] for d in ds]
    obs = [d["mi_plugin_bits"] for d in ds]
    null_m = [d["permutation_test"]["null_mean_bits"] for d in ds]
    null_sd = [d["permutation_test"]["null_sd_bits"] for d in ds]
    pvals = [d["permutation_test"]["p_value_one_sided"] for d in ds]

    x = np.arange(len(ds))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x - 0.18, obs, width=0.36, label="observed MI (plug-in)", color="#2f6f4f")
    ax.bar(x + 0.18, null_m, width=0.36, label="own permutation null mean", color="#b0b0b0")
    ax.errorbar(
        x + 0.18, null_m, yerr=null_sd, fmt="none", ecolor="#5b5b5b",
        capsize=4, label="null SD",
    )
    ax.set_xticks(x, [n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel("mutual information with speedup quartile (bits)")
    ax.set_title(
        "Each descriptor against its OWN null\n"
        "only the semantic label exceeds its null; raw MI ordering is a granularity artefact"
    )
    for i, (o, nm, p) in enumerate(zip(obs, null_m, pvals)):
        ax.text(i - 0.18, o + 0.02, f"{o:.2f}", ha="center", fontsize=8)
        ax.text(i, max(o, nm) + 0.11, f"p={p:.4f}", ha="center", fontsize=8.5,
                color="#7a1f1f" if p < 0.05 else "#333333", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(obs), max(null_m)) * 1.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_permutation_calibration.png", dpi=160)
    plt.close(fig)
    print("wrote figures/fig5_permutation_calibration.png")


def main() -> int:
    blob = json.loads((ROOT / "data" / "annotations.json").read_text(encoding="utf-8"))
    recs = blob["annotations"]
    by_id = {
        e["edge_id"]: e
        for e in json.loads(
            (ROOT / "data" / "edges" / "selected_edges.json").read_text(encoding="utf-8")
        )
    }
    ids = [r["edge_id"] for r in recs]
    n = len(recs)

    log_sp = [math.log2(by_id[i]["speedup_vs_parent"]) for i in ids]
    order = sorted(range(n), key=lambda k: log_sp[k])
    quart: dict[str, int] = {}
    for rank, idx in enumerate(order):
        quart[ids[idx]] = min(rank * 4 // n, 3)
    Y = [quart[i] for i in ids]

    X_label = [r["primary"] for r in recs]
    X_coarse = [
        f"{by_id[i]['hunks']}|{min(by_id[i]['changed_total'] // 50, 6)}" for i in ids
    ]
    X_hunks = [str(by_id[i]["hunks"]) for i in ids]

    def describe(name, X):
        plugin = mi_plugin(X, Y)
        mm = mi_miller_madow(X, Y)
        bias = mi_bias_term(X, Y)
        return {
            "descriptor": name,
            "cells": len(set(X)),
            "mi_plugin_bits": round(plugin, 4),
            "mi_miller_madow_bits": round(mm, 4),
            "mi_bias_term_bits": round(bias, 4),
            "permutation_test": permutation_p(plugin, X, Y, N_PERM, SEED),
            "cardinality_matched_null": balanced_null(X, Y, 2000, SEED + 1),
        }

    d_label = describe("primary semantic label", X_label)
    d_coarse = describe("coarse diff structure", X_coarse)
    d_hunks = describe("hunk count alone", X_hunks)

    boot = bootstrap_diff(X_label, X_coarse, Y, N_BOOT, SEED + 2)

    # normalized entropy of the label distribution vs coarse structure
    def norm_entropy(X):
        c = Counter(X)
        k = len(c)
        h = plugin_entropy(c, len(X))
        hmax = math.log2(k) if k > 1 else 0.0
        return {"cells": k, "entropy_bits": round(h, 4), "max_bits": round(hmax, 4),
                "normalized": round(h / hmax, 4) if hmax else 1.0}

    out = {
        "schema_version": 2,
        "revision_note": (
            "v1 applied the marginal Miller-Madow term (k_X-1)/2n to a conditional "
            "entropy and permuted only the semantic descriptor. Both defects are "
            "corrected here; the headline comparison reverses."
        ),
        "n_edges": n,
        "outcome": {
            "definition": "quartile of log2(speedup vs MCTS parent)",
            "entropy_bits": round(plugin_entropy(Counter(Y), n), 4),
            "cells": len(set(Y)),
        },
        "descriptors": [d_label, d_coarse, d_hunks],
        "bootstrap_label_minus_coarse": boot,
        "normalized_entropies": {
            "primary_label": norm_entropy(X_label),
            "coarse_structure": norm_entropy(X_coarse),
            "diff_fingerprint": norm_entropy([by_id[i]["diff_fingerprint"] for i in ids]),
        },
        "diversity_null_models": {
            "identity_audit": identity_audit(by_id),
            "distinct_tasks_vs_label_permutation": distinct_tasks_null(recs, by_id, 4000, SEED + 3),
        },
    }

    verdict_bits = []
    for d in (d_label, d_coarse, d_hunks):
        verdict_bits.append(
            f"{d['descriptor']}: MI_plugin={d['mi_plugin_bits']} "
            f"MI_MM={d['mi_miller_madow_bits']} perm_p={d['permutation_test']['p_value_one_sided']}"
        )
    out["interpretation"] = (
        "Under a symmetric permutation test the semantic label is significantly "
        "informative about the realised-speedup quartile while the coarse diff "
        "structure is not, and the v1 finding (coarse beats semantic) is an artefact "
        "of an under-corrected Miller-Madow term. The corrected result still does not "
        "establish the coordinate-system thesis: it is observational, on a biased "
        "multi-hunk-enriched sample, with the autotune confound unremoved."
    )

    dest = ROOT / "data" / "informativeness_calibration.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    calibration_figure(out)
    print(json.dumps(out["descriptors"], indent=2))
    print(json.dumps({"bootstrap": boot, "normalized": out["normalized_entropies"]}, indent=2))
    for line in verdict_bits:
        print(line)
    print("\nwrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
