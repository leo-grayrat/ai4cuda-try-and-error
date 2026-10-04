#!/usr/bin/env python3
"""Section 7.1 controls: annotation agreement and does the headline survive blind labels.

Two of the four acceptance criteria stated in Section 7.1 are executable by this
pipeline and are implemented here:

  criterion 4  labels produced without the vocabulary's definitions, examples or
               parameter slots, so category semantics are not shared by construction
  criterion 3  the Section 3.6 permutation result recomputed on the labels from (4),
               and on the subset where the two passes agree

A third control, bottom-up open coding of a task-disjoint probe set
(data/opencode_result.md), is qualitative and mapped by hand.

Independent human annotation with a second inter-rater coefficient (criterion 1's
strong form, and the one genuinely outside this pipeline's means) is still not done
and must remain an acknowledged limitation.

Outputs data/annotation_agreement.json:
  - raw agreement and Cohen's kappa between the original and blind label sets
  - category distributions for both
  - MI(label; speedup quartile) and a permutation p-value for each label set, so the
    headline claim can be read against the possibility that it only reflects the
    vocabulary the same model also wrote

Usage: python scripts/annotation_agreement.py
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261004
N_PERM = 20000
VOCAB = [
    "Fuse", "EliminateMaterialization", "Tile", "IncreaseWorkPerThread", "Vectorize",
    "ChangeLaunchConfig", "WarpReduce", "SharedMemoryTransform", "PrecisionChange",
    "LoopTransform", "HostDeviceOverlap", "Other",
]


def plugin_entropy(c: Counter, n: int) -> float:
    return -sum((v / n) * math.log2(v / n) for v in c.values() if v)


def mi(x: list, y: list) -> float:
    n = len(x)
    return (plugin_entropy(Counter(x), n) + plugin_entropy(Counter(y), n)
            - plugin_entropy(Counter(zip(x, y)), n))


def perm_p(x: list, y: list, seed: int) -> dict:
    rng = random.Random(seed)
    obs = mi(x, y)
    null = []
    for _ in range(N_PERM):
        z = list(y)
        rng.shuffle(z)
        null.append(mi(x, z))
    ge = sum(1 for v in null if v >= obs - 1e-12)
    mean = sum(null) / len(null)
    return {
        "mi_plugin_bits": round(obs, 4),
        "null_mean_bits": round(mean, 4),
        "p_value_one_sided": round((ge + 1) / (N_PERM + 1), 6),
        "n_permutations": N_PERM,
    }


def cohen_kappa(a: list, b: list) -> dict:
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return {
        "n": n,
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
        "cohens_kappa": round((po - pe) / (1 - pe), 4) if pe < 1 else None,
    }


def main() -> int:
    blob = json.loads((ROOT / "data" / "annotations.json").read_text(encoding="utf-8"))
    original = {r["edge_id"]: r["primary"] for r in blob["annotations"]}
    by_id = {
        e["edge_id"]: e
        for e in json.loads(
            (ROOT / "data" / "edges" / "selected_edges.json").read_text(encoding="utf-8")
        )
    }
    blind = {}
    for line in (ROOT / "data" / "relabel_blind.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            assert rec["primary"] in VOCAB, rec
            blind[rec["edge_id"]] = rec["primary"]

    ids = [i for i in original if i in blind]
    if len(ids) != len(original):
        print(f"WARNING: only {len(ids)} of {len(original)} edges have blind labels")

    A = [original[i] for i in ids]
    B = [blind[i] for i in ids]
    Y = [0] * len(ids)
    # outcome: quartile of log2 speedup versus parent, as in Section 3.6
    sp = [math.log2(by_id[i]["speedup_vs_parent"]) for i in ids]
    order = sorted(range(len(ids)), key=lambda k: sp[k])
    q = {}
    for rank, idx in enumerate(order):
        q[idx] = min(rank * 4 // len(ids), 3)
    Y = [q[k] for k in range(len(ids))]

    agree_ids = [k for k, (x, y) in enumerate(zip(A, B)) if x == y]
    C = [A[k] for k in agree_ids]
    Yc = [Y[k] for k in agree_ids]

    def drop_constant(label_seq, outcome_seq):
        keep = [i for i, lab in enumerate(label_seq)
                if Counter(label_seq)[lab] > 1]
        return [label_seq[i] for i in keep], [outcome_seq[i] for i in keep]

    cons, cons_y = drop_constant(C, Yc)

    out = {
        "agreement": cohen_kappa(A, B),
        "distribution_original": dict(Counter(A)),
        "distribution_blind": dict(Counter(B)),
        "categories_unused_by_blind": [c for c in VOCAB if Counter(B).get(c, 0) == 0],
        "headline_survival": {
            "original_labels": perm_p(A, Y, SEED),
            "blind_labels_no_definitions": perm_p(B, Y, SEED + 1),
            "consensus_subset": {
                "n_edges": len(cons),
                **perm_p(cons, cons_y, SEED + 2),
            },
        },
        "interpretation": "",
    }

    p_blind = out["headline_survival"]["blind_labels_no_definitions"]["p_value_one_sided"]
    p_orig = out["headline_survival"]["original_labels"]["p_value_one_sided"]
    if p_blind < 0.05 and p_orig < 0.05:
        interp = ("The Section 3.6 result survives on labels produced without the "
                  "vocabulary's definitions: both passes are individually significant. "
                  "This weakens the circularity objection but does not settle it, since "
                  "both passes share a model family and inter-rater agreement is only "
                  "moderate/low depending on category.")
    elif p_orig < 0.05:
        interp = ("The result does NOT survive the definition-free relabelling: the "
                  "original significance may reflect the vocabulary's own guidance "
                  "rather than structure in the data. Section 3.6 must be downgraded.")
    else:
        interp = "Neither label set is informative; the original claim is unsupported."
    out["interpretation"] = interp

    dest = ROOT / "data" / "annotation_agreement.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("agreement", "distribution_blind",
                                          "categories_unused_by_blind")}, indent=2))
    print(json.dumps(out["headline_survival"], indent=2))
    print("\n", interp)
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
