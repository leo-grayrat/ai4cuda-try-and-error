"""Figures for the paper.

Runs under the anaconda interpreter (that is where matplotlib lives; the
workbuddy python has no plotting stack). Reads runs-formal/*/turns.jsonl
directly so that every figure is computed from the raw logprobs rather than
from a cached aggregate.

Output: figures/fig{1..5}_*.png at 200 dpi.
"""

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOWPROB = -2.0
ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"

# Same vocabularies as analyze_loop.py, so figures and tables cannot disagree.
TRITON_API = {
    "tl", "load", "store", "dot", "sum", "max", "min", "reduce", "arange",
    "zeros", "full", "expand_dims", "broadcast_to", "where", "view", "reshape",
    "trans", "ravel", "atomic_add", "atomic_cas", "atomic_max", "atomic_min",
    "cast", "to", "num_warps", "num_stages", "constexpr", "program_id",
}
STRICT = {"tl", "num_warps", "num_stages", "constexpr", "program_id"} | {
    "threadidx", "blockidx", "blockdim", "griddim", "__shared__", "__syncthreads",
    "__global__", "__device__", "atomicadd", "warpsize", "__shfl_sync",
    "__ballot_sync", "smem", "mma", "wmma",
}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans",
})


def norm(t: str) -> str:
    return t.lstrip(".@#`").strip().lower()


def load() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "runs-formal" / "*" / "turns.jsonl"))):
        run = Path(path).parent.name
        task = run.rsplit("-", 2)[0]
        condition = "truncated" if "-truncated-" in run else "full"
        for line in open(path, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn = rec.get("turn", 0)
            ctx = ("initial" if turn == 0 else
                   ("after_success" if rec.get("result", {}).get("correct")
                    else "after_failure"))
            for e in rec.get("content_logprobs") or []:
                if e.get("logprob") is None:
                    continue
                rows.append({"tok": e.get("token") or "", "lp": e["logprob"],
                             "task": task, "cond": condition, "ctx": ctx})
    return rows


def rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return 100 * sum(1 for r in rows if r["lp"] <= LOWPROB) / len(rows)


def fig1(rows):
    loose_in = [r for r in rows if norm(r["tok"]) in TRITON_API]
    loose_out = [r for r in rows if norm(r["tok"]) not in TRITON_API]
    strict_in = [r for r in rows if norm(r["tok"]) in STRICT]
    strict_out = [r for r in rows if norm(r["tok"]) not in STRICT]

    labels = ["loose vocabulary", "strict vocabulary"]
    inside = [rate(loose_in), rate(strict_in)]
    outside = [rate(loose_out), rate(strict_out)]

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, inside, w, label="CUDA/Triton tokens", color="#c1443f")
    b2 = ax.bar(x + w / 2, outside, w, label="all other tokens", color="#5b7fa6")
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.2f}%",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("low-probability rate (logprob ≤ −2)")
    ax.set_title("CUDA vocabulary is less likely to be low-probability", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, max(outside + inside) * 1.35)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_lowprob_by_vocabulary.png")
    plt.close(fig)


def fig2(rows):
    order = ["initial", "after_success", "after_failure"]
    vals = [rate([r for r in rows if r["ctx"] == c]) for c in order]
    ns = [len([r for r in rows if r["ctx"] == c]) for c in order]
    colors = ["#7a8fa6", "#6a9a6a", "#c1443f"]

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    bars = ax.bar(order, vals, color=colors, width=0.55)
    for rect, v, n in zip(bars, vals, ns):
        ax.annotate(f"{v:.2f}%\nn={n:,}", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("low-probability rate (logprob ≤ −2)")
    ax.set_title("Context dominates: failure feedback raises the rate 4.6×", fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.45)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_lowprob_by_context.png")
    plt.close(fig)


def fig3(rows):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for label, sel, colour in (
        ("CUDA/Triton tokens", [r for r in rows if norm(r["tok"]) in STRICT], "#c1443f"),
        ("all other tokens", [r for r in rows if norm(r["tok"]) not in STRICT], "#5b7fa6"),
    ):
        v = np.sort(np.array([r["lp"] for r in sel]))
        cdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, cdf, label=f"{label} (n={len(v):,})", color=colour, lw=1.4)
    ax.axvline(LOWPROB, color="#888", ls=":", lw=1)
    ax.annotate("threshold −2", (LOWPROB, 0.5), fontsize=7.5, color="#666",
                xytext=(4, 0), textcoords="offset points")
    ax.set_xlim(-6, 0.2)
    ax.set_xlabel("token logprob")
    ax.set_ylabel("cumulative share of tokens")
    ax.set_title("CUDA tokens sit further right, i.e. more confident", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_logprob_cdf.png")
    plt.close(fig)


def fig4(rows):
    tasks = sorted({r["task"] for r in rows})
    vals = [rate([r for r in rows if r["task"] == t]) for t in tasks]
    ns = [len([r for r in rows if r["task"] == t]) for t in tasks]
    order = np.argsort(vals)

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    bars = ax.barh([tasks[i] for i in order], [vals[i] for i in order],
                   color="#7a8fa6", height=0.55)
    for rect, i in zip(bars, order):
        ax.annotate(f"{vals[i]:.2f}%  (n={ns[i]:,})",
                    (rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    va="center", fontsize=8, xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel("low-probability rate (logprob ≤ −2)")
    ax.set_title("Per task: softmax_rows carries most of the tail", fontsize=9.5)
    ax.set_xlim(0, max(vals) * 1.4)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_lowprob_by_task.png")
    plt.close(fig)


def fig5(rows):
    conds = ["full", "truncated"]
    vals = [rate([r for r in rows if r["cond"] == c]) for c in conds]
    ns = [len([r for r in rows if r["cond"] == c]) for c in conds]

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    bars = ax.bar(["full feedback", "truncated feedback"], vals, color="#8a8fa6", width=0.5)
    for rect, v, n in zip(bars, vals, ns):
        ax.annotate(f"{v:.2f}%\nn={n:,}", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("low-probability rate (logprob ≤ −2)")
    ax.set_title("Compressing feedback does not shrink the tail", fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.6)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_intervention.png")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    rows = load()
    print(f"tokens: {len(rows):,}")
    fig1(rows)
    fig2(rows)
    fig3(rows)
    fig4(rows)
    fig5(rows)
    for p in sorted(FIGDIR.glob("*.png")):
        print(f"  {p.name}  {p.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
