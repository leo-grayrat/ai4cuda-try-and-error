"""Render the v2 figures directly from saved response-token records."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_loop import STRICT, TRITON_API, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v2/figures"
BLUE, ORANGE, GREEN = "#484878", "#B77B8D", "#A8B4CF"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "xtick.major.width": .65, "ytick.major.width": .65,
    "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False, "savefig.dpi": 600,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def collect(directory: str) -> dict:
    values = collections.defaultdict(list)
    by_task = collections.defaultdict(lambda: collections.defaultdict(list))
    contexts = collections.defaultdict(list)
    vocabulary = {"strict": [[], []], "loose": [[], []]}
    sources = []
    for path in sorted((ROOT / directory).glob("*/turns.jsonl")):
        task = path.parent.name.rsplit("-", 2)[0]
        raw = path.read_bytes()
        sources.append({"path": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest()})
        for line in raw.decode("utf-8").splitlines():
            record = json.loads(line)
            context = "initial" if record.get("turn", 0) == 0 else (
                "after_success" if record.get("result", {}).get("correct") else "after_failure")
            in_code = False
            for entry in record.get("content_logprobs") or []:
                token = entry.get("token") or ""
                if "```" in token:
                    in_code = not in_code
                lp = entry.get("logprob")
                if lp is None:
                    continue
                region = "code" if in_code else "prose"
                values[region].append(lp)
                by_task[task][region].append(lp)
                contexts[context].append(lp)
                for name, vocab in (("strict", STRICT), ("loose", TRITON_API)):
                    vocabulary[name][0 if norm(token) in vocab else 1].append(lp)
    return {"regions": {k: np.array(v) for k, v in values.items()},
            "tasks": {k: {r: np.array(v) for r, v in d.items()} for k, d in by_task.items()},
            "contexts": {k: np.array(v) for k, v in contexts.items()},
            "vocabulary": {k: [np.array(v) for v in pair] for k, pair in vocabulary.items()},
            "sources": sources}


def rate(values):
    return float(np.mean(values <= -2) * 100)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    svg_path = OUT / f"{name}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
                        + "\n", encoding="utf-8")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=10, color="#333333")
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax.grid(axis="y", color="#e8e8ed", linewidth=.45)
    ax.set_axisbelow(True)


def cdf(ax, values, label, color, style="-"):
    ordered = np.sort(values)
    # Every point is an empirical order statistic; no smoothing/interpolation.
    share = np.arange(1, len(values)+1) / len(values) * 100
    if np.any(share <= 0):
        raise ValueError("Empirical CDF values must be positive on the log axis")
    ax.step(ordered, share, where="post", color=color, linestyle=style,
            label=f"{label} (n={len(values):,})", linewidth=1.4)


def main():
    OUT.mkdir(exist_ok=True)
    a, b = collect("runs-formal"), collect("runs-kb2")
    assert sum(map(len, a["regions"].values())) == 98533
    assert [len(b["regions"][r]) for r in ("code", "prose")] == [322375, 220236]
    assert [len(b["contexts"][r]) for r in ("initial", "after_success", "after_failure")] == [55736, 12147, 474728]
    assert round(rate(b["regions"]["code"]), 2) == .60
    assert round(rate(b["regions"]["prose"]), 2) == 13.56
    assert round(float(b["regions"]["code"].min()), 3) == -13.016

    fig, axes = plt.subplots(1, 2, figsize=(7.204724, 3.0), width_ratios=[.9, 1.25], layout="constrained")
    ax = axes[0]
    x, width = np.arange(2), .33
    for j, (label, color, hatch) in enumerate((("CUDA/Triton", BLUE, ""), ("Other tokens", ORANGE, "//"))):
        y = [rate(a["vocabulary"][name][j]) for name in ("strict", "loose")]
        bars = ax.bar(x + (j-.5)*width, y, width, color=color, hatch=hatch, label=label,
                      edgecolor=color, linewidth=.6, alpha=.85)
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in y], padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=["Strict", "Loose"], ylim=(0, 1.25), ylabel="Low-probability tokens (%)")
    ax.legend(fontsize=8, loc="upper left")
    panel(ax, "a", "Vocabulary definitions")
    ax = axes[1]
    cdf(ax, a["vocabulary"]["strict"][0], "CUDA/Triton", BLUE)
    cdf(ax, a["vocabulary"]["strict"][1], "Other tokens", ORANGE, "--")
    ax.axvline(-2, color="#666666", linestyle=":", linewidth=1)
    ax.set(xlim=(-6, 0), yscale="log", ylim=(.01, 100),
           xlabel="Token log-probability", ylabel="Cumulative tokens (%)")
    ax.legend(fontsize=7, loc="upper left")
    panel(ax, "b", "Strict-vocabulary empirical CDF")
    save(fig, "fig1_vocabulary")

    fig, axes = plt.subplots(1, 2, figsize=(7.204724, 3.55), width_ratios=[1.2, 1], layout="constrained")
    ax = axes[0]
    tasks = sorted(b["tasks"])
    labels = {"12_gemm_multiply_leakyrelu": "GEMM + scale + LeakyReLU",
              "14_gemm_divide_sum_scaling": "GEMM + divide + sum + scale",
              "17_conv2d_instancenorm_divide": "Conv2d + InstanceNorm + divide",
              "19_convtranspose2d_gelu_groupnorm": "ConvTranspose2d + GELU + GN",
              "1_conv2d_relu_biasadd": "Conv2d + ReLU + bias"}
    for i, task in enumerate(tasks):
        code, prose = (rate(b["tasks"][task][r]) for r in ("code", "prose"))
        ax.plot([code, prose], [i, i], color="#d2d2dc", linewidth=.9)
        ax.scatter(code, i, color=BLUE, s=24, label="Code" if i == 0 else None, zorder=3)
        ax.scatter(prose, i, edgecolor=ORANGE, facecolor="white", linewidth=1.2,
                   marker="s", s=24, label="Prose" if i == 0 else None, zorder=3)
    ax.set(yticks=range(len(tasks)), yticklabels=[labels[t] for t in tasks],
           xlabel="Low-probability tokens (%)", xlim=(-.5, 16))
    ax.tick_params(axis="y", labelsize=7)
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, -.33), ncol=2)
    panel(ax, "a", "Code and prose by task")
    ax = axes[1]
    for region, color, style in (("code", BLUE, "-"), ("prose", ORANGE, "--")):
        cdf(ax, b["regions"][region], region.title(), color, style)
    ax.axvline(np.log(1e-9), color="#666666", linestyle=":", linewidth=1)
    ax.text(np.log(1e-9)+.4, .003, "Quoted floor", fontsize=7, rotation=90, va="bottom")
    ax.set(xlim=(-22, .2), yscale="log", ylim=(.0001, 100),
           xlabel="Token log-probability", ylabel="Cumulative tokens (%)")
    ax.legend(fontsize=7, loc="upper left")
    panel(ax, "b", "Regional empirical CDF")
    save(fig, "fig2_regions")

    fig, axes = plt.subplots(1, 2, figsize=(7.204724, 3.0), width_ratios=[1.25, .9], layout="constrained")
    ax = axes[0]
    concentration = {}
    for region, color, style in (("code", BLUE, "-"), ("prose", ORANGE, "--")):
        cost = -np.sort(b["regions"][region])
        cumulative = np.cumsum(cost) / cost.sum() * 100
        percent = np.arange(1, len(cost)+1) / len(cost) * 100
        ax.plot(np.r_[0, percent], np.r_[0, cumulative], color=color, linestyle=style, label=region.title())
        concentration[region] = float(cumulative[int(len(cost)*.05)-1])
    ax.axvline(5, color="#666666", linestyle=":", linewidth=1)
    ax.annotate("Lowest-probability 5%", (5, 0), xytext=(27, 9),
                textcoords="data", fontsize=7, color="#666666",
                arrowprops={"arrowstyle": "-", "color": "#888888", "lw": .6})
    ax.set(xlim=(0, 100), ylim=(0, 103), xlabel="Tokens, lowest probability first (%)",
           ylabel="Cumulative negative log-probability (%)")
    ax.legend(fontsize=8, loc="lower right")
    panel(ax, "a", "Concentration of total cost")
    ax = axes[1]
    probs = [float(np.exp(b["regions"][r].min())) for r in ("code", "prose")]
    ax.scatter(probs, [1, 0], c=[BLUE, ORANGE], marker="o", s=42, zorder=3)
    for y, prob in zip((1, 0), probs):
        ax.annotate(f"{prob:.1e}", (prob, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(1e-9, color="#666666", linestyle=":", linewidth=1)
    ax.text(1.2e-9, .4, "Quoted floor", fontsize=8, rotation=90)
    ax.set(xscale="log", xlim=(4e-10, 1e-5), ylim=(-.5, 1.5),
           yticks=[0, 1], yticklabels=["Prose minimum", "Code minimum"], xlabel="Conditional token probability")
    panel(ax, "b", "Observed regional minima")
    save(fig, "fig3_tail")

    fig, axes = plt.subplots(1, 2, figsize=(7.204724, 2.7), width_ratios=[.9, 1.25], layout="constrained")
    contexts = ["initial", "after_success", "after_failure"]
    labels = ["Initial", "After success", "After failure"]
    colors = [BLUE, GREEN, ORANGE]
    counts = [len(b["contexts"][c]) for c in contexts]
    shares = np.array(counts)/sum(counts)*100
    rates = [rate(b["contexts"][c]) for c in contexts]
    for ax, vals, title, ylabel, limit, letter in (
        (axes[0], shares, "Composition of generated output", "Share of tokens (%)", 108, "a"),
        (axes[1], rates, "Low-probability rate by context", "Low-probability tokens (%)", 8.4, "b")):
        bars = ax.barh(labels, vals, color=colors, height=.48)
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in vals], fontsize=7, padding=4)
        ax.set(xlabel=ylabel, xlim=(0, limit))
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e8e8ed", linewidth=.45)
        ax.tick_params(axis="y", labelsize=8)
        panel(ax, letter, title)
    save(fig, "fig4_context")

    metrics = {
        "source_rule": "All saved content_logprobs; fence-state assignment as in stream_stats.py; context grouping as in analyze_loop.py.",
        "family_a_tokens": sum(map(len, a["regions"].values())),
        "vocabulary_rates": {k: [rate(v) for v in pair] for k, pair in a["vocabulary"].items()},
        "family_b_regions": {r: {"tokens": len(v), "rate": rate(v), "min_logprob": float(v.min())} for r, v in b["regions"].items()},
        "task_rates": {t: {r: rate(v) for r, v in d.items()} for t, d in b["tasks"].items()},
        "concentration_at_5_percent": concentration,
        "context_counts": dict(zip(contexts, counts)), "context_rates": dict(zip(contexts, rates)),
        "sources": a["sources"]+b["sources"],
    }
    (OUT/"figure-data.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
