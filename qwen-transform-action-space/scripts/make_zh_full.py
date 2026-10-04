#!/usr/bin/env python3
"""Assemble the Chinese parallel translation from its two parts.

paper/zh/part1.md and part2.md are produced by two independent translation passes
over disjoint section ranges of paper/main.md. This script concatenates them under
a single header, appends the translator's note, and repairs the one technical
difference that the depth change introduces: figure paths.

It deliberately does NOT re-translate or edit content. A previous inline assembly
command corrupted backticks in the header (shell quoting) and silently dropped the
translator's note; building the file here makes both failures impossible to repeat
and lets verify_zh_alignment.py check the result.

Usage: python scripts/make_zh_full.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZH = ROOT / "paper" / "zh"

HEADER = """# 是 Token 级动作空间选错了坐标系吗？——把程序变换作为 LLM 驱动 kernel 优化的动作空间

> 中文对照稿。以英文稿 `paper/main.md` 为准（Stage 4 v2，经五视角模拟评审修订）。
> 参考文献 32 条按英文原样保留，不翻译。
> 本稿由自动化流水线翻译，未经人工复核；术语与判定词的保留策略见文末「译者说明」。

---

"""

FOOTER = """

---

## 译者说明

- 判定词 `supported` / `not supported` / `unidentifiable` / `withdrawn` 保留英文原词并附中文，
  以免弱化或强化原文的证据强度；`Fuse`、`ChangeLaunchConfig`、`Brute-search`、
  `Structured-action`、`Oracle-action`、`Free-plan`、`Direct` 等类名与条件名不译。
- 近义词分译且全文一致：realize / realization obligation → 实现／实现义务；
  program instantiation → 程序实例化；materialize → 实体化。
- 统计量、p 值、随机种子 20261002、表格数字与参考文献条目均与英文稿逐位一致；
  翻译过程未做任何"顺手更正"。英文稿内部曾存在符号不一致（ASCII 与 Unicode 混用），
  已由作者统一为 `a = (τ, ℓ, θ)` 与 `φ : ΔP → a`，本稿随之一致。
- 图片路径由英文稿的 `../figures/` 调整为 `../../figures/`（本稿位于 `paper/zh/`，深一层）。
  这是唯一为保持可渲染而做的技术性路径修改，不涉及任何内容改动。
"""


def strip_top_matter(text: str) -> str:
    """Drop a part file's own leading note lines, rules, and (optional) single H1.

    The two parts are not structurally identical: part1 covers the title page so it
    has an H1, while part2 starts mid-document at Section 4 and has none. An earlier
    version of this helper searched for an H1 and, finding none in part2, discarded
    its entire 257 lines -- which is how a Chinese full text silently lost every
    section from 4 onward while still reporting success. Never assume a heading
    exists; strip only what is actually at the top.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith(">") or lines[i].strip() in ("", "---")):
        i += 1
    if i < len(lines) and lines[i].startswith("# "):
        i += 1
        while i < len(lines) and (lines[i].startswith(">") or lines[i].strip() in ("", "---")):
            i += 1
    return "\n".join(lines[i:]).lstrip()


def main() -> int:
    p1 = (ZH / "part1.md").read_text(encoding="utf-8")
    p2 = (ZH / "part2.md").read_text(encoding="utf-8")

    b1 = strip_top_matter(p1).rstrip()
    b2 = strip_top_matter(p2).rstrip()

    # fail loudly rather than emit a silently truncated document
    for name, b, src in (("part1", b1, p1), ("part2", b2, p2)):
        if len(b) < 0.8 * len(src.strip()):
            print(
                f"FATAL: {name} lost {len(src.strip()) - len(b)} of "
                f"{len(src.strip())} characters during top-matter stripping",
                file=__import__("sys").stderr,
            )
            return 1
        if "## 8" not in b and name == "part2":
            print("FATAL: part2 body does not contain Section 8", file=__import__("sys").stderr)
            return 1

    body = b1 + "\n\n---\n\n" + b2
    n_paths = len(re.findall(r"(?<!\.)\.\./figures/", body))
    body = body.replace("../figures/", "../../figures/")

    full = HEADER + body + FOOTER
    out = ZH / "full.md"
    out.write_text(full, encoding="utf-8")

    cjk = len(re.findall(r"[\u4e00-\u9fff]", full))
    src_cjk = len(re.findall(r"[\u4e00-\u9fff]", p1 + p2))
    # section headings carried over from the English manuscript
    heads = re.findall(r"^##\s+\S+", full, flags=re.M)
    print(f"wrote {out}")
    print(f"figure paths rewritten: {n_paths}")
    print(f"CJK characters: {cjk} (source parts: {src_cjk})")
    print(f"total characters: {len(full)}")
    print(f"top-level sections: {len(heads)}")
    if cjk < 0.95 * src_cjk:
        print("FATAL: assembled document lost translated content", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
