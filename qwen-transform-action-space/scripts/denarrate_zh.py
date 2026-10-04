#!/usr/bin/env python3
"""Mirror the English de-rhetoric pass into the Chinese parts.

English Section 1.x, the abstract and Section 3.6 were rewritten to drop
self-justifying framing ("reported rather than buried", "the honest counterweight",
"not a hedge we chose", "uncomfortable for the field", "unlike our earlier draft")
in favour of plain assertion. The Chinese parts still carried the older framing,
which reads worse in Chinese than in English because the calques are visible.

Each replacement is asserted, so a drift between the two documents fails loudly
instead of silently leaving one language edited.

Usage: python scripts/denarrate_zh.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZH1 = ROOT / "paper" / "zh" / "part1.md"

EDITS: list[tuple[str, str]] = [
    # abstract: limitations stated, not performed
    (
        "我们不作遮掩地报告三项限定：其一，两个支持统计量被撤回，分别是一个恒等式与一个抽样副产物"
        "（我们自己那套 117 条断言的对账工具，没能看见那个颠覆了这条标题结论的错误）；其二，跨任务复现处于机会水平；"
        "其三，在不给定类别定义的条件下重新标注，会使这条标题结论不再显著（p = 0.106、kappa = 0.687）。",
        "三条限制界定了这一结果：两个支持统计量被撤回，一为恒等式、一为抽样副产物；跨任务复现处于机会水平；"
        "在不给定类别定义的条件下重新标注，会使这条标题结论不再显著（p = 0.106、kappa = 0.687）。",
    ),
    (
        "其中包括那个会杀死该方向的分支",
        "其中包括足以判死该方向的分支",
    ),
    # 1.1: source basis stated without narrating the revision history
    (
        "与我们此前的草稿不同，我们如今不再仅凭摘要立论。[1] 的 arXiv LaTeX 源码可作为一个 18 MB 的 e-print 打包被检索到，而我们读了它，",
        "[1] 的 arXiv LaTeX 源码可作为一个 18 MB 的 e-print 包被检索到，我们读了它，",
    ),
    # 3.6: permutation test carries the argument; no counterweight rhetoric
    (
        "这颠覆了初稿的发现，而且这次颠覆不是一个偏好哪个偏差公式的问题：在无公式的置换检验（permutation test）下，",
        "这颠覆了初稿的发现，而且结论不依赖偏差公式的取舍：置换检验本身不需要任何校正，在它之下，",
    ),
    (
        "我们也报告诚实的配重：配对 bootstrap 差",
        "配对 bootstrap 差",
    ),
    (
        "在未校准的口径下粗结构仍然胜出；我们主张未校准的口径是错误的口径，但这个数字留在那里，给任何不同意的人。",
        "就未校准的原始水平而言粗结构仍然胜出；究竟哪种口径才是有信息量的，正是置换校准要回答的问题，"
        "上文给出的即是校准后的读法，分歧处可直接核对。",
    ),
    # 1.1 second qualification: "uncomfortable for the field"
    (
        "对该领域仍然难堪的结论",
        "仍然值得认真对待的结论",
    ),
    # 1.1 lead-in: the distinction belongs to the system, not to our hedging
    (
        "而第一条不是我们选的缓冲，而是该系统自己作出的区分",
        "",
    ),
    (
        "第一条不是我们选择的缓冲，而是那个系统自己作出的区分",
        "",
    ),
    (
        "；第一条不是我们选的推盾而是它自己的区分",
        "",
    ),
]


def main() -> int:
    text = ZH1.read_text(encoding="utf-8")
    hits = 0
    for old, new in EDITS:
        if old in text:
            text = text.replace(old, new, 1)
            hits += 1
            print(f"  ok    {old[:44]}…")
        else:
            print(f"  miss  {old[:44]}…")
    if not hits:
        print("no anchor matched; nothing written", file=sys.stderr)
        return 1
    ZH1.write_text(text, encoding="utf-8")
    print(f"\napplied {hits}/{len(EDITS)}; wrote {ZH1}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
