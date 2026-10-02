#!/usr/bin/env python3
"""Mirror the Section 3.6 surgery into the Chinese part, and survey calque pronouns.

Two jobs:

1. paper/main.md lost two run-log paragraphs in Section 3.6 (they moved to Appendix B),
   but paper/zh/part1.md still carried their translations. The two deliverables had
   drifted structurally; this removes them and inserts the translated replacement
   paragraph ("How to read the table").

2. The translation contains literal calques of English demonstratives that read badly in
   Chinese: 「那个问题」「那个结果」 where English used "the question" / "the result" as a
   bare referent. This script reports every occurrence with context so the fix can be
   judged per instance rather than by blind find-and-replace.

Usage: python scripts/sync_zh_surgery.py [--survey-only]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZH1 = ROOT / "paper" / "zh" / "part1.md"

REPLACEMENT = (
    "**如何读这张表。** 真正有判别力的是*每个描述子相对自身零模型的超出量*，而不是原始互信息水平，"
    "因为 plug-in 估计量会随占用格数增多而上升；两个描述子都在同一批边、同一批结果上计算，"
    "置换校准则让各自的零模型与自身的粒度相匹配。plug-in 与 Miller–Madow 校正后的水平都列出，"
    "是为了让校正项的取舍不必靠信任来接受。本项测量的生产历史——包括把边际熵校正项误用在本该用"
    "联合校正之处、因而使结论反号的那个版本——记录在附录 B。\n"
)

CALQUES = [
    "那个问题", "那个结果", "那个论断", "那个属性", "那个数量", "那个东西",
    "这一个", "它恰恰", "这正是那个",
]


def main() -> int:
    survey_only = "--survey-only" in sys.argv
    t = ZH1.read_text(encoding="utf-8")

    start = t.find("**我们保留下来的方法论教训。**")
    end = t.find("### 3.7")
    if start > 0 and end > start:
        removed = t[start:end]
        t = t[:start] + REPLACEMENT + "\n" + t[end:]
        print(f"removed {len(removed)} chars of run-log narrative from zh Section 3.6")
        for token in ("4.4", "7.5", "112"):
            if token in t[: t.find("### 3.7")]:
                print(f"  NOTE: token {token} still present in zh 3.x region")
    else:
        print("zh run-log paragraphs not found (already synced?)")

    if not survey_only:
        ZH1.write_text(t, encoding="utf-8")
        print(f"wrote {ZH1}")

    print("\n=== calque survey (both parts) ===")
    for name in ("part1.md", "part2.md"):
        text = (ROOT / "paper" / "zh" / name).read_text(encoding="utf-8")
        for c in CALQUES:
            for m in re.finditer(re.escape(c), text):
                s = max(0, m.start() - 28)
                print(f"  {name} …{text[s:m.end()+26].strip()}…")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
