#!/usr/bin/env python3
"""Repair a mis-slice from the narrative relocation and finish the de-rhetoric pass.

The relocation script cut from "### 7.5" up to "## 8.", which also swept up the
section that came after 7.5 — "7.6 What this paper does not show", the paper's single
consolidated limitations statement. It ended up buried inside Appendix B in both
languages, leaving the body without its limitations section while the appendix carried
one. This script:

  1. lifts that section out of Appendix B and renumbers it 7.5 (the old 7.5 is gone),
     placing it before Section 8 in both paper/main.md and paper/zh/part2.md
  2. repairs any "Section 7.6" cross-reference left pointing at the old number
  3. compresses the residual revision narration in that section and in Section 7.1 to a
     single pointer at Appendix B, and drops the "declaration is not a control" framing

Usage: python scripts/repair_section_order.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "paper" / "main.md"
ZH2 = ROOT / "paper" / "zh" / "part2.md"

EN_TITLE = "### 7.6 What this paper does not show"
ZH_TITLE = "### 7.6 本文没有证明什么"
EN_NEW = "### 7.5 What this paper does not show"
ZH_NEW = "### 7.5 本文没有证明什么"

# revision narration condensed to one pointer
EN_NARRATION_OLD_MARK = "Neither withdrawal is a hedge."
EN_NARRATION_NEW = (
    "Neither withdrawal softens the other. The corrected reading replaces an earlier version of\n"
    "Section 3.6 whose conclusion was the opposite; why the error survived every automated check,\n"
    "and what that says about self-verification in pipelines like this one, is recorded in Appendix B."
)
EN_TAIL_OLD = "We report the corrected positive result alongside the fact that we had it backwards, and we treat both the"
EN_TAIL_NEW = "The corrected result is reported here, and the"

ZH_NARRATION_MARK = "两项撤回都不是推托。"
ZH_NARRATION_NEW = (
    "两项撤回都不减轻另一项。校准后的读法取代了第 3.6 节一个结论相反的早期版本；"
    "那个错误为何能通过全部自动检查，以及这对同类流水线的自我验证意味着什么，记录在附录 B。"
)


def relocate(path: Path, title: str, new_title: str, narration_mark: str,
             narration_new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    i = text.find(title)
    if i < 0:
        print(f"  {path.name}: section title not found")
        return False

    # section runs to the next heading of any level, or to end of file
    m = re.compile(r"\n#{2,3} ").search(text, i + len(title))
    end = m.start() + 1 if m else len(text)
    block = text[i:end]
    text = text[:i] + text[end:]
    block = block.replace(title, new_title, 1)

    j = text.find("\n## 8.")
    if j < 0:
        print(f"  {path.name}: Section 8 anchor missing after removal; aborting file")
        return False
    text = text[:j] + "\n" + block.rstrip() + "\n" + text[j:]
    text = text.replace("Section 7.6", new_title.split()[1]).replace("第 7.6 节", "第 7.5 节")

    # narration compression, applied only where the marker still exists
    k = text.find(narration_mark)
    if k >= 0:
        stop = text.find("\n\n", k)
        text = text[:k] + narration_new + text[stop if stop > 0 else len(text):]
        print(f"  {path.name}: condensed revision narration")

    path.write_text(text, encoding="utf-8")
    print(f"  {path.name}: moved {title} -> {new_title}, before Section 8")
    return True


def main() -> int:
    print("repairing section order")
    ok = relocate(EN, EN_TITLE, EN_NEW, EN_NARRATION_OLD_MARK, EN_NARRATION_NEW)
    ok &= relocate(ZH2, ZH_TITLE, ZH_NEW, ZH_NARRATION_MARK, ZH_NARRATION_NEW)

    # one remaining framing sentence in Section 7.1, both languages
    for path, old, new in (
        (EN, "Declaring this, as our first draft did, is not a control.",
         "A declaration of this is not a control."),
        (ZH2, "像我们初稿那样声明这一点，不是一个对照。",
         "仅这样声明本身不是一个对照。"),
    ):
        text = path.read_text(encoding="utf-8")
        if old in text:
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            print(f"  {path.name}: reframed the 7.1 lead sentence")

    # structural assertion: the body must contain the limitations section before Section 8
    for path, title, appendix in (
        (EN, EN_NEW, "## Appendix B"),
        (ZH2, ZH_NEW, "## 附录 B"),
    ):
        text = path.read_text(encoding="utf-8")
        if text.find(title) < 0:
            print(f"FATAL: {path.name} has no limitations section", file=sys.stderr)
            return 1
        a = text.find(appendix)
        if a < 0:
            print(f"FATAL: {path.name} has no {appendix!r} heading", file=sys.stderr)
            return 1
        if a < text.find(title):
            print(f"FATAL: {path.name} limitations section still sits after the appendix",
                  file=sys.stderr)
            return 1
        if title in text[a:]:
            print(f"FATAL: {path.name} limitations section duplicated inside the appendix",
                  file=sys.stderr)
            return 1
        print(f"  {path.name}: order verified (limitations before appendix)")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
