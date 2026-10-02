#!/usr/bin/env python3
"""Inspect the built PDF's typography instead of trusting that it compiled.

Reports, for the first pages: the font actually used per span, its size, whether
inter-word spaces exist as real glyphs, and the count of overfull lines in the log.
Also writes PNG renders of the first two pages into .cache/ for visual inspection.

Usage: python scripts/inspect_pdf.py [page_count]
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "latex" / "main.pdf"
LOG = ROOT / "paper" / "latex" / "main.log"
CACHE = ROOT / ".cache"


def main() -> int:
    n_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    doc = fitz.open(PDF)
    print(f"pages: {len(doc)}")

    fonts = Counter()
    spaceless = 0
    total_spans = 0
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    total_spans += 1
                    fonts[span["font"]] += 1
                    text = span.get("text", "")
                    if len(text) > 24 and " " not in text:
                        spaceless += 1

    print(f"spans: {total_spans} | distinct fonts: {len(fonts)}")
    for f, c in fonts.most_common(6):
        print(f"  {c:>6}  {f}")
    print(f"long spans with no space at all (suspicious): {spaceless}")

    print("\n--- first page, span by span ---")
    for block in doc[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span.get("text", "")
                if not text.strip():
                    continue
                print(f"  {span['size']:>5.1f}pt {span['font'][:24]:<26}| {text[:88]}")

    if LOG.exists():
        log = LOG.read_text(encoding="utf-8", errors="replace")
        over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\) in (.+?) at lines (\d+)", log)
        print(f"\noverfull hboxes: {len(over)}")
        for amt, kind, ln in sorted(over, key=lambda x: -float(x[0]))[:8]:
            print(f"  {float(amt):>7.1f}pt  {kind}  line {ln}")
        print("underfull/vbox warnings:", len(re.findall(r"Underfull", log)))
        print("missing chars:", len(re.findall(r"Missing character", log)))

    CACHE.mkdir(exist_ok=True)
    for i in range(min(n_pages, len(doc))):
        path = CACHE / f"pdfcheck_page{i + 1}.png"
        doc[i].get_pixmap(dpi=110).save(path)
        print(f"rendered {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
