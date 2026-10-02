#!/usr/bin/env python3
"""Show the LaTeX error context so a build failure can be read rather than guessed.

Prints, for each hard error in main.log, the message, the file line it blames, and the
corresponding lines of main.tex.

Usage: python scripts/show_tex_errors.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "paper" / "latex" / "main.log"
TEX = ROOT / "paper" / "latex" / "main.tex"


def main() -> int:
    log = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    tex = TEX.read_text(encoding="utf-8").splitlines()

    shown = 0
    for i, line in enumerate(log):
        if line.startswith("! ") and shown < 3:
            shown += 1
            print("=" * 70)
            print("ERROR:", line)
            print("context:")
            print("\n".join(log[i : i + 6]))
            m = re.search(r"^l\.(\d+)", "\n".join(log[i : i + 8]), re.M)
            if m:
                ln = int(m.group(1))
                print(f"--- main.tex lines {ln - 3}..{ln + 2} ---")
                for k in range(max(0, ln - 4), min(len(tex), ln + 2)):
                    print(f"{k + 1:>6}| {tex[k][:150]}")
            print()
    if not shown:
        print("no hard errors in log")
        over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)",
                          LOG.read_text(encoding="utf-8", errors="replace"))
        print(f"overfull hboxes: {len(over)}; worst: {max((float(x) for x in over), default=0):.1f}pt")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
