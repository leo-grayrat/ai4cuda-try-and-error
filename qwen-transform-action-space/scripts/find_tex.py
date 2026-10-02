#!/usr/bin/env python3
"""Locate a TeX engine on this machine.

The pipeline previously reported "no TeX engine installed" after checking PATH only.
That conclusion is only valid for the sandbox's PATH view, which is narrower than the
user's interactive shell; a VS Code LaTeX workflow implies an engine exists somewhere.
This script searches PATH, the usual install roots, and any tex binaries under the
user profile, and reports absolute paths that a build could target directly.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ENGINES = ["latexmk", "xelatex", "pdflatex", "lualatex", "tectonic", "bibtex", "biber", "tex"]

ROOTS = [
    r"C:\texlive", r"C:\miktex", r"C:\Program Files\MiKTeX", r"C:\Program Files\texlive",
    r"C:\Program Files (x86)\MiKTeX", r"C:\Tools",
    str(Path.home() / "AppData" / "Local" / "Programs"),
    str(Path.home() / "AppData" / "Roaming"),
    str(Path.home() / "scoop"), str(Path.home() / "bin"),
    r"D:\texlive", r"E:\texlive", r"C:\Users\34788\.local\bin",
]


def shuf(name: str) -> str | None:
    return shutil.which(name)


def main() -> int:
    print("== PATH (this process) ==")
    print(os.environ.get("PATH", "")[:900] or "(empty)")
    print("\n== shutil.which on PATH ==")
    for e in ENGINES:
        print(f"  {e:<10} {shuf(e) or '--'}")

    print("\n== candidate roots ==")
    for r in ROOTS:
        p = Path(r)
        print(f"  {'DIR ' if p.is_dir() else '----'} {r}")

    print("\n== recursive search for tex binaries in roots (depth-limited) ==")
    found: list[str] = []
    names = {f"{e}.exe" for e in ENGINES}
    for r in ROOTS:
        p = Path(r)
        if not p.is_dir():
            continue
        try:
            for depth1 in p.iterdir():
                for depth2 in ([depth1] + (list(depth1.iterdir()) if depth1.is_dir() else [])):
                    try:
                        for f in depth2.rglob("*.exe"):
                            if f.name.lower() in names:
                                found.append(str(f))
                    except (PermissionError, OSError):
                        pass
        except (PermissionError, OSError):
            pass
    if found:
        for f in sorted(set(found))[:25]:
            print("  FOUND", f)
    else:
        print("  none")

    print("\n== where.exe sweep (uses the shell PATH, not this process PATH) ==")
    rc = os.system("where.exe xelatex pdflatex latexmk tectonic 2>NUL")
    print(f"  where.exe exit={rc} (1 = nothing found)")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
