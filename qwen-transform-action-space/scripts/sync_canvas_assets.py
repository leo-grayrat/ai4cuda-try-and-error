#!/usr/bin/env python3
"""Make the run-report canvas renderable from inside the repository.

The host renders the workspace copy at paper/run-report.canvas.tsx, and canvasImage
resolves "./<name>.png" as a sibling of the canvas file. The figure PNGs live in
figures/ and paper/latex/figures/, so the sibling lookup failed with
"Canvas local image could not be read".

This script copies every image referenced by the canvas next to it, then checks both
canvas copies (repository copy and the IDE canvases directory) for:
  - one outer ReportShell, ReportSection-driven sections
  - all qoder/canvas imports present in the SDK declarations
  - every canvasImage target resolvable as a sibling file
  - no external imports, no network calls, no hardcoded hex colours

Run after editing the canvas or regenerating figures:
    python scripts/sync_canvas_assets.py
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANVAS = ROOT / "paper" / "run-report.canvas.tsx"
FIGSRC = ROOT / "figures"

HOST_DIR = (
    Path(os.environ.get("USERPROFILE", str(Path.home())))
    / ".qoder/projects/e-Project-Git-Research-ai4cuda-try-and-error/canvases"
)
SDK_DIR = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".qoder/canvas/sdk"


def sibling_names(src: str) -> list[str]:
    return re.findall(r'canvasImage\("\./([^"]+)"\)', src)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    if not CANVAS.exists():
        print(f"FATAL: {CANVAS} missing", file=sys.stderr)
        return 1

    src = CANVAS.read_text(encoding="utf-8")
    names = sibling_names(src)

    # 1. place referenced images beside the repository canvas
    print("syncing canvas image assets")
    for name in names:
        if (CANVAS.parent / name).exists():
            print(f"  present  {name}")
            continue
        candidate = FIGSRC / name
        if not candidate.exists():
            print(f"  MISSING SOURCE {name}", file=sys.stderr)
            return 1
        shutil.copy2(candidate, CANVAS.parent / name)
        print(f"  copied   {name} -> paper/{name}")

    # 2. keep the IDE canvases directory in sync with the repository copy
    if HOST_DIR.exists():
        host_canvas = HOST_DIR / "transform-action-space-run-report.canvas.tsx"
        shutil.copy2(CANVAS, host_canvas)
        for name in names:
            shutil.copy2(CANVAS.parent / name, HOST_DIR / name)
        print(f"synced IDE copy -> {host_canvas}")
    else:
        print(f"note: IDE canvases dir not present ({HOST_DIR}); repository copy only")

    # 3. validate both copies
    decls = "".join(p.read_text(encoding="utf-8") for p in sorted(SDK_DIR.glob("*.d.ts")))
    exported = set(re.findall(r"^export (?:declare )?(?:function|const|class) (\w+)", decls, re.M))
    exported |= set(re.findall(r"^\s*(\w+),", decls, re.M))

    ok_all = True
    for path in [p for p in (CANVAS, HOST_DIR / "transform-action-space-run-report.canvas.tsx") if p.exists()]:
        print(f"\nvalidating {path}")
        text = path.read_text(encoding="utf-8")
        m = re.search(r"import \{(.*?)\} from \"qoder/canvas\";", text, re.S)
        # split on commas then drop blanks: the import block ends with a trailing
        # comma, which otherwise yields '' and reports a phantom unknown export
        imports = [n.strip() for n in (m.group(1).replace("\n", " ").split(",") if m else [])]
        imports = [n for n in imports if n]
        unknown = [i for i in imports if i not in exported]
        ok_all &= check(f"{len(imports)} imports exist in SDK", not unknown, str(unknown))
        shells = len(re.findall(r"<ReportShell\b", text))
        ok_all &= check("exactly one outer ReportShell", shells == 1, f"found {shells}")
        sections = len(re.findall(r"<ReportSection\b", text))
        ok_all &= check("sections use ReportSection", sections >= 3, f"{sections} sections")
        ext = re.findall(r'from "(?!qoder/canvas)([^"]+)";', text)
        ok_all &= check("no external imports", not ext, str(ext))
        ok_all &= check("no network calls", "fetch(" not in text)
        hexes = re.findall(r"#[0-9a-fA-F]{3,8}\b", text)
        ok_all &= check("no hardcoded colours", not hexes, str(hexes))
        missing = [img for img in sibling_names(text) if not (path.parent / img).exists()]
        ok_all &= check("all canvasImage targets resolve as siblings", not missing, str(missing))

    print("\nCANVAS SYNC + VALIDATION: " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
