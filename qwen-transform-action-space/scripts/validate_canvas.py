#!/usr/bin/env python3
"""Validate the run-report canvas when the host TypeScript check is unavailable.

Checks what can be checked without a compiler:
  1. every name imported from qoder/canvas exists in the SDK declarations
  2. exactly one outer ReportShell, sections use ReportSection
  3. no external imports, no network calls, no hardcoded hex colours
  4. canvasImage targets resolve to real files beside the canvas
Exit 0 only if all pass.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

t = Path(os.environ["USERPROFILE"]) / ".qoder/projects/e-Project-Git-Research-ai4cuda-try-and-error/canvases"
sdk = Path(os.environ["USERPROFILE"]) / ".qoder/canvas/sdk"
canvas = next(t.glob("*.canvas.tsx"))
src = canvas.read_text(encoding="utf-8")

fails: list[str] = []

decls = "".join(p.read_text(encoding="utf-8") for p in sorted(sdk.glob("*.d.ts")))
exported = set(re.findall(r"^export (?:declare )?(?:function|const|class) (\w+)", decls, re.M))
exported |= set(re.findall(r"^\s*(\w+),", decls, re.M))
exported |= set(re.findall(r"export type \{[^{]*?\} from", decls))  # noise, harmless

m = re.search(r"import \{(.*?)\} from \"qoder/canvas\";", src, re.S)
if not m:
    fails.append("no qoder/canvas import block found")
    names = []
else:
    names = [n.strip() for n in m.group(1).replace("\n", " ").split(",") if n.strip()]
missing = [n for n in names if n not in exported]
print(f"imports used: {len(names)} | unresolved against SDK declarations: {missing or 'none'}")
if missing:
    fails.append(f"unknown SDK exports: {missing}")

shell_count = len(re.findall(r"<ReportShell\b", src))
sections = len(re.findall(r"<ReportSection\b", src))
print(f"outer ReportShell: {shell_count} | ReportSection blocks: {sections}")
if shell_count != 1:
    fails.append("expected exactly one ReportShell")
if sections < 3:
    fails.append("expected multiple ReportSection blocks")

ext = re.findall(r"from \"(?!qoder/canvas)([^\"]+)\";", src)
print(f"external imports: {ext or 'none'}")
if ext:
    fails.append(f"external imports present: {ext}")

print(f"fetch present: {'fetch(' in src}")
if "fetch(" in src:
    fails.append("network call in canvas")

hexes = re.findall(r"#[0-9a-fA-F]{3,8}\b", src)
print(f"hardcoded hex colours: {hexes or 'none'}")
if hexes:
    fails.append(f"hardcoded colours: {hexes}")

imgs = re.findall(r'canvasImage\("\./([^"]+)"\)', src)
print(f"canvasImage targets: {imgs}")
for name in imgs:
    if not (t / name).exists():
        fails.append(f"canvasImage target missing: {name}")

print()
if fails:
    print("CANVAS VALIDATION: FAIL")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("CANVAS VALIDATION: PASS (import existence, structure, no external deps, images present)")
