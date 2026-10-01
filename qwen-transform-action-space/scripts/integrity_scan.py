#!/usr/bin/env python3
"""Stage 2.5 integrity scan for the English manuscript.

Checks, mechanically where possible:
  1. non-ASCII CJK characters left in an English manuscript
  2. every in-text citation key resolves to a numbered reference entry
  3. every reference entry is cited at least once (no orphan sources)
  4. each reference carries an identifier we can verify (arXiv id or DOI)
  5. word count and section coverage against the intended outline

Usage: python scripts/integrity_scan.py [path/to/main.md]
Exit 0 if clean, 1 if any check fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

INTENDED_SECTIONS = [
    "Abstract",
    "1. Introduction",
    "2. Position",
    "3. Preliminary Evidence",
    "4. A Falsifiable Experimental Framework",
    "5. Implications Beyond CUDA",
    "6. Related Work",
    "7. Limitations and Open Questions",
    "8. Conclusion",
    "References",
    "Appendix A",
]

CJK = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
REF_LINE = re.compile(r"^(\d+)\.\s+(.+)$", re.M)
ID_PAT = re.compile(r"(arXiv:(?:\d{4}\.\d{4,5}(?:v\d+)?)|doi:10\.\d{4,9}/\S+)")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/main.md")
    if not path.exists():
        print(f"FATAL: {path} not found", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    failures = []

    # 1. CJK leakage
    cjk_hits = [(i + 1, l) for i, l in enumerate(lines) if CJK.search(l)]
    if cjk_hits:
        failures.append(
            f"non-ASCII CJK text in English manuscript at lines: "
            f"{[n for n, _ in cjk_hits]}"
        )
    print(f"[1] CJK leakage: {len(cjk_hits)} line(s)")

    # 2/3. citation <-> reference closure
    cited: set[int] = set()
    for m in CITE.finditer(text):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit():
                cited.add(int(tok))
    refs = {int(n): body for n, body in REF_LINE.findall(text)}
    # restrict reference numbering to the References section
    ref_start = text.find("\n## References")
    if ref_start < 0:
        failures.append("no '## References' section")
        refs = {}
    else:
        refs = {
            int(n): body
            for n, body in REF_LINE.findall(text[ref_start:])
        }

    dangling = sorted(c for c in cited if refs and c not in refs)
    orphans = sorted(k for k in refs if k not in cited)
    print(f"[2] in-text citation keys: {len(cited)}, reference entries: {len(refs)}")
    if dangling:
        failures.append(f"citations with no reference entry: {dangling}")
    if orphans:
        failures.append(f"references never cited in body: {orphans}")

    # 4. verifiable identifier present
    noid = [k for k, v in refs.items() if not ID_PAT.search(v)]
    print(f"[4] references lacking arXiv id or DOI: {len(noid)} {noid}")
    if noid:
        failures.append(f"references without a verifiable identifier: {noid}")

    # 5. structure + length
    words = len(re.findall(r"[A-Za-z0-9'\u00c0-\u024f-]+", text))
    print(f"[5] approx words: {words}")
    missing_sections = [
        s for s in INTENDED_SECTIONS if not re.search(rf"^#{{1,3}}\s+{re.escape(s)}", text, re.M)
    ]
    if missing_sections:
        failures.append(f"missing intended sections: {missing_sections}")
    print(f"[5] sections present: {len(INTENDED_SECTIONS) - len(missing_sections)}/{len(INTENDED_SECTIONS)}")

    print()
    if failures:
        print("INTEGRITY SCAN: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("INTEGRITY SCAN: PASS (mechanical checks only; see STAGE2_5_INTEGRITY.md for the semantic checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
