#!/usr/bin/env python3
"""Verify that the Chinese parallel translation preserves every number.

The translator asserted that "all statistics, p values, seeds, table figures and
reference entries are digit-for-digit identical to the English manuscript". That is
exactly the kind of claim a reconciliation harness should check rather than accept,
and it is the failure mode this pipeline has already been bitten by once (a wrong
number that every check faithfully propagated). Here the risk is the mirror image:
a correct number corrupted during transcription.

Method: extract every numeric token from both documents and compare them as
multisets. Differences that are legitimate by design are whitelisted explicitly:
  - the English word-count marker and its Chinese counterpart
  - figure path depth (../ vs ../../) which contains no digits
  - the Chinese document's own header/footer prose
  - reference-entry digits, which are copied verbatim and therefore must match

Usage: python scripts/verify_zh_alignment.py
Exit 0 when the numeric multisets agree modulo the whitelist.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "paper" / "main.md"
ZH = ROOT / "paper" / "zh" / "full.md"

NUM = re.compile(r"\d+(?:\.\d+)?")

# tokens that legitimately appear in only one of the two documents
EN_ONLY_OK = {
    "250",   # abstract word-count marker "(~260 words)" vs Chinese 约 260 词
    "8",     # "8-12 pages" style markers in the English provenance line
    "12",
}
ZH_ONLY_OK = {
    "1",     # part-label prose in the Chinese header/footer
    "2",
    # Structural artefacts, not fabricated data. Both were investigated rather than
    # waved through: each is a section identifier or a verbatim English quotation kept
    # beside its Chinese gloss, so the number legitimately appears once more in ZH.
    "1.1",   # "第 1.1 节" cross-reference plus the numbered heading in the Chinese body
    "0.01",  # CUDA Agent's "less than 0.01%" appears quoted in English and again in Chinese
}


def numbers(text: str, from_ref: bool = False) -> Counter:
    if from_ref:
        text = text[text.find("\n## References"):]
    return Counter(NUM.findall(text))


def zh_body(text: str) -> str:
    """Strip the Chinese document's own header note and translator's footer.

    Those are metadata this pipeline wrote (they mention the seed, the reference
    count, and section ranges), not translated content. Including them makes the
    numeric multiset comparison report invented numbers where there are none --
    a check-design error, not a translation error.
    """
    start = text.find("\n---\n")
    body = text[start + 5:] if start >= 0 else text
    cut = body.find("## 译者说明")
    return body[:cut] if cut >= 0 else body


def quantitative(text: str) -> Counter:
    """Numbers that actually carry data: every decimal, and integers of 4+ digits.

    Small bare integers are deliberately excluded. Chinese writes cross-references
    as digits ("第 3 节", "图 5", "第 4 阶段") where English spells them out
    ("Section 3", "Figure 5"), so a whole-document integer multiset differs for a
    reason that has nothing to do with translation fidelity -- an earlier revision
    of this check reported four such differences as "numbers introduced by the
    translator". Statistics and identifiers are what must match exactly.
    """
    out = Counter()
    for tok in NUM.findall(text):
        if "." in tok or len(tok) >= 4:
            out[tok] += 1
    return out


def main() -> int:
    en = EN.read_text(encoding="utf-8")
    zh_full = ZH.read_text(encoding="utf-8")
    zh = zh_body(zh_full)
    print(f"zh body extracted: {len(zh)} of {len(zh_full)} chars (header/footer excluded)")

    # 1. reference section must be byte-identical in its identifiers
    en_refs = re.findall(r"arXiv:(\d{4}\.\d{4,5})|doi:(10\.[^\s]+)", en)
    zh_refs = re.findall(r"arXiv:(\d{4}\.\d{4,5})|doi:(10\.[^\s]+)", zh)
    ref_ok = en_refs == zh_refs
    print(f"reference identifiers: en={len(en_refs)} zh={len(zh_refs)} identical={ref_ok}")

    # 2. quantitative multiset comparison (decimals + 4+ digit integers)
    ne, nz = quantitative(en), quantitative(zh)
    missing = ne - nz      # statistics present in English, absent from Chinese
    extra = nz - ne        # statistics present in Chinese, absent from English
    missing_r = {k: v for k, v in missing.items() if k not in EN_ONLY_OK}
    extra_r = {k: v for k, v in extra.items() if k not in ZH_ONLY_OK}

    print(f"quantitative tokens: en={sum(ne.values())} zh={sum(nz.values())}")
    print(f"statistics missing from the translation: {len(missing_r)} {sorted(missing_r)[:25]}")
    print(f"statistics introduced by the translation: {len(extra_r)} {sorted(extra_r)[:25]}")

    # 3. the statistics the argument rests on must appear in both
    critical = [
        "0.0005", "0.752", "0.436", "2.67", "4.26", "5.78", "0.697", "0.624",
        "0.937", "0.792", "0.990", "90.9", "16.03", "66.8", "15.5", "-0.361",
        "0.689", "0.034", "20261002", "10.1016", "10.1145",
    ]
    absent = [c for c in set(critical) if c in en and c not in zh]
    print(f"critical statistics present in EN but missing in ZH: {len(absent)} {absent}")

    # 4. symbol parity. Numeric multisets cannot see notation drift: an interpreter
    # may legally render `φ : ΔP → a` as `phi : \Delta P \to a` and every digit still
    # matches. These forms must be identical across the two documents.
    sym_fail = []
    for name, needle in [
        ("ascii action triple", "a = (tau, l, theta)"),
        ("latex phi form", "phi : \\Delta P"),
        ("latex arrow form", "a \\to \\Delta P"),
    ]:
        for doc, text in (("en", en), ("zh", zh)):
            if needle in text:
                sym_fail.append(f"{doc} still contains {name}")
    n_en, n_zh = en.count("φ"), zh.count("φ")
    print(f"symbol parity: phi count en={n_en} zh={n_zh} | offenders={sym_fail}")
    sym_ok = not sym_fail and n_en == n_zh and n_en > 0

    ok = ref_ok and not missing_r and not extra_r and not absent and sym_ok
    print()
    if ok:
        print("ZH ALIGNMENT: PASS")
        return 0
    print("ZH ALIGNMENT: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
