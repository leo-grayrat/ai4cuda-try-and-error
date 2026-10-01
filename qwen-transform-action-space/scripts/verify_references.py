#!/usr/bin/env python3
"""Stage 4.5 -- independent reference verification, from scratch.

Pipeline rule: the final integrity check must not merely re-read the Stage 2.5
findings, because revision may have introduced new citations or dropped sources.
This script therefore re-resolves every identifier that appears in the manuscript's
reference list against the live arXiv Atom API and CrossRef, and compares the
returned metadata with what the reference entry claims.

Checks per reference:
  1. the identifier resolves (HTTP 200 + a record)
  2. the title in the entry is a case/whitespace-insensitive match of the returned
     title (prefix match allowed, because entries truncate long subtitles with a
     descriptive clause)
  3. author surnames listed in the entry all appear among the returned authors
     (skipped when an entry lists no authors)

Usage: python scripts/verify_references.py
Exit 0 only if every reference passes; otherwise 1 with a per-reference report.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "main.md"
OUT = ROOT / "stage4_5_final" / "reference_verification.json"
TIMEOUT = 40
ARXIV = "https://export.arxiv.org/api/query"
CROSSREF = "https://api.crossref.org/works"
UA = "stage45-integrity/1.0 (citation re-verification)"

REF_LINE = re.compile(r"^(\d+)\.\s+(.+)$", re.M)
ARXIV_ID = re.compile(r"arXiv:(\d{4}\.\d{4,5})", re.I)
# A DOI may contain parentheses (Elsevier: 10.1016/S0004-3702(99)00052-1), so the
# value is captured greedily and then stripped of trailing sentence punctuation.
# An earlier terminator class excluded ')', which truncated the Elsevier DOI at
# "(99" and produced a bogus 404 for reference 11.
DOI = re.compile(r"doi:(10\.\d{4,9}/\S+)", re.I)
# Non-archival scholarly sources (research-blog posts, position pieces) are allowed
# only if the URL resolves AND its page title matches the cited title. Citing a blog
# is not exempt from provenance checking; it just checks a different resolver.
URL = re.compile(r"url:(https?://\S+?)(?=[\s)]|$)", re.I)


def clean_doi(raw: str) -> str:
    d = raw.rstrip(".,;:)") if not raw.endswith(")") else raw.rstrip(".,;:")
    while d.endswith(")") and d.count("(") < d.count(")"):
        d = d[:-1]
    # Publisher DOI casing: Elsevier records are registered lower-case.
    return d.lower() if d.lower().startswith("10.1016") else d


def get(url: str, label: str):
    last = ""
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"{UA} {label}"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    print(f"  [{label}] FAILED: {last}", file=sys.stderr)
    return None


def norm(s: str) -> str:
    s = re.sub(r"\*\*", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return s.strip()


def entry_title(body: str) -> str:
    m = re.search(r"\*\*(.+?)\*\*", body, re.S)
    return m.group(1).strip() if m else ""


def entry_surnames(body: str) -> list[str]:
    """Surnames from a 'F. Last, A. Second, ...' lead-in before the bold title."""
    head = body.split("**")[0]
    out = []
    for token in re.split(r",|;|\band\b", head):
        token = token.strip().rstrip(".")
        if not token:
            continue
        parts = token.split()
        # "A. L. Zhang" -> Zhang ; "et al" -> skip ; "P.-L. Bacon" -> Bacon
        if parts[-1].lower() in ("et", "al", "jr", "sr"):
            continue
        if len(parts) == 1:
            out.append(parts[-1].lower())
        else:
            out.append(parts[-1].lower())
    return out


def arxiv_titles(ids: list[str]) -> dict[str, dict]:
    url = f"{ARXIV}?{urllib.parse.urlencode({'id_list': ','.join(ids), 'max_results': str(len(ids))})}"
    raw = get(url, "arxiv-batch")
    if raw is None:
        return {}
    import xml.etree.ElementTree as ET

    ns = {"a": "http://www.w3.org/2005/Atom"}
    res = {}
    for e in ET.fromstring(raw.decode("utf-8", "replace")).findall("a:entry", ns):
        eid = (e.findtext("a:id", "", ns) or "").strip()
        m = re.search(r"abs/(\d{4}\.\d{4,5})", eid)
        if not m:
            continue
        res[m.group(1)] = {
            "title": re.sub(r"\s+", " ", e.findtext("a:title", "", ns) or "").strip(),
            "authors": [
                (a.findtext("a:name", "", ns) or "").strip()
                for a in e.findall("a:author", ns)
            ],
        }
    return res


def crossref_record(doi: str) -> dict | None:
    raw = get(f"{CROSSREF}/{urllib.parse.quote(doi, safe='()')}", "crossref")
    if raw is None:
        return None
    m = json.loads(raw.decode("utf-8", "replace"))["message"]
    return {
        "title": (m.get("title") or [""])[0],
        "container": (m.get("container-title") or [""])[0],
        "authors": [
            a.get("family", "") or a.get("name", "") for a in m.get("author", [])
        ],
    }


def web_title(url: str) -> dict | None:
    """Fetch a non-archival source; return its <title> and full text for checks.

    Web pages carry no structured author list, so authors are verified by searching
    the page text for each cited surname. That is a real check, not an exemption:
    an entry naming an author the page does not mention still fails.
    """
    raw = get(url, "web")
    if raw is None:
        return None
    text = raw.decode("utf-8", "replace")
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    body = re.sub(r"<[^>]+>", " ", text)
    body = re.sub(r"\s+", " ", body)
    return {"title": title, "authors": [], "text": body}


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z\u00e0-\u024f]+", "", s.lower())


def main() -> int:
    text = PAPER.read_text(encoding="utf-8")
    cut = text.find("\n## References")
    if cut < 0:
        print("FATAL: no References section", file=sys.stderr)
        return 1
    refs = {int(n): b for n, b in REF_LINE.findall(text[cut:])}
    print(f"references listed: {len(refs)}")

    arxiv_ids = []
    for b in refs.values():
        m = ARXIV_ID.search(b)
        if m:
            arxiv_ids.append(m.group(1))
    ax = arxiv_titles(sorted(set(arxiv_ids)))
    print(f"arXiv ids resolved: {len(ax)}/{len(set(arxiv_ids))}")

    results = []
    for num, body in sorted(refs.items()):
        am = ARXIV_ID.search(body)
        dm = DOI.search(body)
        um = URL.search(body)
        rec, ident, kind = None, None, None
        if am:
            ident, kind, rec = am.group(1), "arXiv", ax.get(am.group(1))
        elif dm:
            ident, kind = clean_doi(dm.group(1)), "DOI"
            rec = crossref_record(ident)
        elif um:
            ident, kind = um.group(1), "URL"
            rec = web_title(ident)
        else:
            results.append({"ref": num, "verdict": "NO_IDENTIFIER", "detail": body[:90]})
            continue

        if not rec or not rec.get("title"):
            results.append({"ref": num, "verdict": "UNRESOLVED", "id": ident, "kind": kind})
            continue

        t_entry = entry_title(body)
        t_live = rec["title"]
        ne, nl = norm(t_entry), norm(t_live)
        if kind == "URL":
            # web page titles carry site suffixes ("... · Hazy Research") and the
            # entry may add a venue clause; require the cited title as a substring
            title_ok = bool(nl) and (ne in nl or nl in ne or ne.startswith(nl))
        else:
            # Entries may append a descriptive clause after the title, so allow the live
            # title to be a prefix of the entry title or vice versa. No venue-based
            # exemption is granted: a DOI that resolves to the proceedings record rather
            # than the paper must be reported, not waved through.
            title_ok = ne.startswith(nl) or nl.startswith(ne) or ne == nl

        surnames = [s.lower() for s in entry_surnames(body)]
        if kind == "URL":
            page = norm_name(rec.get("text", ""))
            missing = [s for s in surnames if norm_name(s) not in page]
        else:
            live_all = " ".join(rec["authors"]).lower()
            missing = [s for s in surnames if s and s not in live_all]
        authors_ok = (not surnames) or (not missing)

        verdict = "PASS" if (title_ok and authors_ok) else "MISMATCH"
        results.append(
            {
                "ref": num,
                "kind": kind,
                "id": ident,
                "verdict": verdict,
                "title_ok": title_ok,
                "authors_ok": authors_ok,
                "entry_title": t_entry[:90],
                "live_title": t_live[:90],
                "missing_surnames": missing,
            }
        )

    bad = [r for r in results if r["verdict"] != "PASS"]
    for r in bad:
        print(f"  !! ref {r['ref']}: {r['verdict']} -- {r.get('entry_title') or r.get('detail','')}")
        if r.get("live_title"):
            print(f"       live: {r['live_title']}")
        if r.get("missing_surnames"):
            print(f"       surnames not found: {r['missing_surnames']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"refs": results, "pass": len(results) - len(bad),
                               "total": len(results)}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"{len(results) - len(bad)}/{len(results)} references independently re-verified")
    if bad:
        print("FINAL INTEGRITY REFERENCES: FAIL")
        return 1
    print("FINAL INTEGRITY REFERENCES: PASS (100%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
