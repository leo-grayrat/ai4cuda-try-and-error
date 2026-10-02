#!/usr/bin/env python3
"""Fetch the missing author lists for the eight reference entries that have none.

Two channels, tried in order, because the previous per-id loop exhausted the arXiv API
rate limit: one batched id_list query, then the abstract page's citation_author meta
tags for anything still missing. Prints results only; a human pastes them into
paper/main.md, and verify_references.py re-checks them against the API afterwards.

Usage: python scripts/fetch_authors.py
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IDS = {
    20: "2606.20128",
    21: "2608.12700",
    22: "2607.16241",
    23: "2607.14541",
    24: "2609.22220",
    25: "2609.21058",
    26: "2609.32444",
    28: "2607.18171",
}
UA = {"User-Agent": "citation-metadata-check/1.0 (bibliographic verification)"}
NS = {"a": "http://www.w3.org/2005/Atom"}


def initials(full: str) -> str:
    parts = full.split()
    if len(parts) == 1:
        return parts[0]
    return " ".join(p[0] + "." for p in parts[:-1]) + " " + parts[-1]


def join(names: list[str]) -> str:
    fmt = [initials(n) for n in names]
    if len(fmt) == 1:
        return fmt[0]
    return ", ".join(fmt[:-1]) + ", and " + fmt[-1]


def via_api(ids: list[str]) -> dict[str, list[str]]:
    url = "https://export.arxiv.org/api/query?id_list=" + ",".join(ids)
    req = urllib.request.Request(url, headers=UA)
    root = ET.fromstring(urllib.request.urlopen(req, timeout=60).read())
    out: dict[str, list[str]] = {}
    for e in root.findall("a:entry", NS):
        raw = e.findtext("a:id", "", NS) or ""
        m = re.search(r"(\d{4}\.\d{4,5})", raw)
        if not m:
            continue
        names = [x.findtext("a:name", "", NS) for x in e.findall("a:author", NS)]
        names = [n for n in names if n and n.lower() != "arxiv"]
        if names:
            out[m.group(1)] = names
    return out


def via_abs(aid: str) -> list[str]:
    html = urllib.request.urlopen(
        urllib.request.Request(f"https://arxiv.org/abs/{aid}", headers=UA), timeout=40
    ).read().decode("utf-8", "replace")
    names = re.findall(r'name="citation_author"\s+content="([^"]+)"', html)
    # citation_author is "Last, First"; normalise to "First Last" for initials()
    out = []
    for n in names:
        if "," in n:
            last, first = n.split(",", 1)
            out.append(f"{first.strip()} {last.strip()}")
        elif n.strip():
            out.append(n.strip())
    return out


def main() -> int:
    ids = list(IDS.values())
    got: dict[str, list[str]] = {}
    for attempt in range(3):
        try:
            got.update(via_api(ids))
            break
        except Exception as e:  # noqa: BLE001
            print(f"  api attempt {attempt}: {type(e).__name__}", file=sys.stderr)
            time.sleep(8)

    missing = [i for i in ids if i not in got or len(got.get(i, [])) < 2]
    if missing:
        print(f"api gave {len(ids) - len(missing)}/{len(ids)}; trying abs pages for the rest")
        for aid in missing:
            try:
                names = via_abs(aid)
                if len(names) >= len(got.get(aid, [])):
                    got[aid] = names
            except Exception as e:  # noqa: BLE001
                print(f"  {aid}: abs failed {type(e).__name__}", file=sys.stderr)
            time.sleep(1.2)

    print("\n=== author strings for paper/main.md ===")
    for num, aid in IDS.items():
        names = got.get(aid, [])
        if not names:
            print(f"{num}. [{aid}] STILL MISSING")
        else:
            # full names are printed too: an initials-only string hid the fact that one
            # abs page produced "Max, Fan" where the real author was a single person
            # named Max Fan, i.e. a split that would have been pasted into the paper
            print(f"{num}. {join(names)}  ## {aid} ({len(names)} authors)")
            print(f"      full: {' | '.join(names)}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
