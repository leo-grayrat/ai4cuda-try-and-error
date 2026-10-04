#!/usr/bin/env python3
"""Generate paper/latex/references.bib mechanically from the manuscript.

Rationale: hand-retyping 32 reference entries into a .bib is exactly the
transcription step where citation drift enters. The reference list in
paper/main.md has already been re-resolved against arXiv/CrossRef by
verify_references.py, so the bib is DERIVED from that verified text instead of
re-entered. The numbered list in the manuscript body stays authoritative for
citation keys; this file exists so the LaTeX project compiles with a bibliography
if the author prefers bibtex over an embedded list.

Field extraction is deliberately conservative: what cannot be parsed reliably is
kept verbatim in the `note` field rather than guessed into a structured field.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "main.md"
OUT = ROOT / "paper" / "latex" / "references.bib"

REF_LINE = re.compile(r"^(\d+)\.\s+(.+)$", re.M)
ARXIV_ID = re.compile(r"arXiv:(\d{4}\.\d{4,5})", re.I)
DOI = re.compile(r"doi:(10\.\d{4,9}/\S+)", re.I)
URL_ID = re.compile(r"url:(https?://\S+)", re.I)
TITLE = re.compile(r"\*\*(.+?)\*\*", re.S)


def split_authors_and_venue(body: str) -> tuple[str, str]:
    """Text before the bold title is 'authors. venue-ish'; after it is venue/year."""
    m = TITLE.search(body)
    if not m:
        return "", body
    lead = body[: m.start()].strip().rstrip(".").strip()
    tail = body[m.end():].strip()
    # lead is "A. Author, B. Author, ... " -> convert to " and " separated for bib
    names = [n.strip() for n in re.split(r",(?![A-Z]\.)", lead) if n.strip()]
    return " and ".join(names), tail


def key_for(num: int, title: str) -> str:
    first = re.sub(r"[^a-zA-Z]", "", (title.split() or ["x"])[0]).lower()
    year = ""
    m = re.search(r"\b(19|20)\d{2}\b", title)
    return f"{first}{num}"


def escape(s: str) -> str:
    # keep the manuscript's own punctuation; only protect LaTeX specials that
    # actually appear in these titles
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def split_tail(tail: str) -> tuple[str, str]:
    """Split the post-title remainder into (venue, annotation).

    The manuscript writes `Authors. **Title.** Venue. arXiv:ID, YEAR. *annotation*`.
    Once the identifiers live in their own fields, repeating them inside `note` printed
    the same arXiv id and year twice per entry, so the note keeps only the annotation
    and the venue goes to howpublished.
    """
    m = re.search(r"\*([^*].*?)\*\s*$", tail, re.S)
    annotation = m.group(1).strip() if m else ""
    venue = (tail[: m.start()] if m else tail).strip()
    # drop bare identifier fragments now carried by eprint/doi/url
    venue = re.sub(r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?", "", venue)
    venue = re.sub(r"\bdoi:\s*10\.\d{4,9}/\S+", "", venue)
    venue = re.sub(r"\burl:\s*https?://\S+", "", venue)
    venue = re.sub(r"[,;]\s*(?:19|20)\d{2}\s*[.,]?", ", ", venue)
    venue = re.sub(r"\s{2,}", " ", venue).strip(" .,;-")
    return venue, annotation


def main() -> int:
    text = PAPER.read_text(encoding="utf-8")
    cut = text.find("\n## References")
    if cut < 0:
        raise SystemExit("no References section")
    refs = [(int(n), b) for n, b in REF_LINE.findall(text[cut:])]
    entries = []
    for num, body in refs:
        tm = TITLE.search(body)
        # a trailing period inside the bold title rendered as "Models., 2026" once the
        # style appended its own punctuation
        title = tm.group(1).strip().rstrip(".").strip() if tm else body[:80]
        authors, tail = split_authors_and_venue(body)
        am = ARXIV_ID.search(body)
        dm = DOI.search(body)
        um = URL_ID.search(body)
        kind = "misc"
        # double braces on the title: BibTeX sentence-cases the title field, which was
        # silently turning "Multi-Turn RL for Generating CUDA Kernels" into
        # "Multi-turn rl for generating cuda kernels" in the rendered bibliography
        fields = [f"  title         = {{{{{escape(title)}}}}}"]
        if authors:
            fields.append(f"  author        = {{{escape(authors)}}}")
        # a clickable target: unsrtnat.bst prints the url field, and the pandoc template
        # loads hyperref, so \url becomes an actual link in the PDF. Without this the
        # identifiers appear only as plain text inside `note` and cannot be followed.
        link = ""
        if am:
            fields.append(f"  archivePrefix = {{arXiv}}")
            fields.append(f"  eprint        = {{{am.group(1)}}}")
            fields.append("  primaryClass  = {cs.LG}")
            link = f"https://arxiv.org/abs/{am.group(1)}"
        elif dm:
            d = dm.group(1).rstrip(".,;)")
            fields.append(f"  doi           = {{{d}}}")
            link = f"https://doi.org/{d}"
            kind = "article" if re.search(r"Artificial Intelligence|Journal", tail) else (
                "inproceedings" if re.search(r"PLDI|OSDI|ICML|NeurIPS|ICLR|CGO|ACL|AAAI|CHI", tail)
                else "article"
            )
        elif um:
            link = um.group(1).rstrip(".,;)")
        if link:
            fields.append(f"  url           = {{{link}}}")
        venue, annotation = split_tail(tail)
        if venue:
            fields.append(f"  howpublished  = {{{escape(venue)}}}")
        # the manuscript's per-entry annotation is part of the verified record (what we
        # checked the source against), so it is carried in full rather than truncated
        if annotation:
            fields.append(f"  note          = {{{escape(annotation.replace('*', ''))}}}")
        ym = re.search(r"\b(19|20)\d{2}\b", tail)
        if ym:
            fields.append(f"  year          = {{{ym.group(0)}}}")
        entries.append(f"@{kind}{{{key_for(num, title)},\n" + ",\n".join(fields) + "\n}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # an entry without authors renders as a bare title in the bibliography and is easy to
    # miss in a hand-typed list; report it on every build so it cannot hide
    noauth = [num for num, body in refs
              if not split_authors_and_venue(body)[0].strip()]
    if noauth:
        print(f"WARNING: {len(noauth)} reference entries have no author segment: "
              f"{noauth}")
    header = (
        "% Generated by scripts/make_bib.py from paper/main.md -- do not hand-edit.\n"
        "% Regenerate: python scripts/make_bib.py\n"
        "% Provenance: every entry here was re-resolved against the live arXiv Atom API\n"
        "% or CrossRef by scripts/verify_references.py before this file was written.\n"
    )
    OUT.write_text(header + "\n\n".join(entries) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
