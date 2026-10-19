"""Controlled repetition pressure, separate from natural append history.

Repeat whole real appendices, never generate fictitious failures or new rules.
The original cumulative document remains an exact byte prefix. Line thresholds
select complete-copy doses; no blank-line padding or line reflow is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from runtime import digest


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--targets", type=int, nargs="+", default=[1000, 3000])
    args = parser.parse_args()
    source = ROOT / f"evidence/growth-{args.stage:02}/cumulative.md"
    original = source.read_bytes()
    source_text = original.decode("utf-8").replace("\r\n", "\n")
    base = (ROOT / "sources/CUDA-Agent.SKILL.md").read_text(encoding="utf-8").rstrip()
    assert source_text.startswith(base + "\n\n")
    appendix = source_text[len(base):].strip()
    source_bullets = sum(line.startswith("- ") for line in appendix.splitlines())
    assert source_bullets
    tokenizer = Tokenizer.from_file(str(ROOT.parent / ".cache/skill-effects-tokenizer/tokenizer.json"))
    out = ROOT / "materials/repeat-pressure"
    out.mkdir(exist_ok=True)
    manifest = {"status": "prespecified synthetic repetition dose; draft pending formal condition freeze",
                "generator_sha256": digest(Path(__file__)), "source": source.relative_to(ROOT).as_posix(),
                "source_sha256": digest(source), "original_source_bullet_entries": source_bullets,
                "new_source_bullets": 0, "distinct_requirements": "unchanged from source; not inferred from bullet count",
                "provenance": "Exact repetition of actual additions; not additional observed development episodes",
                "conditions": []}
    for target in args.targets:
        if target <= len(source_text.splitlines()):
            parser.error("pressure target must exceed source line count")
        data, copies = original, 1
        while len(data.decode("utf-8").splitlines()) < target:
            copies += 1
            data += (f"\n\n## Repeated appendix, copy {copies}\n\n" + appendix + "\n").encode("utf-8")
        assert data.startswith(original)
        path = out / f"stage-{args.stage:02}-at-least-{target}-lines.md"
        path.write_bytes(data)
        text = data.decode("utf-8").replace("\r\n", "\n")
        assert sum(line.startswith("- ") for line in text[len(base):].splitlines()) == source_bullets * copies
        manifest["conditions"].append({"target_lines": target, "actual_lines": len(text.splitlines()),
            "appendix_copies": copies, "source_bullet_occurrences": source_bullets * copies,
            "text_tokens": len(tokenizer.encode(text, add_special_tokens=False).ids),
            "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
