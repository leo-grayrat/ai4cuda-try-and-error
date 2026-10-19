"""Build reviewable growth/compression pairs, preserving the original skill.

These files remain draft materials until semantic review and the formal run
manifest freeze. Mechanical coverage only detects absent links, not lost meaning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build():
    spec_path = ROOT / "materials/growth_compression.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    clauses = spec["clauses"]
    assert len({c["id"] for c in clauses}) == len(clauses)
    inventory = []
    for stage in range(1, spec["stages"] + 1):
        path = ROOT / f"evidence/growth-{stage:02}/addition.md"
        number = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.startswith("- "):
                number += 1
                inventory.append({"id": f"G{stage:02}-{number:02}", "stage": stage,
                                  "source": path.relative_to(ROOT).as_posix(),
                                  "source_line": line_number, "text": line[2:]})
    ids = {item["id"] for item in inventory}
    links = {source for clause in clauses for source in clause["sources"]}
    assert ids == links, {"unmapped": sorted(ids - links), "unknown": sorted(links - ids)}
    # A clause cannot rely solely on instructions that appear in a later stage.
    for clause in clauses:
        assert min(int(source[1:3]) for source in clause["sources"]) == clause["stage"]
    base = (ROOT / "sources/CUDA-Agent.SKILL.md").read_bytes()
    tokenizer_path = ROOT.parent / ".cache/skill-effects-tokenizer/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    out = ROOT / "materials/growth-conditions"
    out.mkdir(exist_ok=True)
    manifest = {"status": spec["status"], "spec_sha256": sha(spec_path.read_bytes()),
                "base_sha256": sha(base), "source_bullet_count": len(inventory),
                "coverage_is_not_equivalence": True, "stages": []}
    for stage in range(1, spec["stages"] + 1):
        active = [clause for clause in clauses if clause["stage"] <= stage]
        addition = "\n\n## Consolidated additional development instructions\n\n" + spec["scope"] + "\n\n"
        addition += "\n\n".join(clause["text"] for clause in active) + "\n"
        compressed = base + addition.encode("utf-8")
        assert compressed[:len(base)] == base
        path = out / f"stage-{stage:02}-compressed.md"
        path.write_bytes(compressed)
        raw_path = ROOT / f"evidence/growth-{stage:02}/cumulative.md"
        raw = raw_path.read_bytes()
        row = {"stage": stage, "clauses": [c["id"] for c in active], "conditions": {}}
        for label, content, source_path in [("raw", raw, raw_path), ("compressed", compressed, path)]:
            text = content.decode("utf-8").replace("\r\n", "\n")
            row["conditions"][label] = {"path": source_path.relative_to(ROOT).as_posix(),
                "sha256": sha(content), "lines": len(text.splitlines()),
                "text_tokens": len(tokenizer.encode(text, add_special_tokens=False).ids)}
        manifest["stages"].append(row)
    for item in inventory:
        item["mapped_clauses"] = [c["id"] for c in clauses if item["id"] in c["sources"]]
    (out / "source-bullets.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    build()
