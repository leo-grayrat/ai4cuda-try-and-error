"""Derive treatments from the frozen upstream CUDA Agent skill only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer

from runtime import digest


ROOT = Path(__file__).resolve().parent
FENCES = re.compile(r"(?ms)^```[^\n]*\n.*?^```[ \t]*$")


def main():
    source = ROOT / "sources/CUDA-Agent.SKILL.md"
    original = source.read_text(encoding="utf-8")
    examples = FENCES.findall(original)
    rules = json.loads((ROOT / "materials/cuda_rules.json").read_text(encoding="utf-8"))
    assert len(examples) == 9 and len(rules) == 45
    out = ROOT / "materials/original-conditions"
    out.mkdir(exist_ok=True)
    tokenizer = Tokenizer.from_file(str(ROOT.parent / ".cache/skill-effects-tokenizer/tokenizer.json"))

    def count(text):
        return len(tokenizer.encode(text.replace("\r\n", "\n"), add_special_tokens=False).ids)

    def render(selected):
        text = "# CUDA development guidance\n\n"
        for rule in selected:
            label = {"requirement": "Requirement", "recommendation": "Conditional advice", "example": "Example practice"}[rule["kind"]]
            text += f"- [{rule['id']}] {label}: {rule['text']}\n\n"
        return text + "## Original workspace and code examples\n\n" + "\n\n".join(examples) + "\n"

    # Whitespace-only formatting, outside code fences. Word order and examples
    # remain identical; this is distinct from semantic consolidation.
    pieces, position = [], 0
    for match in FENCES.finditer(original):
        outside = original[position:match.start()]
        pieces.append("\n".join(line.rstrip() + ("\n" if line.startswith("- ") else "") for line in outside.splitlines()) + "\n")
        pieces.append(match.group() + "\n")
        position = match.end()
    pieces.append("\n".join(line.rstrip() + ("\n" if line.startswith("- ") else "") for line in original[position:].splitlines()) + "\n")
    formatted = "".join(pieces)
    assert re.findall(r"\S+", original) == re.findall(r"\S+", formatted)
    assert FENCES.findall(formatted) == examples
    texts = {"formatted": formatted, "optimized": render(rules),
             "core": render([r for r in rules if r["introduced_in"] == "core"]),
             "workflow": render([r for r in rules if r["introduced_in"] in {"core", "workflow"}])}
    removed = {"C05", "C06", "C07"}
    texts["optimized_residual"] = render([r for r in rules if r["id"] not in removed])
    # Match the optimized version to the original's text-token length using
    # declarative platform facts already common to every task prompt.
    target = count(original)
    optimized = texts["optimized"]
    assert count(optimized) < target
    background = ("The experimental platform is Windows with an RTX 5060 Laptop GPU. The CUDA architecture is sm_120. "
                  "The CUDA Toolkit version is 12.9 and the compiler is MSVC. PyTorch is 2.11.0+cu128. "
                  "The extension is named cuda_extension. The workspace contains model.py, model_new.py, kernels and fixed binding infrastructure.\n\n")
    heading = "\n## Platform background\n\n"
    carrier = background * 40
    low, high = 0, len(carrier)
    while high - low > 1:
        mid = (high + low) // 2
        if count(optimized + heading + carrier[:mid]) <= target:
            low = mid
        else:
            high = mid
    distance, length = min((abs(count(optimized + heading + carrier[:n]) - target), n)
                          for n in range(max(0, low - 32), min(len(carrier), high + 32) + 1))
    assert distance <= 1
    texts["length_control"] = optimized + heading + carrier[:length]
    manifest = {"source": source.relative_to(ROOT).as_posix(), "source_sha256": digest(source),
        "rule_map_sha256": digest(ROOT / "materials/cuda_rules.json"),
        "source_text_tokens": target, "fenced_example_count": len(examples),
        "normalized_example_sha256": [hashlib.sha256(block.encode()).hexdigest() for block in examples],
        "basis": "Frozen upstream CUDA Agent skill only; no constructed failure-driven additions or synthetic repetition pressure.",
        "formatted_check": "Identical non-whitespace token sequence and identical fenced examples",
        "optimized_check": "All 45 source-mapped items, including their requirement/advice/example status, plus all original fenced examples",
        "subset_interpretation": "Core/workflow vary explicit items from the same source. Original examples are shared and convey additional implicit practices; these are not pure counts of independent demands.",
        "residual_removed_items": sorted(removed),
        "residual_scope": "Only explicit immutable-file reminders removed. Examples and all non-mechanized workflow/explanation duties retained.",
        "conditions": {}}
    for name, text in texts.items():
        assert FENCES.findall(text) == examples
        path = out / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        manifest["conditions"][name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
            "lines": len(text.splitlines()), "text_tokens": count(text),
            "explicit_rule_ids": [r["id"] for r in rules if name not in {"core", "workflow", "optimized_residual"}
                                  or (name == "core" and r["introduced_in"] == "core")
                                  or (name == "workflow" and r["introduced_in"] in {"core", "workflow"})
                                  or (name == "optimized_residual" and r["id"] not in removed)]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"original_tokens": target, "conditions": manifest["conditions"]}))


if __name__ == "__main__":
    main()
