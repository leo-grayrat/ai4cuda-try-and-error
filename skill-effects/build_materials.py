"""Render traceable rule subsets; this does not invent growth histories."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = (ROOT / "sources/CUDA-Agent.SKILL.md").read_text(encoding="utf-8").splitlines()
    with (ROOT / "materials/cuda_rules.tsv").open(encoding="utf-8", newline="") as handle:
        rules = list(csv.DictReader(handle, delimiter="\t"))
    seen = set()
    for rule in rules:
        if rule["id"] in seen:
            raise ValueError("duplicate rule id")
        seen.add(rule["id"])
        quotes = []
        for span in rule["source_lines"].split(";"):
            bounds = list(map(int, span.split("-")))
            start, end = bounds[0], bounds[-1]
            if not 1 <= start <= end <= len(source):
                raise ValueError(f"invalid source span: {rule['id']}")
            quotes.append({"start": start, "end": end, "text": "\n".join(source[start - 1:end])})
        rule["source_evidence"] = quotes
    (ROOT / "materials/cuda_rules.json").write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")
    for condition, included in [("core", {"core"}), ("workflow", {"core", "workflow"}), ("condensed", {"core", "workflow", "condensed"})]:
        selected = [rule for rule in rules if rule["introduced_in"] in included]
        text = "# CUDA development guidance\n\n"
        for rule in selected:
            label = "Requirement" if rule["kind"] == "requirement" else "Conditional advice" if rule["kind"] == "recommendation" else "Example practice"
            text += f"- [{rule['id']}] {label}: {rule['text']}\n"
        (ROOT / f"materials/{condition}.md").write_text(text, encoding="utf-8")
    summary = {"source": "CUDA-Agent.SKILL.md", "rules": len(rules), "by_kind": {kind: sum(r["kind"] == kind for r in rules) for kind in sorted({r["kind"] for r in rules})}, "conditions": {stage: sum(r["introduced_in"] in stages for r in rules) for stage, stages in [("core", {"core"}), ("workflow", {"core", "workflow"}), ("condensed", {"core", "workflow", "condensed"})]}, "status": "first manual source mapping; requires coverage audit before formal freeze"}
    (ROOT / "materials/inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
