"""A code-first case must stay traceable to its actual parent and verifier."""

import json

from scripts.prepare_credit_case import prepare_case


def test_prepared_case_has_sources_manifest_and_unverified_actions(tmp_path) -> None:
    parent = (
        "import torch\n\n"
        "class ModelNew(torch.nn.Module):\n"
        "    def forward(self, x):\n        return x + 1\n"
    )
    child = parent.replace("return x + 1", "return x + 2")
    reference = "class Model: pass\n"
    out = tmp_path / "case"
    trace = prepare_case(
        parent_source=parent,
        child_source=child,
        reference_source=reference,
        out_dir=out,
        experiment_id="example",
        provenance={"archive": "sample.tar.gz", "task": "2_1", "parent_node": 1, "child_node": 2},
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    screen_manifest = json.loads((out / "screen-manifest.json").read_text(encoding="utf-8"))
    assert (out / "parent.py").read_text(encoding="utf-8") == parent
    assert (out / "child.py").read_text(encoding="utf-8") == child
    assert manifest["evaluator"] == "kernelbench"
    assert [item["id"] for item in screen_manifest["candidates"]] == ["parent", "child"]
    assert screen_manifest["workload"] == manifest["workload"]
    assert manifest["workload"]["backend"] == "cuda"
    assert {item["id"] for item in manifest["candidates"]} == {"parent", "child"}
    assert trace["actions"][0]["independence"] == "unverified"
    assert trace["actions"][0]["candidate_id"] == "parent"
    assert trace["actions"][0]["evidence"] is None
    assert trace["raw_response"] is None
    assert trace["response_tokens"] is None
    assert trace["source_sha256"]["parent"] != trace["source_sha256"]["child"]
    assert trace["candidate_sha256"]["parent"] == trace["source_sha256"]["parent"]


def test_prepared_case_writes_one_selective_revert_for_separate_action(tmp_path) -> None:
    parent = (
        "@triton.jit\ndef kernel(x, y):\n    y = x\n\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n        return x + 1\n"
    )
    child = parent.replace("y = x\n", "y = x + 1\n").replace("return x + 1", "return x + 2")
    trace = prepare_case(
        parent_source=parent,
        child_source=child,
        reference_source="class Model: pass\n",
        out_dir=tmp_path / "case",
        experiment_id="example",
        provenance={"task": "2_1", "parent_node": 1, "child_node": 2},
    )
    assert len(trace["actions"]) == 2
    manifest = json.loads((tmp_path / "case" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["candidates"]) == 4
    assert manifest["workload"]["backend"] == "triton"
    for action in trace["actions"]:
        candidate = tmp_path / "case" / f"without-{action['id']}.py"
        assert candidate.exists()
        assert candidate.read_text(encoding="utf-8") != child
