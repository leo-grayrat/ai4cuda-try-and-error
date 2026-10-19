"""Synthetic evidence integrity checks, not experimental observations."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_blind_review import packet
from runtime import digest


def test_packet_hides_reasoning_labels_and_paths_without_inventing_a_report(tmp_path, monkeypatch):
    monkeypatch.setattr("prepare_blind_review.ROOT", tmp_path)
    run = tmp_path / "runs/f01-000"
    for directory in [run / "control", run / "work/kernels", tmp_path / "tasks/fixture"]:
        directory.mkdir(parents=True, exist_ok=True)
    (run / "control/model.py").write_text("reference", encoding="utf-8")
    (tmp_path / "tasks/fixture/contract.md").write_text("frozen contract", encoding="utf-8")
    (run / "work/model_new.py").write_text("candidate", encoding="utf-8")
    (run / "work/kernels/attempt.cu.bak").write_text("uncompiled backup", encoding="utf-8")
    (run / "response-000.json").write_text(json.dumps({"choices": [{"message": {
        "content": "interim visible comment", "reasoning_content": "PRIVATE_REASONING"}}]}), encoding="utf-8")
    result = {"state": "generation_truncated", "final_evaluation": {}, "source_findings": {},
              "changed_infrastructure": []}
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    events = [{"type": "response", "index": 0}, {"type": "tool", "call": {"function": {
        "name": "workspace", "arguments": json.dumps({"operation": "read", "path": str(run / "work/model_new.py")})}},
        "result": {"output": str(run / "work/model_new.py")}}]
    (run / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    row = {"task": "fixture", "state": "generation_truncated", "submitted": False,
           "condition": "SHOULD_NOT_BE_EXPORTED", "result_sha256": digest(run / "result.json")}
    data = packet(run, row, "R-opaque", {"dimensions": []})
    exported = json.dumps(data)
    assert "PRIVATE_REASONING" not in exported
    assert "SHOULD_NOT_BE_EXPORTED" not in exported
    assert "f01-000" not in exported
    assert str(tmp_path).replace("\\", "\\\\") not in exported
    assert data["submitted_report"] is None
    assert not data["report_available"]
    assert data["visible_timeline"][0]["text"] == "interim visible comment"
    assert data["task_contract"] == "frozen contract"
    assert data["visible_timeline"][0]["response_index"] == 0
    assert "kernels/attempt.cu.bak" in {item["path"] for item in data["final_inventory"]}
    assert data["final_source_sha256"]["model_new.py"] == digest(run / "work/model_new.py")
