import json
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_task_staging_uses_the_real_frozen_model_and_records_its_contract(tmp_path):
    from agent_trial import stage
    work, control = stage(tmp_path, task="dev_rmsnorm")
    expected = Path(__file__).resolve().parents[1] / "tasks/dev_rmsnorm/model.py"
    assert (work / "model.py").read_bytes() == expected.read_bytes()
    assert (control / "model.py").read_bytes() == expected.read_bytes()
    assert (work / "contract.md").is_file()
    spec = json.loads((control / "task_spec.json").read_text())
    assert spec["split"] == "development"


def test_checker_rejects_stale_validation_and_keeps_submission_open(tmp_path):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=True)
    executor.files.write("kernels/f.cu", "__global__ void f() {}")
    executor.files.write("kernels/f_binding.cpp", "REGISTER_BINDING(f, register_f);")
    result = executor.execute({"operation": "finish", "report": "done"})
    assert result["finish_rejected"]
    assert result["validation_missing_or_stale"] == ["compile", "verify", "benchmark"]
    assert not executor.submitted


def test_document_only_submission_is_not_silently_given_a_checker(tmp_path):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=False)
    result = executor.execute({"operation": "finish", "report": "done"})
    assert result["submitted"]


def test_agent_can_read_the_log_filename_returned_by_its_tools(tmp_path):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=False)
    (tmp_path / "tool-001-compile.log").write_text("actual compiler diagnosis", encoding="utf-8")
    assert executor.execute({"operation": "read", "path": "tool-001-compile.log"})["content"] == "actual compiler diagnosis"


def test_writing_a_workspace_log_name_cannot_replace_the_real_tool_log(tmp_path):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=False)
    log = tmp_path / "tool-001-compile.log"
    log.write_text("real tool evidence", encoding="utf-8")
    executor.execute({"operation": "write", "path": log.name, "content": "fake log"})
    assert executor.execute({"operation": "read", "path": log.name})["content"] == "real tool evidence"
    assert log.read_text(encoding="utf-8") == "real tool evidence"


def test_document_only_can_submit_missing_code_for_independent_failure_scoring(tmp_path):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=False)
    executor.files.delete("model_new.py")
    assert executor.execute({"operation": "finish", "report": "done"})["submitted"]


def test_checker_reports_missing_code_without_crashing_the_tool(tmp_path):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=True)
    executor.files.delete("model_new.py")
    result = executor.execute({"operation": "finish", "report": "done"})
    assert result["finish_rejected"]
    assert "missing_model_new" in result["source_findings"]


@pytest.mark.parametrize("later_operation", ["compile", "verify", "benchmark"])
def test_failed_later_gate_cannot_reuse_earlier_success(tmp_path, monkeypatch, later_operation):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=True)
    executor.files.write("kernels/f.cu", "__global__ void f() {}")
    executor.files.write("kernels/f_binding.cpp", "REGISTER_BINDING(f, register_f);")
    build, artifact = executor.build_hash(), executor.artifact_hash()
    executor.compiled_hash = build
    executor.validated = {"compile": build, "verify": artifact, "benchmark": artifact}

    class FailedProcess:
        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr("agent_trial.subprocess.Popen", lambda *args, **kwargs: FailedProcess())
    assert not executor._process(later_operation)["ok"]
    result = executor.execute({"operation": "finish", "report": "done"})
    assert result["finish_rejected"]
    assert later_operation in result["validation_missing_or_stale"]


def test_verification_after_benchmark_requires_a_later_benchmark(tmp_path, monkeypatch):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=True)
    executor.files.write("kernels/f.cu", "__global__ void f() {}")
    executor.files.write("kernels/f_binding.cpp", "REGISTER_BINDING(f, register_f);")
    build, artifact = executor.build_hash(), executor.artifact_hash()
    executor.compiled_hash = build
    executor.validated = {"compile": build, "benchmark": artifact}

    class PassingProcess:
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("agent_trial.subprocess.Popen", lambda *args, **kwargs: PassingProcess())
    assert executor._process("verify")["ok"]
    result = executor.execute({"operation": "finish", "report": "done"})
    assert result["finish_rejected"]
    assert result["validation_missing_or_stale"] == ["benchmark"]


def test_formal_task_cannot_be_used_to_write_development_rules():
    from agent_trial import check_task_split
    with pytest.raises(ValueError):
        check_task_split("sigmoid", "development")
    assert check_task_split("dev_rmsnorm", "development")["split"] == "development"


def test_every_task_preserves_the_original_model_class():
    from prepare_tasks import ROOT, catalog, model_ast
    for task in catalog()["tasks"]:
        source = (ROOT / "sources/KernelBench" / task["source"]).read_text(encoding="utf-8")
        rendered = (ROOT / "tasks" / task["id"] / "model.py").read_text(encoding="utf-8")
        assert rendered.startswith(source)
        assert model_ast(rendered) == model_ast(source)


def test_task_manifest_hashes_actual_bytes_including_windows_newlines():
    from prepare_tasks import ROOT
    manifest = json.loads((ROOT / "tasks/manifest.json").read_text(encoding="utf-8"))
    for task in manifest["tasks"]:
        path = ROOT / "tasks" / task["id"] / "model.py"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == task["model_sha256"]


@pytest.mark.parametrize("payload", ['[]', '{"operation": []}', '{"path":"kernels/f.cu","content":"code"}', '{"operation":"write","path":3,"content":"code"}', '{"operation":"write","path":"kernels/f.cu","content":3}'])
def test_malformed_model_tool_requests_become_feedback_instead_of_runner_crashes(payload):
    from agent_trial import parse_tool_arguments
    with pytest.raises(ValueError):
        parse_tool_arguments(payload)
