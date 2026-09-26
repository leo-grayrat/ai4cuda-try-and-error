import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import (
    child_environment, collect_opencode_usage, collect_usage, evaluate_trial,
    run_exit_code, stage_task, stop_process_tree, summarize_events,
    summarize_opencode_events,
)


def test_usage_counts_turns_and_keeps_cache_separate():
    events = [
        {"type": "turn.completed", "usage": {"input_tokens": 120, "cached_input_tokens": 20, "output_tokens": 30}},
        {"type": "turn.completed", "usage": {"input_tokens": 80, "cached_input_tokens": 50, "output_tokens": 10}},
    ]
    result = collect_usage(events)
    assert result == {"turns": 2, "input_tokens": 200, "cached_input_tokens": 70, "output_tokens": 40}


def test_missing_turn_usage_is_incomplete_not_zero():
    result = summarize_events([{"type": "turn.completed"}, {"type": "item.completed", "item": {"type": "command_execution"}}])
    assert result["status"] == "incomplete_trace"
    assert result["tool_events"] == 1
    assert result["usage"] is None


def test_recovered_transport_error_does_not_invalidate_completed_turn():
    events = [
        {"type": "error", "message": "Reconnecting"},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 2}},
    ]
    assert summarize_events(events)["status"] == "complete"


def test_stage_task_excludes_independent_evaluator(tmp_path):
    source = tmp_path / "task"
    source.mkdir()
    (source / "task.md").write_text("Optimize CUDA addition", encoding="utf-8")
    (source / "reference.py").write_text("def run(x): return x", encoding="utf-8")
    (source / "candidate.py").write_text("def run(x): return x", encoding="utf-8")
    (source / "evaluate.py").write_text("SECRET_TEST", encoding="utf-8")
    work = tmp_path / "work"
    record = stage_task(source, work)
    assert sorted(p.name for p in work.iterdir()) == ["candidate.py", "reference.py", "task.md"]
    assert record["task_sha256"] and record["candidate_sha256"]
    assert not (work / "evaluate.py").exists()


def test_evaluation_saves_independent_result(tmp_path):
    task = tmp_path / "task"
    trial = tmp_path / "trial"
    (trial / "work").mkdir(parents=True)
    task.mkdir()
    (trial / "work" / "candidate.py").write_text("answer = 42", encoding="utf-8")
    (task / "evaluate.py").write_text(
        "import json,sys,pathlib,os\n"
        "source = pathlib.Path(sys.argv[1]).read_text()\n"
        "print(json.dumps({'correct': '42' in source, 'samples_ms': [1.0], 'triton_cache': os.environ.get('TRITON_CACHE_DIR')}))\n",
        encoding="utf-8",
    )
    assert evaluate_trial(task, trial, sys.executable) == 0
    saved = json.loads((trial / "evaluation.json").read_text(encoding="utf-8"))
    assert saved["correct"] is True
    assert saved["triton_cache"] == str(trial / "triton-cache")


def test_evaluation_reports_incorrect_candidate_as_failure(tmp_path):
    task = tmp_path / "task"
    trial = tmp_path / "trial"
    (trial / "work").mkdir(parents=True)
    task.mkdir()
    (trial / "work" / "candidate.py").write_text("wrong", encoding="utf-8")
    (task / "evaluate.py").write_text(
        "import json\nprint(json.dumps({'correct': False, 'error': 'wrong answer'}))\n",
        encoding="utf-8",
    )
    assert evaluate_trial(task, trial, sys.executable) != 0
    assert json.loads((trial / "evaluation.json").read_text(encoding="utf-8"))["correct"] is False


def test_child_environment_uses_existing_codex_home_on_windows():
    env = child_environment({"USERPROFILE": "C:\\Users\\sample"}, windows=True)
    assert env["HOME"] == "C:\\Users\\sample"
    assert env["CODEX_HOME"] == "C:\\Users\\sample\\.codex"


def test_child_environment_uses_same_python_and_trial_cache(tmp_path):
    python = tmp_path / "venv" / "Scripts" / "python.exe"
    env = child_environment({"PATH": "system"}, windows=True, python_executable=python, cache_dir=tmp_path / "cache")
    assert env["PATH"].startswith(str(python.parent) + ";")
    assert env["TRITON_CACHE_DIR"] == str(tmp_path / "cache")


def test_failed_run_status_produces_nonzero_cli_exit():
    assert run_exit_code({"status": "complete"}) == 0
    assert run_exit_code({"status": "timeout"}) == 1
    assert run_exit_code({"status": "runner_failed"}) == 1


def test_opencode_sums_model_steps_and_separates_cache_and_reasoning():
    events = [
        {"type": "step_finish", "part": {"reason": "tool-calls", "cost": 0.01, "tokens": {
            "total": 133, "input": 100, "output": 10, "reasoning": 3,
            "cache": {"read": 20, "write": 0},
        }}},
        {"type": "step_finish", "part": {"reason": "stop", "cost": 0.02, "tokens": {
            "total": 72, "input": 2, "output": 5, "reasoning": 5,
            "cache": {"read": 60, "write": 0},
        }}},
    ]
    assert collect_opencode_usage(events) == {
        "steps": 2, "input_tokens": 182, "uncached_input_tokens": 102,
        "cached_input_tokens": 80, "cache_write_input_tokens": 0,
        "output_tokens": 23, "visible_output_tokens": 15,
        "reasoning_output_tokens": 8, "total_tokens": 205,
        "cost_usd": 0.03,
    }


def test_opencode_requires_terminal_step_and_counts_tool_feedback():
    steps = [{"type": "step_finish", "part": {"reason": "stop", "cost": 0, "tokens": {
        "total": 3, "input": 1, "output": 1, "reasoning": 0,
        "cache": {"read": 1, "write": 0},
    }}}]
    tool = {"type": "tool_use", "part": {"state": {"status": "completed", "output": "ok"}}}
    assert summarize_opencode_events([tool] + steps)["status"] == "complete"
    assert summarize_opencode_events([tool] + steps)["tool_events"] == 1
    assert summarize_opencode_events([tool] + steps)["tool_feedback_events"] == 1
    assert summarize_opencode_events([tool])["status"] == "incomplete_trace"


def test_timeout_stops_exact_windows_process_tree(monkeypatch):
    calls = []
    class Process:
        pid = 4321
        def poll(self):
            return None
        def kill(self):
            calls.append("fallback")
    def fake_run(command, **kwargs):
        calls.append(command)
        class Result:
            returncode = 0
        return Result()
    monkeypatch.setattr("study.subprocess.run", fake_run)
    stop_process_tree(Process(), windows=True)
    assert calls == [["taskkill.exe", "/PID", "4321", "/T", "/F"]]
