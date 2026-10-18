import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_usage_preserves_cache_and_reasoning_without_double_counting():
    from runtime import summarize_usage
    rows = [{"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
             "prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 20,
             "completion_tokens_details": {"reasoning_tokens": 12}}]
    result = summarize_usage(rows)
    assert result["input_tokens"] == 100
    assert result["uncached_input_tokens"] == 20
    assert result["cached_input_tokens"] == 80
    assert result["output_tokens"] == 30
    assert result["reasoning_tokens"] == 12
    assert result["total_tokens"] == 130
    with pytest.raises(ValueError):
        summarize_usage([{**rows[0], "total_tokens": 131}])
    with pytest.raises(ValueError):
        summarize_usage([{**rows[0], "prompt_cache_hit_tokens": 81}])


def test_tool_permissions_deny_writes_but_keep_source_readable(tmp_path):
    from runtime import FileTools
    (tmp_path / "model.py").write_text("reference")
    tools = FileTools(tmp_path, permissions=True)
    assert tools.read("model.py") == "reference"
    with pytest.raises(PermissionError):
        tools.write("model.py", "replacement")
    assert (tmp_path / "model.py").read_text() == "reference"
    tools.write("kernels/fused.cu", "__global__ void run() {}")
    tools.write("model_new.py", "candidate")
    assert (tmp_path / "model_new.py").read_text() == "candidate"
    tools.delete("kernels/fused.cu")
    with pytest.raises(PermissionError):
        tools.delete("model.py")


@pytest.mark.parametrize("name", ["../elsewhere.py", "C:/Windows/x", "file:stream", "kernels/../../x", "kernels\\..\\..\\x", "CON", "kernels/NUL.cu"])
def test_all_conditions_reject_paths_outside_the_trial(tmp_path, name):
    from runtime import FileTools
    for permissions in [False, True]:
        with pytest.raises((ValueError, PermissionError)):
            FileTools(tmp_path, permissions).write(name, "bad")


def test_document_only_condition_really_allows_reference_edits(tmp_path):
    from runtime import FileTools
    tools = FileTools(tmp_path, permissions=False)
    tools.write("model.py", "changed")
    assert tools.read("model.py") == "changed"


def test_static_check_finds_real_forward_ops_not_comments(tmp_path):
    from runtime import source_checks
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels/f.cu").write_text("__global__ void f() {}")
    (tmp_path / "kernels/f_binding.cpp").write_text("REGISTER_BINDING(f, register_f);")
    model = tmp_path / "model_new.py"
    model.write_text("import torch\nimport cuda_extension\nclass ModelNew:\n def forward(self,x):\n  # torch.relu(x) is not executed\n  return cuda_extension.f(x)\n")
    assert "python_compute" not in source_checks(tmp_path)
    model.write_text("import torch as t\nclass ModelNew:\n def forward(self,x):\n  return t.relu(x)\n")
    assert "python_compute" in source_checks(tmp_path)


def test_checker_does_not_pretend_static_checks_prove_numerical_correctness(tmp_path):
    from runtime import source_checks
    failures = source_checks(tmp_path)
    assert "missing_model_new" in failures
    assert "missing_cuda_kernel" in failures
    assert "missing_binding" in failures
    assert "numerical_correctness" not in failures


def test_staging_keeps_reference_but_never_supplies_a_solved_candidate(tmp_path):
    from agent_trial import stage
    work, control = stage(tmp_path)
    assert (control / "model.py").read_bytes() == (work / "model.py").read_bytes()
    assert not (control / "model_new.py").exists()
    assert not list((work / "kernels").iterdir())
    assert "NotImplementedError" in (work / "model_new.py").read_text()


def test_executor_temp_and_tmp_share_a_real_directory(tmp_path):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=False)
    assert executor.env["TEMP"] == executor.env["TMP"]
    assert Path(executor.env["TEMP"]).is_dir()


def test_non_utf8_compiler_output_remains_feedback_not_a_tool_exception(tmp_path):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=False)
    script = executor.control / "utils/compile.py"
    script.write_text("import os,sys\nos.write(1, b'compiler: \\xd3\\xef\\xb7\\xa8 error')\nsys.exit(1)\n", encoding="utf-8")
    result = executor.process("compile")
    assert not result["ok"]
    assert result["returncode"] == 1
    assert "compiler:" in result["output"]
    assert "error" in result["output"]


def test_legal_tensor_metadata_guards_are_not_compute_violations(tmp_path):
    from runtime import source_checks
    (tmp_path / "model_new.py").write_text("import torch\nclass ModelNew:\n def forward(self,x):\n  if x.dtype is not torch.float32 or x.shape[0] % 4 != 0:\n   return x\n  return x\n")
    assert "python_compute" not in source_checks(tmp_path)


def test_direct_tensor_arithmetic_is_detected(tmp_path):
    from runtime import source_checks
    (tmp_path / "model_new.py").write_text("class ModelNew:\n def forward(self,x):\n  return x * 2\n")
    assert "python_compute" in source_checks(tmp_path)


def test_compiler_feedback_preserves_errors_before_long_include_lists():
    from runtime import compiler_feedback
    output = "binding.cpp(20): error C2664: wrong launcher argument\n" + "note: including file header.h\n" * 3000 + "ninja: build stopped\n"
    result = compiler_feedback(output)
    assert "error C2664: wrong launcher argument" in result
    assert "ninja: build stopped" in result
    assert len(result) <= 6000


def test_python_runtime_config_changes_do_not_force_native_recompilation(tmp_path):
    from agent_trial import Executor
    executor = Executor(tmp_path, permissions=False, checker=False)
    before = executor.build_hash()
    executor.files.write("model_new.py", "# runtime config changed\n")
    assert executor.build_hash() == before
    executor.files.write("kernels/new.cu", "__global__ void f() {}")
    assert executor.build_hash() != before
