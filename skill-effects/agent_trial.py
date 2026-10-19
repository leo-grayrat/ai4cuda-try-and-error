"""Observable CUDA C++ agent loop shared by all experimental conditions.

Trials call the real DeepSeek API and compile/execute real native CUDA code.
No shell tool is offered. Credentials are loaded locally and never recorded.
Development trials are not formal effectiveness samples.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import requests

from environment_probe import ROOT, msvc_environment
from runtime import FileTools, compiler_feedback, digest, source_checks, summarize_usage
from prepare_tasks import catalog
from gpu_lock import gpu_lock

TOOL = {"type": "function", "function": {
    "name": "workspace",
    "description": "Operate on the CUDA extension workspace. read/write/delete use relative paths; write replaces a whole file. compile builds all C++/CUDA sources. verify runs numerical comparison. benchmark compares candidate with eager and torch.compile. finish submits the current files and report. No arbitrary shell is available.",
    "parameters": {"type": "object", "properties": {
        "operation": {"type": "string", "enum": ["list", "read", "write", "delete", "compile", "verify", "benchmark", "finish"]},
        "path": {"type": "string"}, "content": {"type": "string"},
        "report": {"type": "string", "description": "Final explanation for finish."},
        "label": {"type": "string", "description": "Optional descriptive name for a compile, verification or benchmark experiment."},
    }, "required": ["operation"]},
}}

BASE_PROMPT = """Accelerate the PyTorch Model in model.py by implementing a CUDA C++ extension and a compatible ModelNew in model_new.py. The CUDA extension is named cuda_extension; binding.cpp and binding_registry.h provide automatic registration for kernels/*_binding.cpp. The original Model constructor and state_dict must work with ModelNew. The objective is correct output and candidate latency <= 0.95 times the torch.compile baseline, with the best performance attainable within the available budget.

Implement tensor computation in raw CUDA kernels, with cuBLAS permitted for GEMM and cuDNN for convolution. Tensor allocation and properties are allowed; PyTorch tensor computation in Python or C++ does not implement the requested custom kernel task. Numerical tolerance is atol=1e-2, rtol=1e-2.

Platform: Windows, RTX 5060 Laptop (sm_120), PyTorch 2.11.0+cu128, CUDA Toolkit 12.9, MSVC. Use the provided compile/verify/benchmark tools as the equivalents of the Linux shell commands in any supplied skill; the tools set architecture and compiler flags. This platform note is the same for all conditions. Benchmark uses CUDA events because the upstream CUPTI profiler is unavailable here.

For Windows cuBLAS bindings, the installed CUDA toolkit provides cublas_v2.h and cublas.lib. A binding may use #pragma comment(lib, "cublas.lib") under _MSC_VER with the fixed compiler tool. No change to compile infrastructure is needed.

The workspace begins with model.py, model_new.py, binding.cpp, binding_registry.h and utils/. Use workspace tools to inspect and implement the solution. Finish by calling finish with your final report. No text response alone submits the task. You have {steps} model requests and at most {output_budget} generated tokens in total, including reasoning. No new libraries can be installed through these tools.

Every workspace call needs an operation. A write uses operation="write", path and content as separate fields. Log filenames returned by compile, verify and benchmark can be read with operation="read"; tool evidence logs are read-only.
"""


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_tool_arguments(raw: str) -> dict:
    if not isinstance(raw, str):
        raise ValueError("workspace arguments must be a JSON object encoded as text")
    arguments = json.loads(raw)
    if not isinstance(arguments, dict):
        raise ValueError("workspace arguments must be an object")
    operation = arguments.get("operation")
    allowed = {"list", "read", "write", "delete", "compile", "verify", "benchmark", "finish"}
    if not isinstance(operation, str) or operation not in allowed:
        raise ValueError("operation must be one of: " + ", ".join(sorted(allowed)))
    for field in (["path", "content"] if operation == "write" else ["path"] if operation in {"read", "delete"} else []):
        if not isinstance(arguments.get(field), str):
            raise ValueError(f"{operation} requires a string {field}")
    for field in ["label", "report"]:
        if field in arguments and not isinstance(arguments[field], str):
            raise ValueError(f"{field} must be text")
    return arguments


def check_task_split(task: str, purpose: str):
    if task == "axpby":
        return {"id": "axpby", "split": "development", "cases": [{"shape": [1, 128], "alpha": 2.0}, {"shape": [7, 257], "alpha": -0.7}, {"shape": [0, 257], "alpha": 1.3}]}
    spec = next((row for row in catalog()["tasks"] if row["id"] == task), None)
    if spec is None:
        raise ValueError("unknown task")
    if purpose == "development" and spec["split"] != "development":
        raise ValueError("held-out tasks cannot supply development episodes or rule-writing feedback")
    return spec


def stage(run: Path, task: str = "axpby") -> tuple[Path, Path]:
    work, control = run / "work", run / "control"
    shutil.copytree(ROOT / "sources/CUDA-Agent", work)
    shutil.copytree(ROOT / "sources/CUDA-Agent", control)
    (control / "model_new.py").unlink()
    shutil.copy2(ROOT / "event_timing_probe.py", control / "event_timing_probe.py")
    shutil.copy2(ROOT / "verify_candidate.py", control / "verify_candidate.py")
    spec = check_task_split(task, "formal")
    if task != "axpby":
        for directory in [work, control]:
            shutil.copy2(ROOT / "tasks" / task / "model.py", directory / "model.py")
        shutil.copy2(ROOT / "tasks" / task / "contract.md", work / "contract.md")
    else:
        (work / "contract.md").write_text("Implement AXPBY for contiguous float32 CUDA tensors, matching the Model constructor. Support different random inputs and non-default streams for these cases: " + json.dumps(spec["cases"]) + ". Do not modify inputs; previously returned outputs must remain unchanged after later calls.\n", encoding="utf-8")
    save(control / "task_spec.json", spec)
    # Start without an already-solved candidate; keep the unchanged upstream task.
    for path in (work / "kernels").iterdir():
        path.unlink()
    (work / "model_new.py").write_text("import torch.nn as nn\n\nclass ModelNew(nn.Module):\n    def __init__(self, *args, **kwargs):\n        super().__init__()\n    def forward(self, *inputs):\n        raise NotImplementedError('Implement CUDA extension')\n", encoding="utf-8")
    compile_path = control / "utils/compile.py"
    text = compile_path.read_text(encoding="utf-8")
    if os.name == "nt":
        text = text.replace("cuda_extension.so", "cuda_extension.pyd").replace("extra_cflags=['-O3', '-std=c++17']", "extra_cflags=['/O2', '/std:c++17']")
        text = text.replace("import torch.utils.cpp_extension as cpp_ext", "import torch.utils.cpp_extension as cpp_ext\ncpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace')")
    compile_path.write_text(text, encoding="utf-8")
    shutil.copy2(compile_path, work / "utils/compile.py")
    return work, control


class Executor:
    def __init__(self, run: Path, permissions: bool, checker: bool, task: str = "axpby"):
        self.run = run
        self.work, self.control = stage(run, task=task)
        self.files = FileTools(self.work, permissions)
        self.checker = checker
        self.env = msvc_environment()
        for key, name in [("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"), ("TEMP", "temp"), ("TMP", "temp")]:
            folder = run / name
            folder.mkdir(exist_ok=True)
            self.env[key] = str(folder)
        self.frozen = {str(p.relative_to(self.work)): digest(p) for p in self.work.rglob("*") if p.is_file() and p.name != "model_new.py"}
        self.frozen_control = {str(p.relative_to(self.control)): digest(p) for p in self.control.rglob("*") if p.is_file()}
        self.count = 0
        self.submitted = False
        self.compiled_hash = None
        self.validated = {}

    def integrity(self):
        changed = [name for name, expected in self.frozen.items() if not (self.work / name).is_file() or digest(self.work / name) != expected]
        changed += ["control/" + name for name, expected in self.frozen_control.items() if not (self.control / name).is_file() or digest(self.control / name) != expected]
        return changed

    def build_hash(self):
        import hashlib
        rows = [(str(p.relative_to(self.work)), digest(p)) for p in self.work.rglob("*") if p.is_file() and p.suffix in {".cu", ".cpp", ".h", ".cuh"} and "build" not in p.relative_to(self.work).parts]
        return hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()

    def artifact_hash(self):
        import hashlib
        return hashlib.sha256((self.build_hash() + digest(self.work / "model_new.py")).encode()).hexdigest()

    def process(self, operation: str):
        if operation == "compile":
            return self._process(operation)
        with gpu_lock(ROOT / "runs/gpu.lock") as waited:
            result = self._process(operation)
        return {**result, "gpu_wait_seconds": waited}

    def _process(self, operation: str):
        # A later failed attempt cannot inherit an earlier pass. Repeating an
        # earlier gate also requires the subsequent gates to run again in order.
        subsequent = {"compile": ("compile", "verify", "benchmark"),
                      "verify": ("verify", "benchmark"), "benchmark": ("benchmark",)}
        for name in subsequent[operation]:
            self.validated.pop(name, None)
        if operation != "compile" and self.compiled_hash != self.build_hash():
            return {"ok": False, "error": "Compile the current candidate before verification or benchmarking."}
        if operation == "compile":
            command = [sys.executable, str(self.control / "utils/compile.py")]
        else:
            setup = f"import sys,runpy; sys.path.insert(0,{str(self.work)!r}); sys.path.insert(0,{str(self.control)!r}); "
            if operation == "verify":
                command = [sys.executable, str(self.control / "verify_candidate.py"), "--work", str(self.work), "--control", str(self.control)]
            else:
                command = [sys.executable, "-c", setup + f"import utils.profiling; runpy.run_path({str(self.control / 'event_timing_probe.py')!r},run_name='__main__')"]
        log = self.run / f"tool-{self.count:03d}-{operation}.log"
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=self.work, env=self.env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=stream, stderr=subprocess.STDOUT, check=False)
                process.wait()
                code = None
        output = log.read_text(encoding="utf-8", errors="replace")
        changed = self.integrity()
        ok = code == 0 and not changed
        if operation == "compile":
            self.compiled_hash = self.build_hash() if ok else None
        if ok:
            self.validated[operation] = self.build_hash() if operation == "compile" else self.artifact_hash()
        return {"ok": ok, "returncode": code, "seconds": time.monotonic() - started, "log": log.name, "output": compiler_feedback(output), "changed_infrastructure": changed}

    def execute(self, args: dict):
        self.count += 1
        operation = args["operation"]
        if operation == "list":
            return {"files": self.files.list()}
        if operation == "read":
            name = args["path"]
            if re.fullmatch(r"tool-\d+-(?:compile|verify|benchmark)\.log", name):
                log = self.run / name
                return {"content": log.read_text(encoding="utf-8", errors="replace")[:18000]}
            return {"content": self.files.read(args["path"])[:18000]}
        if operation == "write":
            self.files.write(args["path"], args["content"])
            return {"ok": True, "sha256": digest(self.files.path(args["path"]))}
        if operation == "delete":
            self.files.delete(args["path"])
            return {"ok": True}
        if operation in {"compile", "verify", "benchmark"}:
            return {**self.process(operation), "label": args.get("label")}
        if operation == "finish":
            save(self.run / "submitted-report.json", {"report": args.get("report", "")})
            checked_seconds = 0.0
            if self.checker:
                checked_at = time.monotonic()
                findings = source_checks(self.work)
                changed = self.integrity()
                artifact = self.artifact_hash() if (self.work / "model_new.py").is_file() else None
                expected = {"compile": self.build_hash(), "verify": artifact, "benchmark": artifact}
                stale = [name for name, value in expected.items() if value is None or self.validated.get(name) != value]
                checked_seconds = time.monotonic() - checked_at
                if findings or changed or stale:
                    return {"ok": False, "finish_rejected": True, "source_findings": findings, "changed_infrastructure": changed, "validation_missing_or_stale": stale, "checker_seconds": checked_seconds}
            self.submitted = True
            return {"ok": True, "submitted": True, "checker_seconds": checked_seconds}
        raise ValueError("unknown operation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output-budget", type=int, default=40000)
    parser.add_argument("--max-response-tokens", type=int, default=32768)
    parser.add_argument("--permissions", action="store_true")
    parser.add_argument("--checker", action="store_true")
    parser.add_argument("--task", default="axpby")
    parser.add_argument("--purpose", choices=["development", "formal"], default="development")
    args = parser.parse_args()
    task_spec = check_task_split(args.task, args.purpose)
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a single directory name")
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "runner-snapshot"
    snapshot.mkdir()
    runner_hashes = {}
    for filename in ["agent_trial.py", "runtime.py", "environment_probe.py", "event_timing_probe.py", "verify_candidate.py", "gpu_lock.py"]:
        shutil.copy2(ROOT / filename, snapshot / filename)
        runner_hashes[filename] = digest(snapshot / filename)
    executor = Executor(run, args.permissions, args.checker, task=args.task)
    auth = json.loads((Path.home() / ".local/share/opencode/auth.json").read_text(encoding="utf-8"))
    headers = {"Authorization": "Bearer " + auth["deepseek"]["key"]}
    prompt = BASE_PROMPT.format(steps=args.steps, output_budget=args.output_budget)
    report_contract = (ROOT / "materials/report_contract.md").read_text(encoding="utf-8")
    prompt += "\n\n" + report_contract
    save(run / "report_contract.json", {"sha256": digest(ROOT / "materials/report_contract.md"), "text": report_contract})
    if (executor.work / "contract.md").exists():
        prompt += "\n\nTask input contract:\n" + (executor.work / "contract.md").read_text(encoding="utf-8")
    if args.skill:
        prompt += "\n\nAdditional development guidance:\n" + args.skill.read_text(encoding="utf-8")
    messages = [{"role": "system", "content": "You are a CUDA kernel development agent using the supplied workspace tools."}, {"role": "user", "content": prompt}]
    save(run / "config.json", {"purpose": args.purpose, "pid": os.getpid(), "runner_hashes": runner_hashes, "model": "deepseek-flash", "thinking": "enabled", "effort": "high", "steps": args.steps, "output_budget": args.output_budget, "max_response_tokens": args.max_response_tokens, "permissions": args.permissions, "checker": args.checker, "skill": str(args.skill) if args.skill else None, "skill_sha256": digest(args.skill) if args.skill else None, "task": args.task, "task_spec": task_spec, "task_model_sha256": digest(executor.work / "model.py"), "started_utc": datetime.now(timezone.utc).isoformat(), "tools": [TOOL]})
    models = requests.get("https://api.deepseek.com/models", headers=headers, timeout=30)
    models.raise_for_status()
    save(run / "models.json", models.json())
    usage = []
    state = "step_budget"
    start = time.monotonic()
    with (run / "events.jsonl").open("w", encoding="utf-8") as events:
        def event(data):
            events.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **data}, ensure_ascii=False) + "\n")
            events.flush()
        for index in range(args.steps):
            remaining = args.output_budget - sum(u["completion_tokens"] for u in usage)
            if remaining < 256:
                state = "output_budget"
                break
            body = {"model": "deepseek-flash", "messages": messages, "tools": [TOOL], "max_tokens": min(args.max_response_tokens, remaining), "temperature": 1.0, "thinking": {"type": "enabled"}, "reasoning_effort": "high"}
            save(run / f"request-{index:03d}.json", body)
            print(f"request {index + 1}, remaining output budget {remaining}", flush=True)
            try:
                response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=body, timeout=(20, 180))
                save(run / f"response-{index:03d}.json", response.json())
                response.raise_for_status()
                data = response.json()
                usage.append(data["usage"])
                event({"type": "response", "index": index, "usage": data["usage"], "model": data.get("model"), "finish_reason": data["choices"][0]["finish_reason"]})
                choice = data["choices"][0]
                message = choice["message"]
                messages.append(message)
                if choice["finish_reason"] == "length":
                    state = "generation_truncated"
                    break
                calls = message.get("tool_calls") or []
                if not calls:
                    messages.append({"role": "user", "content": "Use the workspace tool to continue, or submit with operation finish."})
                for call in calls:
                    try:
                        if call["function"]["name"] != "workspace":
                            raise ValueError("unknown tool")
                        arguments = parse_tool_arguments(call["function"]["arguments"])
                        result = executor.execute(arguments)
                    except (ValueError, OSError, KeyError) as exc:
                        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    event({"type": "tool", "index": index, "call": call, "result": result})
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
                    print(f"tool {call['function']['name']}: {str(result)[:130]}", flush=True)
                if executor.submitted:
                    state = "submitted"
                    break
            except (requests.RequestException, ValueError, KeyError) as exc:
                state = "api_or_trace_error"
                event({"type": "error", "error": str(exc), "index": index})
                break
        # Same independent final evaluation after every trial, including failures.
        final = {}
        for operation in ["compile", "verify", "benchmark"]:
            executor.count += 1
            final[operation] = executor.process(operation)
            if not final[operation]["ok"]:
                break
        save(run / "final-evaluation.json", final)
        save(run / "messages.json", messages)
        usage_error = None
        try:
            totals = summarize_usage(usage) if usage else None
        except (KeyError, ValueError) as exc:
            totals, usage_error = None, str(exc)
        result = {"state": state, "seconds": time.monotonic() - start, "requests": len(usage), "requests_attempted": len(list(run.glob("request-*.json"))), "usage": totals, "usage_error": usage_error, "source_findings": source_checks(executor.work), "changed_infrastructure": executor.integrity(), "final_evaluation": final}
        save(run / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "final_evaluation"}, ensure_ascii=False), flush=True)
    return 0 if state == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
