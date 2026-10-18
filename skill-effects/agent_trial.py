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
import shutil
import subprocess
import sys
import time

import requests

from environment_probe import ROOT, msvc_environment
from runtime import FileTools, compiler_feedback, digest, source_checks, summarize_usage

TOOL = {"type": "function", "function": {
    "name": "workspace",
    "description": "Operate on the CUDA extension workspace. read/write/delete use relative paths; write replaces a whole file. compile builds all C++/CUDA sources. verify runs numerical comparison. benchmark compares candidate with eager and torch.compile. finish submits the current files and report. No arbitrary shell is available.",
    "parameters": {"type": "object", "properties": {
        "operation": {"type": "string", "enum": ["list", "read", "write", "delete", "compile", "verify", "benchmark", "finish"]},
        "path": {"type": "string"}, "content": {"type": "string"},
        "report": {"type": "string", "description": "Final explanation for finish."},
    }, "required": ["operation"]},
}}

BASE_PROMPT = """Accelerate the PyTorch Model in model.py by implementing a CUDA C++ extension and a compatible ModelNew in model_new.py. The CUDA extension is named cuda_extension; binding.cpp and binding_registry.h provide automatic registration for kernels/*_binding.cpp. The original Model constructor and state_dict must work with ModelNew. The objective is correct output and candidate latency <= 0.95 times the torch.compile baseline, with the best performance attainable within the available budget.

Platform: Windows, RTX 5060 Laptop (sm_120), PyTorch 2.11.0+cu128, CUDA Toolkit 12.9, MSVC. Use the provided compile/verify/benchmark tools as the equivalents of the Linux shell commands in any supplied skill; the tools set architecture and compiler flags. This platform note is the same for all conditions. Benchmark uses CUDA events because the upstream CUPTI profiler is unavailable here.

The workspace begins with model.py, model_new.py, binding.cpp, binding_registry.h and utils/. Use workspace tools to inspect and implement the solution. Finish by calling finish with your final report. No text response alone submits the task. You have {steps} model requests and at most {output_budget} generated tokens in total, including reasoning. No new libraries can be installed through these tools.
"""


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def stage(run: Path) -> tuple[Path, Path]:
    work, control = run / "work", run / "control"
    shutil.copytree(ROOT / "sources/CUDA-Agent", work)
    shutil.copytree(ROOT / "sources/CUDA-Agent", control)
    (control / "model_new.py").unlink()
    shutil.copy2(ROOT / "event_timing_probe.py", control / "event_timing_probe.py")
    # Start without an already-solved candidate; keep the unchanged upstream task.
    for path in (work / "kernels").iterdir():
        path.unlink()
    (work / "model_new.py").write_text("import torch.nn as nn\n\nclass ModelNew(nn.Module):\n    def __init__(self, alpha):\n        super().__init__()\n        self.alpha = alpha\n    def forward(self, a, b):\n        raise NotImplementedError('Implement CUDA extension')\n", encoding="utf-8")
    compile_path = control / "utils/compile.py"
    text = compile_path.read_text(encoding="utf-8")
    if os.name == "nt":
        text = text.replace("cuda_extension.so", "cuda_extension.pyd").replace("extra_cflags=['-O3', '-std=c++17']", "extra_cflags=['/O2', '/std:c++17']")
        text = text.replace("import torch.utils.cpp_extension as cpp_ext", "import torch.utils.cpp_extension as cpp_ext\ncpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace')")
    compile_path.write_text(text, encoding="utf-8")
    shutil.copy2(compile_path, work / "utils/compile.py")
    return work, control


class Executor:
    def __init__(self, run: Path, permissions: bool, checker: bool):
        self.run = run
        self.work, self.control = stage(run)
        self.files = FileTools(self.work, permissions)
        self.checker = checker
        self.env = msvc_environment()
        for key, name in [("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"), ("TEMP", "temp"), ("TMP", "temp")]:
            folder = run / name
            folder.mkdir(exist_ok=True)
            self.env[key] = str(folder)
        self.frozen = {str(p.relative_to(self.work)): digest(p) for p in self.work.rglob("*") if p.is_file() and p.name != "model_new.py"}
        self.count = 0
        self.submitted = False
        self.compiled_hash = None

    def integrity(self):
        return [name for name, expected in self.frozen.items() if not (self.work / name).is_file() or digest(self.work / name) != expected]

    def build_hash(self):
        import hashlib
        rows = [(str(p.relative_to(self.work)), digest(p)) for p in self.work.rglob("*") if p.is_file() and p.suffix in {".cu", ".cpp", ".h", ".cuh"} and "build" not in p.relative_to(self.work).parts]
        return hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()

    def process(self, operation: str):
        if operation != "compile" and self.compiled_hash != self.build_hash():
            return {"ok": False, "error": "Compile the current candidate before verification or benchmarking."}
        if operation == "compile":
            command = [sys.executable, str(self.control / "utils/compile.py")]
        else:
            setup = f"import sys,runpy; sys.path.insert(0,{str(self.work)!r}); sys.path.insert(0,{str(self.control)!r}); "
            if operation == "verify":
                command = [sys.executable, "-c", setup + "runpy.run_module('utils.verification',run_name='__main__')"]
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
        return {"ok": ok, "returncode": code, "seconds": time.monotonic() - started, "log": log.name, "output": compiler_feedback(output), "changed_infrastructure": changed}

    def execute(self, args: dict):
        self.count += 1
        operation = args["operation"]
        if operation == "list":
            return {"files": self.files.list()}
        if operation == "read":
            return {"content": self.files.read(args["path"])[:18000]}
        if operation == "write":
            self.files.write(args["path"], args["content"])
            return {"ok": True, "sha256": digest(self.files.path(args["path"]))}
        if operation == "delete":
            self.files.delete(args["path"])
            return {"ok": True}
        if operation in {"compile", "verify", "benchmark"}:
            return self.process(operation)
        if operation == "finish":
            save(self.run / "submitted-report.json", {"report": args.get("report", "")})
            findings = source_checks(self.work)
            changed = self.integrity()
            if self.checker and (findings or changed):
                return {"ok": False, "finish_rejected": True, "source_findings": findings, "changed_infrastructure": changed}
            self.submitted = True
            return {"ok": True, "submitted": True}
        raise ValueError("unknown operation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output-budget", type=int, default=40000)
    parser.add_argument("--permissions", action="store_true")
    parser.add_argument("--checker", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a single directory name")
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "runner-snapshot"
    snapshot.mkdir()
    runner_hashes = {}
    for filename in ["agent_trial.py", "runtime.py", "environment_probe.py", "event_timing_probe.py"]:
        shutil.copy2(ROOT / filename, snapshot / filename)
        runner_hashes[filename] = digest(snapshot / filename)
    executor = Executor(run, args.permissions, args.checker)
    auth = json.loads((Path.home() / ".local/share/opencode/auth.json").read_text(encoding="utf-8"))
    headers = {"Authorization": "Bearer " + auth["deepseek"]["key"]}
    prompt = BASE_PROMPT.format(steps=args.steps, output_budget=args.output_budget)
    if args.skill:
        prompt += "\n\nAdditional development guidance:\n" + args.skill.read_text(encoding="utf-8")
    messages = [{"role": "system", "content": "You are a CUDA kernel development agent using the supplied workspace tools."}, {"role": "user", "content": prompt}]
    save(run / "config.json", {"purpose": "development", "pid": os.getpid(), "runner_hashes": runner_hashes, "model": "deepseek-flash", "thinking": "enabled", "effort": "high", "steps": args.steps, "output_budget": args.output_budget, "permissions": args.permissions, "checker": args.checker, "skill": str(args.skill) if args.skill else None, "skill_sha256": digest(args.skill) if args.skill else None, "task": "pinned upstream AXPBY example, unchanged model.py", "started_utc": datetime.now(timezone.utc).isoformat(), "tools": [TOOL]})
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
            body = {"model": "deepseek-flash", "messages": messages, "tools": [TOOL], "max_tokens": min(8192, remaining), "temperature": 1.0, "thinking": {"type": "enabled"}, "reasoning_effort": "high"}
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
                        arguments = json.loads(call["function"]["arguments"])
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
        result = {"state": state, "seconds": time.monotonic() - start, "requests": len(usage), "usage": summarize_usage(usage) if usage else None, "source_findings": source_checks(executor.work), "changed_infrastructure": executor.integrity(), "final_evaluation": final}
        save(run / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "final_evaluation"}, ensure_ascii=False), flush=True)
    return 0 if state == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
