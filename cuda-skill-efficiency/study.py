"""Thin recorder for paired CUDA-agent skill trials; it does not implement an agent loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_task(source: Path, work: Path) -> dict[str, str]:
    """Copy only agent-visible task files; the independent evaluator stays outside."""
    work.mkdir(parents=True, exist_ok=False)
    for name in ("task.md", "reference.py", "candidate.py"):
        shutil.copy2(source / name, work / name)
    return {
        "task_sha256": sha256(source / "task.md"),
        "reference_sha256": sha256(source / "reference.py"),
        "candidate_sha256": sha256(source / "candidate.py"),
        "evaluator_sha256": sha256(source / "evaluate.py"),
    }


def collect_usage(events: list[dict]) -> dict[str, int] | None:
    turns = [e for e in events if e.get("type") == "turn.completed"]
    if not turns or any(not isinstance(e.get("usage"), dict) for e in turns):
        return None
    keys = ("input_tokens", "cached_input_tokens", "output_tokens")
    if any(any(not isinstance(e["usage"].get(k), int) for k in keys) for e in turns):
        return None
    return {"turns": len(turns), **{k: sum(e["usage"][k] for e in turns) for k in keys}}


def summarize_events(events: list[dict]) -> dict:
    usage = collect_usage(events)
    failed = any(e.get("type") == "turn.failed" for e in events)
    return {
        "status": "agent_failed" if failed else "complete" if usage else "incomplete_trace",
        "usage": usage,
        "tool_events": sum(
            e.get("type") == "item.completed"
            and e.get("item", {}).get("type") in {"command_execution", "file_change", "mcp_tool_call"}
            for e in events
        ),
    }


def collect_opencode_usage(events: list[dict]) -> dict[str, int | float] | None:
    steps = [event for event in events if event.get("type") == "step_finish"]
    if not steps:
        return None
    fields = ("total", "input", "output", "reasoning")
    totals = {key: 0 for key in fields}
    cache_read = cache_write = 0
    cost = 0.0
    for event in steps:
        part = event.get("part", {})
        tokens = part.get("tokens", {})
        cache = tokens.get("cache", {})
        if any(not isinstance(tokens.get(key), int) for key in fields):
            return None
        if any(not isinstance(cache.get(key), int) for key in ("read", "write")):
            return None
        if not isinstance(part.get("cost"), (int, float)):
            return None
        for key in fields:
            totals[key] += tokens[key]
        cache_read += cache["read"]
        cache_write += cache["write"]
        cost += part["cost"]
    input_tokens = totals["input"] + cache_read + cache_write
    output_tokens = totals["output"] + totals["reasoning"]
    if totals["total"] != input_tokens + output_tokens:
        return None
    return {
        "steps": len(steps),
        "input_tokens": input_tokens,
        "uncached_input_tokens": totals["input"],
        "cached_input_tokens": cache_read,
        "cache_write_input_tokens": cache_write,
        "output_tokens": output_tokens,
        "visible_output_tokens": totals["output"],
        "reasoning_output_tokens": totals["reasoning"],
        "total_tokens": totals["total"],
        "cost_usd": round(cost, 10),
    }


def summarize_opencode_events(events: list[dict]) -> dict:
    usage = collect_opencode_usage(events)
    steps = [event for event in events if event.get("type") == "step_finish"]
    tools = [event for event in events if event.get("type") == "tool_use"]
    failed = any(event.get("type") == "error" for event in events)
    stopped = bool(steps) and steps[-1].get("part", {}).get("reason") == "stop"
    return {
        "status": "agent_failed" if failed else "complete" if usage and stopped else "incomplete_trace",
        "usage": usage,
        "tool_events": len(tools),
        "tool_feedback_events": sum(
            tool.get("part", {}).get("state", {}).get("status") in {"completed", "error"}
            for tool in tools
        ),
    }


def run_exit_code(summary: dict) -> int:
    return 0 if summary["status"] == "complete" else 1


def estimated_skill_tokens(skill_text: str) -> tuple[int | None, str | None]:
    if not skill_text:
        return 0, "none"
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(skill_text)), "o200k_base estimate; not model-exact"
    except ImportError:
        return None, None


def child_environment(
    parent: dict[str, str], windows: bool | None = None,
    python_executable: Path | None = None, cache_dir: Path | None = None,
) -> dict[str, str]:
    env = parent.copy()
    on_windows = windows if windows is not None else os.name == "nt"
    if on_windows:
        profile = env.get("USERPROFILE")
        if profile:
            env.setdefault("HOME", profile)
            env.setdefault("CODEX_HOME", str(Path(profile) / ".codex"))
    if python_executable:
        env["PATH"] = str(python_executable.parent) + (";" if on_windows else os.pathsep) + env.get("PATH", "")
    if cache_dir:
        env["TRITON_CACHE_DIR"] = str(cache_dir)
        env["CUPY_CACHE_DIR"] = str(cache_dir)
    return env


def stop_process_tree(proc: subprocess.Popen, windows: bool | None = None) -> None:
    """Stop the launched CLI and its children when the trial deadline expires."""
    if proc.poll() is not None:
        return
    on_windows = windows if windows is not None else os.name == "nt"
    if on_windows:
        try:
            result = subprocess.run(
                ["taskkill.exe", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
    proc.kill()


def run_trial(
    task: Path, skill: Path | None, model: str, output: Path, timeout_s: int,
    backend: str = "opencode",
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    work = output / "work"
    task_hashes = stage_task(task, work)
    skill_text = skill.read_text(encoding="utf-8") if skill else ""
    prompt = (task / "task.md").read_text(encoding="utf-8")
    prompt += "\n\nEdit only candidate.py. Check the result against reference.py on a small and a large CUDA input. Report what you checked."
    if skill_text:
        prompt += "\n\n---\n\nAdditional CUDA development guidance:\n\n" + skill_text
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    if backend == "opencode":
        command = [
            "opencode.cmd" if os.name == "nt" else "opencode", "run", "--pure", "--auto",
            "--model", model, "--format", "json", "--title",
            f"AI4CUDA {task.name} {skill.name if skill else 'none'}", prompt,
        ]
    elif backend == "codex":
        command = [
            "codex.cmd" if os.name == "nt" else "codex", "exec", "--json", "--ephemeral",
            "--ignore-user-config", "--disable", "plugins", "--disable", "skill_search",
            "--skip-git-repo-check", "--approve-for-me",
            "-m", model, "-C", str(work), "-",
        ]
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    started = time.time()
    events: list[dict] = []
    seen_hashes: set[str] = set()
    snapshots: list[dict] = []
    versions = output / "versions"
    versions.mkdir()
    cache = output / "triton-cache"
    cache.mkdir()
    env = child_environment(os.environ, python_executable=Path(sys.executable), cache_dir=cache)
    env["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
    version = subprocess.run(
        [command[0], "--version"], cwd=work, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
    )
    with (output / "events.jsonl").open("w", encoding="utf-8") as raw, (output / "stderr.txt").open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            command, stdin=subprocess.PIPE if backend == "codex" else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=err,
            text=True, encoding="utf-8", errors="replace", cwd=work,
            env=env,
        )
        expired = threading.Event()
        def on_timeout() -> None:
            if proc.poll() is None:
                expired.set()
                stop_process_tree(proc)
        timer = threading.Timer(timeout_s, on_timeout)
        timer.start()
        assert proc.stdout is not None
        if backend == "codex":
            assert proc.stdin is not None
            proc.stdin.write(prompt)
            proc.stdin.close()
        for line in proc.stdout:
            raw.write(line)
            raw.flush()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            candidate = work / "candidate.py"
            if candidate.exists():
                digest = sha256(candidate)
                if digest not in seen_hashes:
                    seen_hashes.add(digest)
                    shutil.copy2(candidate, versions / f"{len(snapshots):03d}-{digest[:12]}.py")
                    snapshots.append({"event_index": len(events) - 1, "sha256": digest})
        return_code = proc.wait()
        timed_out = expired.is_set()
        timer.cancel()
    summary = summarize_opencode_events(events) if backend == "opencode" else summarize_events(events)
    if timed_out:
        summary["status"] = "timeout"
    elif return_code != 0 and summary["status"] != "agent_failed":
        summary["status"] = "runner_failed"
    count, method = estimated_skill_tokens(skill_text)
    summary.update({
        "task": task.name,
        "task_hashes": task_hashes,
        "skill": skill.name if skill else "none",
        "skill_sha256": sha256(skill) if skill else None,
        "skill_tokens_estimate": count,
        "skill_token_method": method,
        "model_requested": model,
        "model_revision": None,
        "runner": backend,
        "runner_version": version.stdout.strip() if version.returncode == 0 else None,
        "usage_granularity": "OpenCode model step" if backend == "opencode" else "Codex turn total",
        "command": command[:-1] + ["<prompt>"],
        "return_code": return_code,
        "elapsed_s": round(time.time() - started, 3),
        "source_versions": snapshots,
        "final_candidate_sha256": sha256(work / "candidate.py"),
    })
    (output / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def evaluate_trial(task: Path, trial: Path, python: str) -> int:
    if (trial / "evaluation.json").exists():
        raise FileExistsError(trial / "evaluation.json")
    cache = trial / "triton-cache"
    cache.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = str(cache)
    result = subprocess.run(
        [python, str(task / "evaluate.py"), str(trial / "work" / "candidate.py")],
        cwd=task, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        env=env,
    )
    (trial / "evaluation.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (trial / "evaluation.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode == 0:
        parsed = json.loads(result.stdout)
        (trial / "evaluation.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        return 0 if parsed.get("correct") is True else 1
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--skill", type=Path)
    run.add_argument("--backend", choices=("opencode", "codex"), default="opencode")
    run.add_argument("--model")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--timeout-s", type=int, default=300)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--task", type=Path, required=True)
    evaluate.add_argument("--trial", type=Path, required=True)
    evaluate.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.command == "run":
        model = args.model or ("deepseek/deepseek-flash" if args.backend == "opencode" else "gpt-5.5")
        summary = run_trial(
            args.task.resolve(), args.skill.resolve() if args.skill else None,
            model, args.output.resolve(), args.timeout_s, args.backend,
        )
        print(json.dumps(summary, ensure_ascii=False))
        raise SystemExit(run_exit_code(summary))
    else:
        raise SystemExit(evaluate_trial(args.task.resolve(), args.trial.resolve(), args.python))


if __name__ == "__main__":
    main()
