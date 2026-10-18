"""Run the pinned CUDA-Agent example with minimal Windows build adaptation.

This is an environment probe, never an effectiveness experiment. Originals in
sources/ remain untouched. All modifications and subprocess output are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def msvc_environment() -> dict[str, str]:
    env = os.environ.copy()
    if os.name != "nt":
        return env
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    install = subprocess.check_output(
        [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
        text=True,
    ).strip()
    vcvars = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
    result = subprocess.check_output(
        f'"{vcvars}" >nul && set', shell=True, text=True, errors="replace"
    )
    for line in result.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            env[key] = value
    env["DISTUTILS_USE_SDK"] = "1"
    env["TORCH_CUDA_ARCH_LIST"] = "12.0"
    env["MAX_JOBS"] = "2"
    return env


def run_step(command: list[str], work: Path, env: dict[str, str], log, timeout: int = 600):
    process = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        return process.wait(timeout=timeout), "finished"
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=log, stderr=subprocess.STDOUT, check=False)
        else:
            process.kill()
        process.wait()
        return None, "timeout"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="environment-01")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a single directory name")
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=False)
    work = run / "work"
    shutil.copytree(ROOT / "sources/CUDA-Agent", work)
    original = (work / "utils/compile.py").read_text(encoding="utf-8")
    adapted = original
    if os.name == "nt":
        adapted = adapted.replace("cuda_extension.so", "cuda_extension.pyd")
        adapted = adapted.replace("extra_cflags=['-O3', '-std=c++17']", "extra_cflags=['/O2', '/std:c++17']")
        # This API-hosted Windows session reports OEM code page 1, which is
        # not a usable decoder for compiler output. Change decoding only.
        adapted = adapted.replace("import torch.utils.cpp_extension as cpp_ext", "import torch.utils.cpp_extension as cpp_ext\ncpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace')")
        (work / "utils/compile.py").write_text(adapted, encoding="utf-8")
    (run / "adaptation.json").write_text(json.dumps({
        "platform": sys.platform,
        "changes": [".so -> .pyd", "C++ -O3/-std=c++17 -> /O2 /std:c++17", "compiler output decoding OEM -> UTF-8 with replacement"] if os.name == "nt" else [],
        "original_compile_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "adapted_compile_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
        "python": sys.executable,
        "arch": "12.0",
    }, indent=2), encoding="utf-8")
    env = msvc_environment()
    for key, folder in [("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"), ("TEMP", "temp"), ("TMP", "temp")]:
        destination = run / folder
        destination.mkdir(exist_ok=True)
        env[key] = str(destination)
    steps = [
        ("environment", [sys.executable, "-c", "import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.cuda.is_available(), CUDA_HOME); print(torch.cuda.get_device_name(0))"]),
        ("compile", [sys.executable, "-m", "utils.compile"]),
        ("verification", [sys.executable, "-m", "utils.verification"]),
        ("profiling", [sys.executable, "-m", "utils.profiling", "--iters", "30"]),
        ("event_timing", [sys.executable, str(ROOT / "event_timing_probe.py")]),
    ]
    records = []
    for name, command in steps:
        start = time.monotonic()
        print(f"Starting {name}", flush=True)
        with (run / f"{name}.log").open("w", encoding="utf-8") as log:
            code, state = run_step(command, work, env, log)
        record = {"step": name, "command": command, "returncode": code, "state": state, "seconds": time.monotonic() - start}
        records.append(record)
        (run / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        if code != 0:
            print((run / f"{name}.log").read_text(encoding="utf-8")[-5000:], flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
