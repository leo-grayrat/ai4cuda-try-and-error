"""CLI for reproducible kernel candidate measurements and two analyses."""

import argparse
import copy
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from kernel_lab.core import (
    compare_hardware,
    credit_analysis,
    hardware_info,
    make_record,
    read_json,
    validate_manifest,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="measure every candidate on this GPU")
    run.add_argument("manifest")
    run.add_argument("output")
    run.add_argument("--kernelbench-root", help="path to an upstream KernelBench checkout")
    run.add_argument("--evaluator-python", help="Python executable with KernelBench and CUDA dependencies")
    run.add_argument("--evaluator-source", help="replay a saved evaluator source file")
    run.add_argument("--order-seed", type=int, help="shuffle candidate evaluation order")
    credit = commands.add_parser("credit", help="analyze a measured 2x2 intervention")
    credit.add_argument("record")
    series = commands.add_parser("credit-series", help="summarize repeated 2x2 runs on one GPU")
    series.add_argument("records", nargs="+")
    compare = commands.add_parser("compare", help="compare identical candidates on two GPUs")
    compare.add_argument("gpu_a_record")
    compare.add_argument("gpu_b_record")
    args = parser.parse_args()

    if args.command == "run":
        output_path = Path(args.output).resolve()
        if output_path.exists():
            raise FileExistsError(f"Run record already exists: {output_path}")
        assets = output_path.with_suffix(".assets")
        if assets.exists():
            raise FileExistsError(f"Run asset folder already exists: {assets}")
        manifest = read_json(args.manifest)
        validate_manifest(manifest)
        snapshot_manifest = copy.deepcopy(manifest)
        reference_hash = None
        upstream_revision = None
        runtime = {"python": sys.version.split()[0]}
        source_paths = {}
        if manifest["evaluator"] == "numba_axpy":
            if args.evaluator_source:
                evaluator_path = Path(args.evaluator_source).resolve()
                specification = importlib.util.spec_from_file_location("saved_numba_evaluator", evaluator_path)
                if specification is None or specification.loader is None:
                    raise ValueError(f"Cannot load saved evaluator: {evaluator_path}")
                evaluator = importlib.util.module_from_spec(specification)
                specification.loader.exec_module(evaluator)
            else:
                from kernel_lab import numba_axpy as evaluator

                evaluator_path = Path(evaluator.__file__)
            measurements = evaluator.evaluate(manifest)
            source = evaluator_path.read_bytes()
            source_paths["evaluator"] = evaluator_path
            source_paths["requirements"] = Path(__file__).resolve().parent / "requirements-smoke.txt"
            runtime.update({
                "numba": importlib.metadata.version("numba"),
                "numba_cuda": importlib.metadata.version("numba-cuda"),
                "numpy": importlib.metadata.version("numpy"),
            })
        elif manifest["evaluator"] == "kernelbench":
            if not args.kernelbench_root:
                raise ValueError("--kernelbench-root is required for KernelBench experiments")
            from kernel_lab import kernelbench_adapter

            evaluator_path = Path(args.evaluator_source).resolve() if args.evaluator_source else Path(kernelbench_adapter.__file__).resolve()
            source = evaluator_path.read_bytes()
            source_paths["evaluator"] = evaluator_path

            root = Path(args.kernelbench_root).resolve()
            manifest_dir = Path(args.manifest).resolve().parent
            reference = (manifest_dir / manifest["reference_path"]).resolve()
            reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
            source_paths["reference"] = reference
            snapshot_manifest["reference_path"] = "reference.py"
            revision_result = subprocess.run(
                ["git", "-c", "http.sslBackend=openssl", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            upstream_revision = revision_result.stdout.strip()
            evaluator_python = args.evaluator_python or sys.executable
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(Path(__file__).resolve().parent), environment.get("PYTHONPATH", "")])
            measurements = []
            evaluation_order = list(manifest["candidates"])
            if args.order_seed is not None:
                random.Random(args.order_seed).shuffle(evaluation_order)
            for candidate in evaluation_order:
                candidate_path = (manifest_dir / candidate["source_path"]).resolve()
                source_paths[candidate["id"]] = candidate_path
                next(item for item in snapshot_manifest["candidates"] if item["id"] == candidate["id"])["source_path"] = f"{candidate['id']}{candidate_path.suffix}"
                command = [
                    evaluator_python, str(evaluator_path),
                    "--reference", str(reference), "--candidate", str(candidate_path),
                    "--seed", str(manifest["workload"].get("seed", 42)),
                    "--correct-trials", str(manifest["workload"].get("correct_trials", 5)),
                    "--perf-trials", str(manifest["workload"].get("perf_trials", 100)),
                ]
                completed = subprocess.run(command, env=environment, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(f"KernelBench candidate {candidate['id']} failed:\n{completed.stderr[-4000:]}")
                result = json.loads(completed.stdout)
                result["id"] = candidate["id"]
                result["code_hash"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
                measurements.append(result)
            runtime = measurements[0].get("runtime_versions", runtime)
        else:
            raise ValueError(f"Unknown evaluator: {manifest['evaluator']}")
        result = make_record(
            manifest, measurements, hardware_info(), source,
            reference_sha256=reference_hash, upstream_revision=upstream_revision,
            runtime=runtime,
        )
        result["evaluation_order"] = [item["id"] for item in evaluation_order] if manifest["evaluator"] == "kernelbench" else [item["id"] for item in manifest["candidates"]]
        assets.mkdir(parents=True, exist_ok=False)
        source_paths["orchestrator"] = Path(__file__).resolve()
        source_paths["analysis"] = Path(__file__).resolve().parent / "kernel_lab" / "core.py"
        result["artifacts"] = {}
        for label, source_path in source_paths.items():
            copied = assets / f"{label}{source_path.suffix}"
            shutil.copyfile(source_path, copied)
            result["artifacts"][label] = copied.relative_to(output_path.parent).as_posix()
        write_json(assets / "manifest.json", snapshot_manifest)
        result["artifacts"]["manifest"] = (assets / "manifest.json").relative_to(output_path.parent).as_posix()
        write_json(output_path, result)
        print(f"Saved {len(measurements)} candidates to {args.output}")
    elif args.command == "credit":
        print(json.dumps(credit_analysis(read_json(args.record)), ensure_ascii=False, indent=2))
    elif args.command == "credit-series":
        from kernel_lab.core import credit_series

        print(json.dumps(credit_series([read_json(path) for path in args.records]), ensure_ascii=False, indent=2))
    else:
        result = compare_hardware(read_json(args.gpu_a_record), read_json(args.gpu_b_record))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
