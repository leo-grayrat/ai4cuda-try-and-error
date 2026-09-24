"""Summarize measured leave-one-action-out effects for one prepared case."""

import argparse
import json
from pathlib import Path

from kernel_lab.credit_evidence import summarize_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.records]
    summary = summarize_case(trace, records)
    summary["record_paths"] = [str(path) for path in args.records]
    payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
