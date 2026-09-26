"""Join a model generation with a prepared code-credit trace.

The generation JSON must contain the exact raw_response, sampled token_ids,
and tokenizer-produced token_offsets as [start, end] character positions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_lab.response_mapping import record_generation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("child", type=Path)
    parser.add_argument("generation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    generation = json.loads(args.generation.read_text(encoding="utf-8"))
    recorded = record_generation(
        trace,
        args.child.read_text(encoding="utf-8"),
        generation["raw_response"],
        generation["token_ids"],
        [tuple(offset) for offset in generation["token_offsets"]],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(recorded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment_id": trace["experiment_id"],
                      "mapping_status": recorded["source_to_response"]["status"]}))


if __name__ == "__main__":
    main()
