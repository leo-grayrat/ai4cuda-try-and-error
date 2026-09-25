"""Export paired dataset formatted for token-weighted offline policy fine-tuning.

For each OracleCase, this script exports training instances where each token has:
- text token / token id
- Baseline credit weight A_t (from GAE diffusion of terminal reward)
- Oracle credit weight B_t (from strict semantic decision concentration)

This format can be directly ingested by policy gradient loss:
  Loss_A = - sum_t ( A_t * log P(y_t | y_<t, x) )
  Loss_B = - sum_t ( B_t * log P(y_t | y_<t, x) )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kernel_lab.oracle_credit import (
    OracleCase,
    compute_baseline_credit,
    compute_oracle_credit,
)


def export_paired_dataset(
    cases_dir: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    records = []
    for json_file in sorted(cases_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        case = OracleCase.from_dict(data)

        num_tokens = len(case.tokens)
        baseline_credits = compute_baseline_credit(num_tokens, case.terminal_reward)
        oracle_credits = compute_oracle_credit(num_tokens, case.decisions, case.terminal_reward)

        opt_indices = set()
        for d in case.decisions:
            if d.is_optimization:
                for s in d.spans:
                    if s.is_optimization:
                        opt_indices.update(s.token_indices)

        token_records = []
        for i, (tok, (start, end)) in enumerate(zip(case.tokens, case.token_offsets)):
            token_records.append({
                "index": i,
                "token": tok,
                "char_span": [start, end],
                "is_optimization": i in opt_indices,
                "baseline_credit": baseline_credits[i],
                "oracle_credit": oracle_credits[i],
            })

        prompt = f"Optimize the following PyTorch/CUDA kernel for task '{case.benchmark_task}':\n```python\n{case.parent_source}\n```"

        records.append({
            "case_id": case.case_id,
            "title": case.title,
            "benchmark_task": case.benchmark_task,
            "prompt": prompt,
            "completion": case.child_source,
            "terminal_reward": case.terminal_reward,
            "tokens": token_records,
            "summary": {
                "total_tokens": num_tokens,
                "optimization_tokens": len(opt_indices),
                "total_reward": case.terminal_reward,
            },
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    cases_dir = Path("experiments/oracle-cases")
    out_file = Path("experiments/oracle-cases/paired_credit_dataset.json")
    records = export_paired_dataset(cases_dir, out_file)
    print(f"Exported {len(records)} paired credit training records to {out_file}")
