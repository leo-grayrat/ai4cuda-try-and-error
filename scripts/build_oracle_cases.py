"""Generate canonical Oracle cases for semantic credit assignment."""

import json
from pathlib import Path

from kernel_lab.oracle_credit import (
    CodeSpan,
    OracleCase,
    OracleDecision,
    align_spans_to_tokens,
    simple_tokenize_with_offsets,
)

CASES_DIR = Path("experiments/oracle-cases")
CASES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Case 1: 2_20 Device Memory Copy Elimination (clone().detach() -> detach())
# -------------------------------------------------------------
parent_2_20 = '''import torch
from torch import nn

class ModelNew(nn.Module):
    """Parent model performing redundant tensor copy before kernel invocation."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Redundant intermediate memory allocation and copy
        temp = x.clone().detach()
        out = torch.relu(temp)
        return out
'''

child_2_20 = '''import torch
from torch import nn

class ModelNew(nn.Module):
    """Optimized model: eliminated clone().detach() to avoid device memory allocation."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optimized: in-place view avoiding clone overhead
        temp = x.detach()
        out = torch.relu(temp)
        return out
'''

target_str = "temp = x.detach()"
start_char = child_2_20.find(target_str)
end_char = start_char + len(target_str)

tokens_1, offsets_1 = simple_tokenize_with_offsets(child_2_20)
span_1 = CodeSpan(
    start_char=start_char,
    end_char=end_char,
    description="Eliminated clone() to prevent redundant device memory allocation and copying",
    is_optimization=True,
)
align_spans_to_tokens(offsets_1, [span_1])

case_1 = OracleCase(
    case_id="case_2_20_copy_elimination",
    title="Device Memory Copy Elimination (2_20)",
    benchmark_task="KernelBench Level 2 Task 20",
    parent_source=parent_2_20,
    child_source=child_2_20,
    raw_response=f"Here is the optimized kernel implementation:\n```python\n{child_2_20}\n```",
    tokens=tokens_1,
    token_offsets=offsets_1,
    decisions=[
        OracleDecision(
            decision_id="eliminate_intermediate_copy",
            category="memory_elimination",
            description="Replaced x.clone().detach() with x.detach() to save allocation overhead",
            spans=[span_1],
            is_optimization=True,
        )
    ],
    terminal_reward=2.0,
    metadata={
        "measured_speedup": 1.301,
        "mechanism": "Avoids cudaMalloc & device-to-device copy",
    },
)

# -------------------------------------------------------------
# Case 2: Vector Add Loop Unrolling (1 element -> 2 elements per thread)
# -------------------------------------------------------------
base_add = Path("experiments/kernelbench-add/base.py").read_text(encoding="utf-8")
child_add = Path("experiments/kernelbench-add/A.py").read_text(encoding="utf-8")

# Find the unrolled computation lines in child
unroll_cuda_str = """    int stride = gridDim.x * blockDim.x;
    if (i < n) out[i] = a[i] + b[i];
    int j = i + stride;
    if (j < n) out[j] = a[j] + b[j];"""
unroll_blocks_str = "blocks = (a.numel() + 2 * threads - 1) // (2 * threads)"

tokens_2, offsets_2 = simple_tokenize_with_offsets(child_add)
span_2a = CodeSpan(
    start_char=child_add.find(unroll_cuda_str),
    end_char=child_add.find(unroll_cuda_str) + len(unroll_cuda_str),
    description="CUDA kernel instruction: process 2 elements per thread via grid stride",
    is_optimization=True,
)
span_2b = CodeSpan(
    start_char=child_add.find(unroll_blocks_str),
    end_char=child_add.find(unroll_blocks_str) + len(unroll_blocks_str),
    description="Host launch grid computation: halving number of blocks for 2x unrolling",
    is_optimization=True,
)
align_spans_to_tokens(offsets_2, [span_2a, span_2b])

case_2 = OracleCase(
    case_id="case_add_unroll_two",
    title="Vector Add 2x Grid-Stride Loop Unrolling",
    benchmark_task="KernelBench Vector Add",
    parent_source=base_add,
    child_source=child_add,
    raw_response=f"Optimized implementation unrolling 2 elements per thread:\n```python\n{child_add}\n```",
    tokens=tokens_2,
    token_offsets=offsets_2,
    decisions=[
        OracleDecision(
            decision_id="unroll_two_elements",
            category="instruction_parallelism",
            description="Process 2 elements per thread to increase instruction level parallelism and memory coalescing",
            spans=[span_2a, span_2b],
            is_optimization=True,
        )
    ],
    terminal_reward=1.5,
    metadata={
        "mechanism": "Increases instruction-level parallelism (ILP) and hides memory latency",
    },
)

# -------------------------------------------------------------
# Case 3: Row Sum Vectorized Stride Reduction
# -------------------------------------------------------------
base_row_sum = Path("experiments/kernelbench-row-sum/base.py").read_text(encoding="utf-8")
child_row_sum = Path("experiments/kernelbench-row-sum/A.py").read_text(encoding="utf-8")

unroll_reduction_str = """    for (int col = 2 * lane; col < columns; col += 256) {
        sum += x[row * columns + col];
        if (col + 1 < columns) sum += x[row * columns + col + 1];
    }"""

tokens_3, offsets_3 = simple_tokenize_with_offsets(child_row_sum)
span_3 = CodeSpan(
    start_char=child_row_sum.find(unroll_reduction_str),
    end_char=child_row_sum.find(unroll_reduction_str) + len(unroll_reduction_str),
    description="Dual-element accumulation loop per thread in shared-memory row reduction",
    is_optimization=True,
)
align_spans_to_tokens(offsets_3, [span_3])

case_3 = OracleCase(
    case_id="case_row_sum_stride256_unroll",
    title="Row Sum 2-Way Stride Reduction",
    benchmark_task="KernelBench Row Sum",
    parent_source=base_row_sum,
    child_source=child_row_sum,
    raw_response=f"Row sum with dual-element stride reduction:\n```python\n{child_row_sum}\n```",
    tokens=tokens_3,
    token_offsets=offsets_3,
    decisions=[
        OracleDecision(
            decision_id="dual_stride_reduction",
            category="reduction_unrolling",
            description="Accumulate pairs of column elements per lane before shared-memory tree reduction",
            spans=[span_3],
            is_optimization=True,
        )
    ],
    terminal_reward=2.0,
    metadata={
        "mechanism": "Halves loop iterations and balances shared memory tree reduction stages",
    },
)

# Write all 3 cases
for c in [case_1, case_2, case_3]:
    out_file = CASES_DIR / f"{c.case_id}.json"
    out_file.write_text(json.dumps(c.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_file}")
