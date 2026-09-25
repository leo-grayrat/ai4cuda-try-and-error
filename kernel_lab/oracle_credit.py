"""Oracle semantic credit assignment vs. terminal outcome token credit.

This module implements the core contrast:
1. Baseline (CUDA Agent style):
   Terminal performance reward R placed at the final token, diffused via
   discount/GAE back through the token sequence.
2. Oracle Semantic Credit:
   The same total performance reward R strictly conserved, but distributed
   directly across token spans corresponding to human-identified optimization decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any


@dataclass
class CodeSpan:
    start_char: int
    end_char: int
    description: str
    is_optimization: bool = True
    token_indices: list[int] = field(default_factory=list)


@dataclass
class OracleDecision:
    decision_id: str
    category: str  # e.g. "memory_elimination", "loop_unroll", "kernel_fusion", "tiling"
    description: str
    spans: list[CodeSpan]
    is_optimization: bool = True


@dataclass
class OracleCase:
    case_id: str
    title: str
    benchmark_task: str
    parent_source: str
    child_source: str
    raw_response: str
    tokens: list[str]
    token_offsets: list[tuple[int, int]]
    decisions: list[OracleDecision]
    terminal_reward: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OracleCase:
        decisions = []
        for d in data.get("decisions", []):
            spans = [CodeSpan(**s) for s in d.get("spans", [])]
            decisions.append(
                OracleDecision(
                    decision_id=d["decision_id"],
                    category=d["category"],
                    description=d["description"],
                    spans=spans,
                    is_optimization=d.get("is_optimization", True),
                )
            )
        return cls(
            case_id=data["case_id"],
            title=data["title"],
            benchmark_task=data["benchmark_task"],
            parent_source=data["parent_source"],
            child_source=data["child_source"],
            raw_response=data["raw_response"],
            tokens=data["tokens"],
            token_offsets=[tuple(x) for x in data["token_offsets"]],
            decisions=decisions,
            terminal_reward=float(data["terminal_reward"]),
            metadata=data.get("metadata", {}),
        )


def simple_tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Tokenize source text into whitespace/punctuation tokens with exact character offsets.
    
    This provides a standalone, deterministic tokenization for code inspection
    when no external model tokenizer is attached.
    """
    import re
    # Match identifiers/keywords, numeric literals, operators/punctuation, or newlines
    pattern = re.compile(r"[a-zA-Z_]\w*|\d+(?:\.\d+)?|[^\s\w]|\n")
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in pattern.finditer(text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def align_spans_to_tokens(
    token_offsets: list[tuple[int, int]],
    spans: list[CodeSpan],
) -> None:
    """Map character spans to token indices in-place."""
    for span in spans:
        span.token_indices = [
            i
            for i, (start, end) in enumerate(token_offsets)
            if not (end <= span.start_char or start >= span.end_char)
        ]


def compute_baseline_credit(
    num_tokens: int,
    terminal_reward: float,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> list[float]:
    """Compute baseline token advantages from a single terminal outcome reward.
    
    In CUDA Agent / standard sparse-reward RL:
      r_t = 0 for t < T - 1, and r_{T-1} = terminal_reward.
    Under GAE(gamma, lambda), the backward credit propagation decays as:
      advantage[t] proportional to (gamma * lambda) ** (T - 1 - t).
    We normalize the sum to equal terminal_reward so that total reward is strictly conserved.
    """
    if num_tokens <= 0:
        return []
    if math.isclose(terminal_reward, 0.0):
        return [0.0] * num_tokens

    decay = gamma * lambda_
    raw_weights = [decay ** (num_tokens - 1 - t) for t in range(num_tokens)]
    total_raw = sum(raw_weights)
    if total_raw == 0:
        return [0.0] * num_tokens

    scale = terminal_reward / total_raw
    return [w * scale for w in raw_weights]


def compute_oracle_credit(
    num_tokens: int,
    decisions: list[OracleDecision],
    terminal_reward: float,
) -> list[float]:
    """Distribute the exact same total terminal reward strictly to optimization decision spans.
    
    Total reward is strictly conserved: sum(oracle_credit) == terminal_reward.
    Non-optimization / boilerplate tokens receive 0.0 performance credit.
    """
    if num_tokens <= 0:
        return []
    if math.isclose(terminal_reward, 0.0):
        return [0.0] * num_tokens

    # Collect all token indices that belong to true optimization decisions
    opt_token_set: set[int] = set()
    for decision in decisions:
        if decision.is_optimization:
            for span in decision.spans:
                if span.is_optimization:
                    for idx in span.token_indices:
                        if 0 <= idx < num_tokens:
                            opt_token_set.add(idx)

    if not opt_token_set:
        # Fallback if no spans marked: uniform distribution
        uniform_val = terminal_reward / num_tokens
        return [uniform_val] * num_tokens

    credit = [0.0] * num_tokens
    per_token_reward = terminal_reward / len(opt_token_set)
    for idx in opt_token_set:
        credit[idx] = per_token_reward

    return credit


def analyze_credit_discrepancy(
    tokens: list[str],
    decisions: list[OracleDecision],
    baseline_credit: list[float],
    oracle_credit: list[float],
) -> dict[str, Any]:
    """Analyze how credit is allocated between true optimization tokens and boilerplate tokens."""
    num_tokens = len(tokens)
    assert len(baseline_credit) == num_tokens
    assert len(oracle_credit) == num_tokens

    opt_indices: set[int] = set()
    for decision in decisions:
        if decision.is_optimization:
            for span in decision.spans:
                if span.is_optimization:
                    for idx in span.token_indices:
                        if 0 <= idx < num_tokens:
                            opt_indices.add(idx)

    total_baseline = sum(baseline_credit)
    total_oracle = sum(oracle_credit)

    baseline_opt_credit = sum(baseline_credit[i] for i in opt_indices)
    baseline_boilerplate_credit = total_baseline - baseline_opt_credit

    oracle_opt_credit = sum(oracle_credit[i] for i in opt_indices)
    oracle_boilerplate_credit = total_oracle - oracle_opt_credit

    return {
        "num_tokens": num_tokens,
        "num_optimization_tokens": len(opt_indices),
        "num_boilerplate_tokens": num_tokens - len(opt_indices),
        "total_reward": total_baseline,
        "total_reward_conserved": math.isclose(total_baseline, total_oracle, rel_tol=1e-5),
        "baseline": {
            "optimization_credit": baseline_opt_credit,
            "boilerplate_credit": baseline_boilerplate_credit,
            "optimization_ratio": baseline_opt_credit / total_baseline if total_baseline else 0.0,
            "boilerplate_ratio": baseline_boilerplate_credit / total_baseline if total_baseline else 0.0,
        },
        "oracle": {
            "optimization_credit": oracle_opt_credit,
            "boilerplate_credit": oracle_boilerplate_credit,
            "optimization_ratio": oracle_opt_credit / total_oracle if total_oracle else 0.0,
            "boilerplate_ratio": oracle_boilerplate_credit / total_oracle if total_oracle else 0.0,
        },
    }
