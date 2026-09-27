"""Utilities for an oracle semantic-credit experiment.

This module intentionally does not approximate CUDA Agent's critic. A faithful
token-level GAE baseline requires value estimates from the critic for the actual
rollout. When those values are unavailable, the comparison is not defined and
must not be replaced by a positional decay heuristic.

The oracle side is only a representation of a human-specified intervention:
which generated-token positions correspond to an optimization decision.
Whether emphasizing those positions improves learning remains an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class CodeSpan:
    """Character span in the actual raw model response.

    start_char is inclusive and end_char is exclusive. token_indices is
    populated only after alignment against offsets from the real model
    tokenizer used for that response.
    """

    start_char: int
    end_char: int
    description: str
    token_indices: list[int] = field(default_factory=list)


@dataclass
class OracleDecision:
    decision_id: str
    category: str
    description: str
    spans: list[CodeSpan]


def validate_token_offsets(raw_response: str, token_offsets: list[tuple[int, int]]) -> None:
    """Validate tokenizer-provided character offsets for one raw response."""

    previous_end = 0
    for index, (start, end) in enumerate(token_offsets):
        if not (0 <= start <= end <= len(raw_response)):
            raise ValueError(f"token offset {index} is outside the raw response")
        if start < previous_end:
            raise ValueError("token offsets must be non-overlapping and ordered")
        previous_end = end


def align_spans_to_tokens(
    raw_response: str,
    token_offsets: list[tuple[int, int]],
    spans: Iterable[CodeSpan],
) -> None:
    """Align oracle response spans to real tokenizer offsets in-place."""

    validate_token_offsets(raw_response, token_offsets)
    for span in spans:
        if not (0 <= span.start_char < span.end_char <= len(raw_response)):
            raise ValueError("oracle span is outside the raw response")
        span.token_indices = [
            i
            for i, (start, end) in enumerate(token_offsets)
            if not (end <= span.start_char or start >= span.end_char)
        ]
        if not span.token_indices:
            raise ValueError("oracle span does not overlap any generated token")


def optimization_token_indices(
    decisions: Iterable[OracleDecision],
    *,
    num_tokens: int,
) -> list[int]:
    """Return the sorted union of token positions covered by oracle decisions."""

    indices: set[int] = set()
    for decision in decisions:
        for span in decision.spans:
            for index in span.token_indices:
                if not 0 <= index < num_tokens:
                    raise ValueError("oracle token index is outside the generated sequence")
                indices.add(index)
    return sorted(indices)


def compute_oracle_token_signal(
    num_tokens: int,
    decisions: Iterable[OracleDecision],
    total_signal: float,
) -> list[float]:
    """Place an experimental scalar signal only on oracle decision tokens.

    Equal sharing inside the oracle span is only one controlled intervention.
    It is not claimed to be the uniquely correct credit assignment.

    An empty oracle is an error rather than a silent fallback to uniform credit.
    """

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")

    indices = optimization_token_indices(decisions, num_tokens=num_tokens)
    if not indices:
        raise ValueError("oracle semantic credit requires at least one labeled token")

    result = [0.0] * num_tokens
    per_token = total_signal / len(indices)
    for index in indices:
        result[index] = per_token
    return result


def compute_gae_advantages(
    rewards: list[float],
    values: list[float],
    *,
    gamma: float = 1.0,
    lambda_: float = 0.95,
    bootstrap_value: float = 0.0,
) -> list[float]:
    """Compute standard GAE from provided critic values.

    This is the primitive needed for a faithful CUDA-Agent-style comparison.
    It deliberately requires the critic's value estimates; it does not invent
    them. Advantages are not rewards and need not sum to terminal reward.
    """

    if len(rewards) != len(values):
        raise ValueError("rewards and values must have the same length")
    if not rewards:
        return []
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    advantages = [0.0] * len(rewards)
    next_advantage = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        next_value = bootstrap_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        next_advantage = delta + gamma * lambda_ * next_advantage
        advantages[t] = next_advantage

    return advantages


def terminal_reward_sequence(num_tokens: int, terminal_reward: float) -> list[float]:
    """Construct a sparse terminal-outcome reward sequence."""

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    rewards = [0.0] * num_tokens
    rewards[-1] = terminal_reward
    return rewards
