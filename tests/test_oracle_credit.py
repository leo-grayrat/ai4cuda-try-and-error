"""Tests for the minimal, non-fabricated oracle-credit primitives."""

import math

from kernel_lab.oracle_credit import (
    CodeSpan,
    OracleDecision,
    align_spans_to_tokens,
    compute_gae_advantages,
    compute_oracle_token_signal,
    terminal_reward_sequence,
)


def test_oracle_alignment_uses_raw_response_offsets() -> None:
    raw = "prefix code suffix"
    offsets = [(0, 6), (7, 11), (12, 18)]
    span = CodeSpan(7, 11, "the generated code token")

    align_spans_to_tokens(raw, offsets, [span])

    assert span.token_indices == [1]


def test_oracle_signal_targets_only_labeled_tokens() -> None:
    span = CodeSpan(0, 1, "decision", token_indices=[1, 2])
    decision = OracleDecision("d1", "memory", "remove a copy", [span])

    signal = compute_oracle_token_signal(5, [decision], total_signal=2.0)

    assert signal == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert math.isclose(sum(signal), 2.0)


def test_empty_oracle_is_not_silently_replaced_by_uniform_credit() -> None:
    try:
        compute_oracle_token_signal(4, [], total_signal=1.0)
    except ValueError as exc:
        assert "at least one labeled token" in str(exc)
    else:
        raise AssertionError("an empty oracle must fail")


def test_gae_with_zero_values_reduces_to_position_decay() -> None:
    rewards = terminal_reward_sequence(4, 2.0)
    advantages = compute_gae_advantages(
        rewards,
        [0.0, 0.0, 0.0, 0.0],
        gamma=1.0,
        lambda_=0.95,
    )

    assert advantages == [
        2.0 * 0.95**3,
        2.0 * 0.95**2,
        2.0 * 0.95,
        2.0,
    ]


def test_critic_values_change_gae_and_break_reward_conservation_assumption() -> None:
    rewards = terminal_reward_sequence(3, 2.0)
    advantages = compute_gae_advantages(
        rewards,
        [0.5, 1.0, 1.5],
        gamma=1.0,
        lambda_=0.95,
    )
    zero_value_advantages = compute_gae_advantages(
        rewards,
        [0.0, 0.0, 0.0],
        gamma=1.0,
        lambda_=0.95,
    )

    assert advantages != zero_value_advantages
    assert not math.isclose(sum(advantages), 2.0)
