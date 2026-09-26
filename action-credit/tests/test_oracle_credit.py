"""Tests for Oracle semantic credit assignment and baseline credit generator."""

import json
import math
from pathlib import Path

from kernel_lab.oracle_credit import (
    CodeSpan,
    OracleCase,
    OracleDecision,
    align_spans_to_tokens,
    analyze_credit_discrepancy,
    compute_baseline_credit,
    compute_oracle_credit,
    simple_tokenize_with_offsets,
)


def test_simple_tokenize_with_offsets() -> None:
    code = "out = a.detach() + 10\n"
    tokens, offsets = simple_tokenize_with_offsets(code)
    assert len(tokens) == len(offsets)
    # verify offsets match substring exactly
    for tok, (s, e) in zip(tokens, offsets):
        assert code[s:e] == tok


def test_credit_conservation_and_targeting() -> None:
    code = "def forward(x):\n    t = x.detach()\n    return torch.relu(t)\n"
    tokens, offsets = simple_tokenize_with_offsets(code)
    target = "t = x.detach()"
    s = code.find(target)
    e = s + len(target)

    span = CodeSpan(start_char=s, end_char=e, description="Eliminate clone")
    align_spans_to_tokens(offsets, [span])
    assert len(span.token_indices) > 0

    decision = OracleDecision(
        decision_id="d1",
        category="memory",
        description="Eliminate clone",
        spans=[span],
        is_optimization=True,
    )

    reward = 2.0
    baseline_credit = compute_baseline_credit(len(tokens), reward)
    oracle_credit = compute_oracle_credit(len(tokens), [decision], reward)

    # 1. Total reward strictly conserved
    assert math.isclose(sum(baseline_credit), reward, rel_tol=1e-5)
    assert math.isclose(sum(oracle_credit), reward, rel_tol=1e-5)

    # 2. Oracle credit only placed on decision tokens
    for i in range(len(tokens)):
        if i in span.token_indices:
            assert oracle_credit[i] > 0.0
        else:
            assert oracle_credit[i] == 0.0

    # 3. Discrepancy analysis
    analysis = analyze_credit_discrepancy(tokens, [decision], baseline_credit, oracle_credit)
    assert analysis["total_reward_conserved"] is True
    assert analysis["oracle"]["boilerplate_credit"] == 0.0
    # In baseline, boilerplate tokens (imports, def, return, etc.) eat up a large portion of credit
    assert analysis["baseline"]["boilerplate_credit"] > 0.0


def test_oracle_cases_from_disk() -> None:
    cases_dir = Path("experiments/oracle-cases")
    json_files = list(cases_dir.glob("case_*.json"))
    assert len(json_files) == 3

    for p in json_files:
        data = json.loads(p.read_text(encoding="utf-8"))
        case = OracleCase.from_dict(data)
        assert len(case.tokens) > 0
        assert len(case.decisions) > 0
        assert case.terminal_reward > 0.0

        baseline = compute_baseline_credit(len(case.tokens), case.terminal_reward)
        oracle = compute_oracle_credit(len(case.tokens), case.decisions, case.terminal_reward)

        analysis = analyze_credit_discrepancy(case.tokens, case.decisions, baseline, oracle)
        assert analysis["total_reward_conserved"] is True
        # Oracle concentrates 100% on optimization tokens
        assert math.isclose(analysis["oracle"]["optimization_ratio"], 1.0, rel_tol=1e-4)
        # Baseline assigns the vast majority of credit to boilerplate / glue tokens
        assert analysis["baseline"]["boilerplate_ratio"] > 0.5
