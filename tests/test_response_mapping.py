"""Generated-token evidence is recorded only where source alignment is exact."""

import pytest

from kernel_lab.core import source_hash
from kernel_lab.response_mapping import map_response_to_actions, record_generation


def test_maps_changed_source_line_to_exact_response_tokens() -> None:
    source = "def kernel(x):\n    return x + 1\n"
    raw = "```python\n" + source + "```"
    offsets = [(i, i + 1) for i in range(len(raw))]
    actions = [{"id": "a", "edits": [{"child_lines": [2, 2]}]}]
    result = map_response_to_actions(source, raw, offsets, actions)
    assert result["status"] == "exact_source_match"
    assert result["actions"]["a"]["status"] == "mapped"
    assert result["actions"]["a"]["token_indices"]
    assert raw[result["actions"]["a"]["char_span"][0]:result["actions"]["a"]["char_span"][1]] == "    return x + 1\n"


def test_ambiguous_or_missing_source_is_not_mapped() -> None:
    source = "return x + 1\n"
    raw = source + source
    actions = [{"id": "a", "edits": [{"child_lines": [1, 1]}]}]
    result = map_response_to_actions(source, raw, [(0, len(raw))], actions)
    assert result["status"] == "ambiguous_source_match"
    assert result["actions"]["a"]["status"] == "unmapped"
    assert result["actions"]["a"]["token_indices"] == []


def test_deletion_without_child_lines_has_no_emitted_code_tokens() -> None:
    source = "def kernel(x):\n    return x\n"
    actions = [{"id": "a", "edits": [{"child_lines": [2, 1]}]}]
    result = map_response_to_actions(source, source, [(i, i + 1) for i in range(len(source))], actions)
    assert result["actions"]["a"]["status"] == "no_emitted_lines"


def test_token_crossing_action_boundary_is_not_credited() -> None:
    source = "x = 1\ny = x + 2\n"
    actions = [{"id": "a", "edits": [{"child_lines": [2, 2]}]}]
    result = map_response_to_actions(source, source, [(0, 7), (7, len(source))], actions)
    assert result["actions"]["a"]["token_indices"] == [1]
    result = map_response_to_actions(source, source, [(0, len(source))], actions)
    assert result["actions"]["a"]["token_indices"] == []


def test_inline_comment_and_docstring_tokens_get_no_performance_credit() -> None:
    source = (
        "def kernel(x):\n"
        "    \"\"\"atomic work\"\"\"\n"
        "    return x + 1  # tl.load is faster\n"
    )
    offsets = [(i, i + 1) for i in range(len(source))]
    actions = [
        {"id": "doc", "edits": [{"child_lines": [2, 2]}]},
        {"id": "code", "edits": [{"child_lines": [3, 3]}]},
    ]
    result = map_response_to_actions(source, source, offsets, actions)
    assert result["actions"]["doc"]["token_indices"] == []
    code_indices = result["actions"]["code"]["token_indices"]
    assert code_indices
    assert max(code_indices) < source.index("# tl.load")


def test_records_raw_generation_and_exact_token_alignment() -> None:
    source = "def kernel(x):\n    return x + 1\n"
    raw = "```python\n" + source + "```"
    trace = {
        "source_sha256": {"child": source_hash(source.encode())},
        "actions": [{"id": "a", "edits": [{"child_lines": [2, 2]}]}],
        "raw_response": None,
        "response_tokens": None,
        "source_to_response": None,
    }
    result = record_generation(
        trace, source, raw, token_ids=list(range(len(raw))),
        token_offsets=[(i, i + 1) for i in range(len(raw))],
    )
    assert result["raw_response"] == raw
    assert result["response_tokens"][10] == {"id": 10, "start": 10, "end": 11}
    assert result["source_to_response"]["actions"]["a"]["status"] == "mapped"
    assert trace["raw_response"] is None


def test_record_generation_rejects_wrong_applied_source_or_missing_token_ids() -> None:
    source = "return x\n"
    trace = {
        "source_sha256": {"child": source_hash(source.encode())},
        "actions": [],
    }
    with pytest.raises(ValueError, match="child source hash"):
        record_generation(trace, "return y\n", "return y\n", [1], [(0, 9)])
    with pytest.raises(ValueError, match="same length"):
        record_generation(trace, source, source, [], [(0, len(source))])
