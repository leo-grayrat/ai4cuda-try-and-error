"""Conservative mapping from applied source lines to sampled response tokens."""

from __future__ import annotations

from copy import deepcopy
import io
import tokenize

from kernel_lab.code_credit import _python_docstring_lines
from kernel_lab.core import source_hash


def map_response_to_actions(
    child_source: str,
    raw_response: str,
    token_offsets: list[tuple[int, int]],
    actions: list[dict],
) -> dict:
    """Map only an exact, unique full-source occurrence in the raw response.

    A changed line absent from the emitted source (for example, a deletion)
    receives no fabricated token credit. ``token_offsets`` are response-local
    character offsets from the tokenizer used during generation.
    """
    if not child_source:
        raise ValueError("child_source must not be empty")
    if any(start < 0 or end < start or end > len(raw_response) for start, end in token_offsets):
        raise ValueError("token offsets are outside the raw response")
    occurrences = raw_response.count(child_source)
    status = ("exact_source_match" if occurrences == 1 else
              "missing_source_match" if occurrences == 0 else "ambiguous_source_match")
    result = {"status": status, "actions": {}}
    if occurrences != 1:
        for action in actions:
            result["actions"][action["id"]] = {
                "status": "unmapped", "char_span": None, "char_spans": [], "token_indices": [],
            }
        return result
    source_start = raw_response.find(child_source)
    lines = child_source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    docstring_lines = _python_docstring_lines(child_source)
    comment_starts: dict[int, int] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(child_source).readline):
            if token.type == tokenize.COMMENT:
                row, column = token.start
                comment_starts[row] = min(column, comment_starts.get(row, column))
    except (tokenize.TokenError, IndentationError):
        result["status"] = "unparseable_source"
        for action in actions:
            result["actions"][action["id"]] = {
                "status": "unmapped", "char_span": None, "char_spans": [], "token_indices": [],
            }
        return result
    for action in actions:
        spans = []
        has_deletion = False
        for edit in action["edits"]:
            first, last = edit["child_lines"]
            if first > last:
                has_deletion = True
                continue
            if first < 1 or last > len(lines):
                raise ValueError("Action line span is outside applied source")
            for line_number in range(first, last + 1):
                if line_number in docstring_lines:
                    continue
                left = starts[line_number - 1]
                right = starts[line_number]
                if line_number in comment_starts:
                    right = min(right, left + comment_starts[line_number])
                if child_source[left:right].strip():
                    spans.append([source_start + left, source_start + right])
        if not spans:
            result["actions"][action["id"]] = {
                "status": "no_emitted_lines" if has_deletion else "no_code_tokens",
                "char_span": None, "char_spans": [], "token_indices": [],
            }
            continue
        token_indices = [index for index, (start, end) in enumerate(token_offsets)
                         if end > start and any(left <= start and end <= right for left, right in spans)]
        result["actions"][action["id"]] = {
            "status": "partially_mapped" if has_deletion and token_indices else
                      "mapped" if token_indices else "unmapped",
            "char_span": spans[0] if len(spans) == 1 else None,
            "char_spans": spans,
            "token_indices": token_indices,
        }
    return result


def record_generation(
    trace: dict,
    child_source: str,
    raw_response: str,
    token_ids: list[int],
    token_offsets: list[tuple[int, int]],
) -> dict:
    """Attach actual sampled tokens to an exact prepared child source.

    The caller supplies tokenizer offsets from generation. This function does
    not retokenize text after the fact or infer any missing token identity.
    """
    if source_hash(child_source.encode()) != trace["source_sha256"]["child"]:
        raise ValueError("Prepared child source hash differs")
    if len(token_ids) != len(token_offsets):
        raise ValueError("token_ids and token_offsets must have the same length")
    if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in token_ids):
        raise ValueError("Every token id must be an integer")
    if any(start < 0 or end < start or end > len(raw_response) for start, end in token_offsets):
        raise ValueError("token offsets are outside the raw response")
    if any(previous[1] > current[0] for previous, current in zip(token_offsets, token_offsets[1:])):
        raise ValueError("token offsets must be ordered and nonoverlapping")
    recorded = deepcopy(trace)
    recorded["raw_response"] = raw_response
    recorded["response_tokens"] = [
        {"id": token_id, "start": start, "end": end}
        for token_id, (start, end) in zip(token_ids, token_offsets)
    ]
    recorded["source_to_response"] = map_response_to_actions(
        child_source, raw_response, token_offsets, trace["actions"]
    )
    return recorded
