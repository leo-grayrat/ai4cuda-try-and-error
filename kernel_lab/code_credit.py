"""[FROZEN - HEURISTIC BASELINE]
Locate candidate edits in a parent/child kernel revision using diffs and AST heuristics.

CRITICAL STATUS:
This module is strictly frozen as a naive heuristic baseline. It represents
"how far can manual rules/diffs guess key optimizations without program semantic understanding".
DO NOT patch, repair, or add new heuristics here when false positives or false negatives occur;
observed failures serve as baseline ablation evidence, not software bugs to fix.
"""

from __future__ import annotations

import ast
import difflib
import re


_RUNTIME_CONFIG = re.compile(r"(?:torch\.backends|allow_tf32|cudnn\.|set_float32_matmul_precision)")
_PRECISION = re.compile(r"(?:autocast|\b(?:float16|bfloat16|float32|fp16|bf16|tf32)\b|\.half\()", re.IGNORECASE)
_MEMORY = re.compile(r"(?:tl\.(?:load|store)|__shared__|shared_memory|cudaMemcpy|ldg\b|stg\b)")
_SYNC = re.compile(r"(?:__syncthreads|__syncwarp|tl\.debug_barrier|atomic(?:Add|CAS|Max|Min)?)")
_LAUNCH = re.compile(r"(?:<<<|>>>|num_warps|num_stages|blockDim|grid\s*=|threads_per_block|\b\w*kernel\s*\[)")
_TRANSFER = re.compile(r"(?:\.cuda\(|\.to\(|\.is_cuda\b|cudaMemcpy|\.device\b)")
_HOST_COMPUTE = re.compile(r"(?:torch\.|\.relu_?\(|\.matmul\(|\bmatmul\b|\bmm\(|\.softmax\(|\.sum\(|\.view\(|\.reshape\(|\.contiguous\(|\.launch\(|\bhasattr\(|\.register_buffer\()")


def _without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith(("#", "//")):
            continue
        lines.append(re.split(r"\s(?:#|//)", line, maxsplit=1)[0])
    return "\n".join(lines)


def _python_kernel_lines(source: str) -> set[int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    result: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorator_text = " ".join(ast.unparse(item) for item in node.decorator_list)
        kernel_calls = any(
            isinstance(child, ast.Call) and ast.unparse(child.func) in {"tl.load", "tl.store"}
            for child in ast.walk(node)
        )
        is_kernel = ("triton.jit" in decorator_text or "cuda.jit" in decorator_text
                     or "__global__" in decorator_text or kernel_calls)
        if is_kernel:
            result.update(range(node.lineno, node.end_lineno + 1))
    return result


def _python_scope(source: str, line: int) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "<module>"
    matches = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.lineno <= line <= node.end_lineno]
    if not matches:
        return "<module>"
    node = min(matches, key=lambda item: item.end_lineno - item.lineno)
    return f"{node.name}@{node.lineno}"


def _python_docstring_lines(source: str) -> set[int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    result: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            result.update(range(first.lineno, first.end_lineno + 1))
    return result


def _active_slice_text(lines: list[str], start: int, end: int, excluded_lines: set[int]) -> str:
    return _without_comments("".join(
        line for line_number, line in enumerate(lines[start:end], start + 1)
        if line_number not in excluded_lines
    ))


def _cuda_kernel_lines(source: str) -> set[int]:
    """Find CUDA kernel bodies with brace depth; malformed source stays unmarked."""
    result: set[int] = set()
    for match in re.finditer(r"\b__(?:global|device)__\b", source):
        opening = source.find("{", match.end())
        if opening < 0 or ";" in source[match.end():opening]:
            continue
        depth = 0
        closing = None
        for index in range(opening, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing is None:
            continue
        start_line = source.count("\n", 0, match.start()) + 1
        end_line = source.count("\n", 0, closing) + 1
        result.update(range(start_line, end_line + 1))
    return result


def _classify(text: str, touched_lines: range, kernel_lines: set[int]) -> tuple[str, int]:
    text = _without_comments(text)
    if not text.strip():
        return "boilerplate", 0
    if _RUNTIME_CONFIG.search(text):
        return "runtime_config", 45
    if _PRECISION.search(text):
        return "precision", 88
    if _SYNC.search(text):
        return "kernel_sync", 95
    if _MEMORY.search(text):
        return "kernel_memory", 95
    if _LAUNCH.search(text):
        return "kernel_launch", 90
    if _TRANSFER.search(text):
        return "host_transfer", 70
    if any(line in kernel_lines for line in touched_lines):
        return "kernel_compute", 80
    if _HOST_COMPUTE.search(text):
        return "host_compute", 60
    return "boilerplate", 0


def locate_edits(parent: str, child: str, *, language: str) -> list[dict]:
    """Return changed line units ordered by code-derived performance relevance.

    Line ranges are inclusive and one-based. An empty side of an insertion or
    deletion uses ``[start, start - 1]``. The original zero-based slices are
    retained to construct an exact single-unit revert.
    """
    if language not in {"python", "cuda"}:
        raise ValueError("language must be 'python' or 'cuda'")
    before = parent.splitlines(keepends=True)
    after = child.splitlines(keepends=True)
    before_docs = _python_docstring_lines(parent) if language == "python" else set()
    after_docs = _python_docstring_lines(child) if language == "python" else set()
    kernel_lines = (_python_kernel_lines(child) | _python_kernel_lines(parent)
                    if language == "python" else _cuda_kernel_lines(child) | _cuda_kernel_lines(parent))
    units = []
    for index, (tag, i1, i2, j1, j2) in enumerate(
        difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes()
    ):
        if tag == "equal":
            continue
        changed_text = (_active_slice_text(before, i1, i2, before_docs)
                        + _active_slice_text(after, j1, j2, after_docs))
        touched = range(j1 + 1, j2 + 1) if j1 != j2 else range(i1 + 1, i2 + 1)
        category, score = _classify(changed_text, touched, kernel_lines)
        units.append({
            "id": f"edit-{index}",
            "operation": tag,
            "category": category,
            "score": score,
            "parent_lines": [i1 + 1, i2],
            "child_lines": [j1 + 1, j2],
            "parent_slice": [i1, i2],
            "child_slice": [j1, j2],
            "changed_lines": max(i2 - i1, j2 - j1),
        })
    return sorted(units, key=lambda unit: (-unit["score"], unit["child_lines"][0]))


def locate_actions(parent: str, child: str, *, language: str) -> list[dict]:
    """Group edits that share a function or a newly introduced identifier.

    Groups are hypotheses about one transformation. Their independence is
    unknown until a reverted candidate compiles and passes the verifier.
    """
    units = locate_edits(parent, child, language=language)
    if not units:
        return []
    before, after = parent.splitlines(keepends=True), child.splitlines(keepends=True)
    before_docs = _python_docstring_lines(parent) if language == "python" else set()
    after_docs = _python_docstring_lines(child) if language == "python" else set()
    scopes = []
    introduced = []
    for unit in units:
        i1, i2 = unit["parent_slice"]
        j1, j2 = unit["child_slice"]
        before_text = _active_slice_text(before, i1, i2, before_docs)
        after_text = _active_slice_text(after, j1, j2, after_docs)
        old_ids = set(re.findall(r"\b[A-Za-z_]\w*\b", before_text))
        new_ids = set(re.findall(r"\b[A-Za-z_]\w*\b", after_text))
        introduced.append({name for name in new_ids - old_ids if len(name) >= 4})
        line = j1 + 1 if j1 < j2 else i1 + 1
        source = child if j1 < j2 else parent
        scopes.append(_python_scope(source, line) if language == "python" else "<unknown>")

    leaders = list(range(len(units)))

    def find(index: int) -> int:
        while leaders[index] != index:
            leaders[index] = leaders[leaders[index]]
            index = leaders[index]
        return index

    def join(left: int, right: int) -> None:
        leaders[find(right)] = find(left)

    for left in range(len(units)):
        for right in range(left + 1, len(units)):
            same_function = (scopes[left] == scopes[right] != "<module>"
                             and units[left]["score"] > 0 and units[right]["score"] > 0)
            shared_new_name = bool(introduced[left] & introduced[right])
            if same_function or shared_new_name:
                join(left, right)
    groups: dict[int, list[dict]] = {}
    for index, unit in enumerate(units):
        groups.setdefault(find(index), []).append(unit)
    actions = []
    for members in groups.values():
        members.sort(key=lambda unit: unit["child_lines"][0])
        best = max(members, key=lambda unit: unit["score"])
        actions.append({
            "id": f"action-{members[0]['id']}",
            "category": best["category"],
            "score": best["score"],
            "edits": members,
            "independence": "unverified",
        })
    return sorted(actions, key=lambda action: (-action["score"], action["edits"][0]["child_lines"][0]))


def revert_edit(parent: str, child: str, unit: dict) -> str:
    """Undo one recorded edit in the child, preserving all other edits."""
    before = parent.splitlines(keepends=True)
    after = child.splitlines(keepends=True)
    i1, i2 = unit["parent_slice"]
    j1, j2 = unit["child_slice"]
    if not (0 <= i1 <= i2 <= len(before) and 0 <= j1 <= j2 <= len(after)):
        raise ValueError("Edit slices fall outside the source")
    return "".join(after[:j1] + before[i1:i2] + after[j2:])


def revert_action(parent: str, child: str, action: dict) -> str:
    """Undo every edit in an action, using original child coordinates."""
    before = parent.splitlines(keepends=True)
    after = child.splitlines(keepends=True)
    for unit in sorted(action["edits"], key=lambda item: item["child_slice"][0], reverse=True):
        i1, i2 = unit["parent_slice"]
        j1, j2 = unit["child_slice"]
        after[j1:j2] = before[i1:i2]
    return "".join(after)
