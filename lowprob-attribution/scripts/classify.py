"""Token classification: classify each token in an assistant response into
a token-type and a context-type, then annotate with is_sentinel (logprob == -9999).

Token types:
    NL_reasoning       — natural language text, reasoning, commentary
    generic_code       — common programming tokens: for, if, def, import, return, +, *, [], etc.
    triton_specific    — Triton keywords: tl.load, tl.store, num_warps, BLOCK_SIZE, @triton.jit, etc.
    cuda_specific      — CUDA C++ keywords: threadIdx, blockIdx, __shared__, __syncthreads, etc.
    number             — numeric literals
    identifier         — variable names, function names (alpha-numeric identifiers not above)
    tool_syntax        — JSON / tool-call structural tokens: { } [ ] " : , and whitespace inside tool args

Context types (derived from preceding step's tool result):
    initial                  — first turn (no prior tool feedback)
    after_compile_error      — preceding tool returned a compile error
    after_correctness_fail   — preceding tool returned a correctness mismatch
    after_profiling          — preceding tool returned performance feedback (correct but timing)

The classification rules are frozen here and shall not be edited after data
collection begins. Any inaccuracies will be reported as limitations.
"""

from __future__ import annotations

import re
import json
from pathlib import Path

# --- keyword sets (frozen) ---

TRITON_KEYWORDS = {
    # triton.language primitives
    "tl", "load", "store", "dot", "sum", "max", "min", "reduce",
    "arange", "zeros", "full", "expand_dims", "broadcast_to",
    "where", "view", "reshape", "trans", "ravel",
    "static_assert", "constexpr", "program_id",
    "atomic_add", "atomic_cas", "atomic_max", "atomic_min",
    "cast", "to", " extern",
    # decorators / config
    "triton", "jit", "autotune", "Config", "heuristics",
    "num_warps", "num_stages", "BLOCK",
    # common Triton variable names
    "offs", "mask", "pid", "stride",
}

CUDA_KEYWORDS = {
    # CUDA C++ built-ins
    "threadIdx", "blockIdx", "blockDim", "gridDim",
    "__shared__", "__syncthreads", "__device__", "__global__",
    "__host__", "__constant__", "__managed__",
    "__shfl_sync", "__ballot_sync", "__activemask",
    "warpSize", "atomicAdd", "atomicCAS",
    "__float2int_rn", "__int2float_rn",
    "fmaf", "__fmaf_rn",
    "smem", "syncwarp",
    # warp / cooperative
    "__cooperative_groups", "thread_block_tile",
    # tensor core
    "wmma", "mma_sync", "load_matrix_sync", "store_matrix_sync",
    "fill_fragment", "mma",
    # memory fence
    "__threadfence", "__threadfence_block", "__threadfence_system",
}

# generic code tokens — these are NOT triton/cuda specific
GENERIC_CODE_TOKENS = {
    "for", "if", "else", "elif", "while", "def", "class", "import",
    "from", "return", "yield", "raise", "try", "except", "finally",
    "with", "as", "in", "not", "and", "or", "is", "None", "True", "False",
    "lambda", "pass", "break", "continue", "assert", "del", "global",
    "nonlocal", "async", "await",
    "int", "float", "double", "void", "char", "bool", "unsigned",
    "long", "short", "signed", "const", "static", "struct", "typedef",
    "sizeof", "auto", "template", "typename", "namespace", "using",
    "public", "private", "protected", "class",
    "torch", "tensor", "cuda", "empty", "zeros", "ones", "randn",
    "cat", "stack", "reshape", "view", "permute", "contiguous",
    "float16", "float32", "float64", "bfloat16", "half",
    "int32", "int64", "uint8", "int8",
}

# regex helpers
RE_NUMBER = re.compile(r"^\d+(\.\d+)?([eE][+-]?\d+)?$")
RE_IDENTIFIER = re.compile(r"^[a-zA-Z_]\w*$")
RE_PUNCT_OR_WS = re.compile(r"^[\s\{\}\[\]\(\)\"':,;]+$")


def classify_token_type(token: str, segment_type: str) -> str:
    """Classify a single token into a token type.

    segment_type is 'nl' (natural language), 'code', or 'tool' (tool-call JSON).
    """
    if segment_type == "nl":
        return "NL_reasoning"

    if segment_type == "tool":
        if RE_PUNCT_OR_WS.match(token):
            return "tool_syntax"
        if RE_NUMBER.match(token):
            return "number"
        if RE_IDENTIFIER.match(token):
            return "identifier"
        return "tool_syntax"

    # segment_type == 'code'
    bare = token.lstrip("@")

    if RE_NUMBER.match(token):
        return "number"

    # check cuda first (more specific), then triton, then generic
    if bare in CUDA_KEYWORDS or any(bare.startswith(kw) for kw in CUDA_KEYWORDS if "." not in kw and len(kw) > 4):
        return "cuda_specific"

    # triton: check both bare token and dot-prefixed (tl.load -> "load")
    parts = bare.split(".")
    if any(p in TRITON_KEYWORDS for p in parts):
        return "triton_specific"

    if bare in GENERIC_CODE_TOKENS:
        return "generic_code"

    if RE_IDENTIFIER.match(token):
        return "identifier"

    # operators, punctuation in code
    return "generic_code"


def classify_context_type(turn_index: int, preceding_feedback: str | None) -> str:
    """Determine context type based on turn position and preceding tool feedback.

    preceding_feedback is a short tag: 'compile_error', 'correctness_fail',
    'profiling', 'success', or None.
    """
    if turn_index == 0 or preceding_feedback is None:
        return "initial"
    if preceding_feedback == "compile_error":
        return "after_compile_error"
    if preceding_feedback == "correctness_fail":
        return "after_correctness_fail"
    if preceding_feedback == "profiling":
        return "after_profiling"
    return "initial"


def is_sentinel(logprob: float) -> bool:
    """DeepSeek API returns -9999.0 when the generated token is outside top-N."""
    return logprob is not None and logprob <= -9998.0


def parse_response_segments(text: str) -> list[tuple[str, str]]:
    """Split assistant response text into (segment_type, segment_text) pairs.

    Handles markdown code fences ```...``` and inline tool-call JSON.
    Returns list of ('nl', ...), ('code', ...), ('tool', ...) segments.
    """
    segments = []
    # split on code fences
    parts = re.split(r"(```[\s\S]*?```)", text)
    for part in parts:
        if not part.strip():
            continue
        if part.startswith("```"):
            # code block: strip fence lines
            lines = part.split("\n")
            inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            segments.append(("code", inner))
        else:
            # could contain tool-call JSON inline; detect by looking for { ... }
            segments.append(("nl", part))
    return segments


def annotate_tokens(content_logprobs: list[dict], text: str, turn_index: int,
                    preceding_feedback: str | None) -> list[dict]:
    """Produce one annotated token record per token in content_logprobs.

    Each output record:
        {token, logprob, is_sentinel, token_type, context_type, turn}
    """
    ctx = classify_context_type(turn_index, preceding_feedback)
    segments = parse_response_segments(text)

    # map each token to its segment by walking through text
    # simplified: assume tokens roughly follow text order; assign segment by
    # which segment the token's position falls into
    annotated = []
    for entry in content_logprobs:
        token = entry.get("token", "")
        logprob = entry.get("logprob", -9999.0)
        # find segment: try each segment, assign to first code segment that contains token
        seg_type = "nl"
        for stype, stext in segments:
            if token.strip() and token.strip() in stext:
                seg_type = stype
                break
        ttype = classify_token_type(token, seg_type)
        annotated.append({
            "token": token,
            "logprob": logprob,
            "is_sentinel": is_sentinel(logprob),
            "token_type": ttype,
            "context_type": ctx,
            "turn": turn_index,
        })
    return annotated
