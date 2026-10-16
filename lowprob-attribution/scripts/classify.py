"""Token classification: tag each token with a type and its generation context.

Token types: NL_reasoning / generic_code / triton_specific / cuda_specific /
number / identifier / tool_syntax.

Context types: initial / after_compile_error / after_correctness_fail /
after_profiling / after_tool.

Two fields are classified differently:

* `content` is split into prose and fenced-code segments, then each token is
  typed from its own text.
* `reasoning_content` is prose, so it stays NL_reasoning by default, but a token
  is re-tagged when it is a marker no prose sentence would contain (CUDA
  built-ins, `tl`, `num_warps`, `BLOCK_SIZE`, ...). This keeps reasoning from
  swallowing the CUDA/Triton vocabulary the study is about, without misfiling
  ordinary words as code.

Observables per token (PLAN.md section 9): the logprob itself, `in_topn`
(whether the chosen token appears in the returned top_logprobs), and lowprob
flags at two thresholds. The -9999 sentinel is deliberately NOT used: DeepSeek
returns the real logprob for a token outside top-N, and the sentinel rate is
0% at every top_logprobs setting tested.

Classification rules and keyword sets are frozen. Inaccuracies are reported as
limitations rather than patched once collection has started.
"""

from __future__ import annotations

import re

# --- keyword sets (frozen) ---

TRITON_KEYWORDS = {
    # triton.language primitives
    "tl", "load", "store", "dot", "sum", "max", "min", "reduce",
    "arange", "zeros", "full", "expand_dims", "broadcast_to",
    "where", "view", "reshape", "trans", "ravel",
    "static_assert", "constexpr", "program_id",
    "atomic_add", "atomic_cas", "atomic_max", "atomic_min",
    "cast", "to",
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

# generic code tokens - explicitly NOT triton/cuda specific
GENERIC_CODE_TOKENS = {
    "for", "if", "else", "elif", "while", "def", "class", "import",
    "from", "return", "yield", "raise", "try", "except", "finally",
    "with", "as", "in", "not", "and", "or", "is", "None", "True", "False",
    "lambda", "pass", "break", "continue", "assert", "del", "global",
    "nonlocal", "async", "await",
    "int", "float", "double", "void", "char", "bool", "unsigned",
    "long", "short", "signed", "const", "static", "struct", "typedef",
    "sizeof", "auto", "template", "typename", "namespace", "using",
    "public", "private", "protected",
    "torch", "tensor", "cuda", "empty", "zeros", "ones", "randn",
    "cat", "stack", "reshape", "view", "permute", "contiguous",
    "float16", "float32", "float64", "bfloat16", "half",
    "int32", "int64", "uint8", "int8",
}

# Words like "sum", "max", "min", "to", "load", "store", "where", "view" belong to
# the Triton keyword set but also occur constantly in ordinary prose, so they must
# NOT be used to detect code inside reasoning. These markers are narrow on purpose:
# they are C/C++ identifier styles or Triton-only spellings that a prose sentence
# essentially never contains.
NL_CODE_MARKERS_CUDA = CUDA_KEYWORDS
NL_CODE_MARKERS_TRITON = {
    "tl", "num_warps", "num_stages", "BLOCK_SIZE", "BLOCK",
    "constexpr", "program_id", "autotune", "triton",
}

LOWPROB_THRESHOLDS = (-2.0, -5.0)

# regex helpers
RE_NUMBER = re.compile(r"^\d+(\.\d+)?([eE][+-]?\d+)?$")
RE_IDENTIFIER = re.compile(r"^[a-zA-Z_]\w*$")
RE_PUNCT_OR_WS = re.compile(r"^[\s\{\}\[\]\(\)\"':,;]+$")


def classify_token_type(token: str, segment_type: str) -> str:
    """Classify a single token. segment_type is 'nl', 'code' or 'tool'."""
    if segment_type == "nl":
        # Reasoning is prose, but it quotes code while planning a kernel. A token
        # is only re-tagged when it is a marker no prose sentence would contain;
        # numbers and identifiers stay NL_reasoning so the category keeps its
        # meaning ("language the model wrote while thinking").
        bare = token.lstrip("@")
        if bare in NL_CODE_MARKERS_CUDA:
            return "cuda_specific"
        if bare in NL_CODE_MARKERS_TRITON:
            return "triton_specific"
        return "NL_reasoning"

    if segment_type == "tool":
        if RE_PUNCT_OR_WS.match(token):
            return "tool_syntax"
        if RE_NUMBER.match(token):
            return "number"
        if RE_IDENTIFIER.match(token):
            return "identifier"
        return "tool_syntax"

    bare = token.lstrip("@")

    if RE_NUMBER.match(token):
        return "number"

    # cuda first (more specific), then triton, then generic
    if bare in CUDA_KEYWORDS or any(
        bare.startswith(kw) for kw in CUDA_KEYWORDS if "." not in kw and len(kw) > 4
    ):
        return "cuda_specific"

    if any(part in TRITON_KEYWORDS for part in bare.split(".")):
        return "triton_specific"

    if bare in GENERIC_CODE_TOKENS:
        return "generic_code"

    if RE_IDENTIFIER.match(token):
        return "identifier"

    return "generic_code"


def in_topn(entry: dict) -> bool:
    """Whether the chosen token is present in the returned top_logprobs list."""
    token = entry.get("token")
    return any(alt.get("token") == token for alt in (entry.get("top_logprobs") or []))


def parse_response_segments(text: str) -> list[tuple[str, str]]:
    """Split text into (segment_type, segment_text) pairs, handling ``` fences."""
    segments = []
    for part in re.split(r"(```[\s\S]*?```)", text or ""):
        if not part.strip():
            continue
        if part.startswith("```"):
            lines = part.split("\n")
            inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            segments.append(("code", inner))
        else:
            segments.append(("nl", part))
    return segments


def segment_bounds(text: str) -> list[tuple[int, int, str]]:
    """Character ranges of the segments inside `text`, in text order."""
    bounds = []
    cursor = 0
    for segment_type, segment_text in parse_response_segments(text):
        index = text.find(segment_text, cursor)
        if index < 0:
            continue
        bounds.append((index, index + len(segment_text), segment_type))
        cursor = index + len(segment_text)
    return bounds


def annotate_tokens(logprobs: list[dict], text: str, field: str,
                    turn_index: int, context_type: str,
                    tool_start: int | None = None) -> list[dict]:
    """One annotated record per token.

    The token is located in `text` by advancing a cursor, so segment assignment
    follows the token stream rather than substring search (the earlier version
    matched any segment merely containing the token, which misfiled most code
    tokens as natural language).

    `tool_start` is the character offset where tool-call arguments begin, for
    fields that carry them after the visible text (vLLM puts both in one
    logprob stream). Tokens at or past that offset are tool output, not prose.
    """
    if field == "reasoning_content":
        # Reasoning is the model thinking; it is not a code artifact even when it
        # quotes code, and the whole field belongs to one segment.
        bounds = [(0, max(len(text or ""), 1), "nl")]
    else:
        bounds = segment_bounds(text or "")
        if tool_start is not None:
            # Truncate the prose segments at the argument boundary, then append
            # the tool region. Appending without truncating would leave a
            # full-text prose span starting at 0 that always matches first and
            # shadows the tool span entirely.
            bounds = [(start, min(end, tool_start), kind)
                      for start, end, kind in bounds if start < tool_start]
            bounds.append((tool_start, max(len(text or ""), tool_start + 1), "tool"))
            bounds.sort()

    annotated = []
    cursor = 0
    for entry in logprobs:
        token = entry.get("token") or ""
        index = text.find(token, cursor) if text else -1
        if index < 0:
            index = cursor
        cursor = index + len(token)

        segment_type = "nl"
        for start, end, kind in bounds:
            if start <= index < end:
                segment_type = kind
                break

        logprob = entry.get("logprob")
        record = {
            "token": token,
            "logprob": logprob,
            "in_topn": in_topn(entry),
            "token_type": classify_token_type(token, segment_type),
            "context_type": context_type,
            "turn": turn_index,
            "logprob_field": field,
        }
        for threshold in LOWPROB_THRESHOLDS:
            record[f"lowprob_{abs(threshold):g}"] = (
                logprob is not None and logprob <= threshold
            )
        annotated.append(record)
    return annotated
