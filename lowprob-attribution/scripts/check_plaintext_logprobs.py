"""Minimal check: does the plain-text route put code tokens under `logprobs`?

The whole reason for leaving the agent framework is that tool-call arguments are
not covered by `logprobs` while message.content is. So the question this answers
is narrow and binary:

    if the model writes the kernel in its reply body (no tools offered),
    do all of that code's tokens come back with a logprob?

If yes, the route is viable: the model keeps its real behaviour (writing code in
the reply), and the measurement we need exists. If no, the route is dead.

Usage:
    python check_plaintext_logprobs.py --model deepseek-flash
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.deepseek.com/v1/chat/completions"
TASK = Path(__file__).resolve().parents[1] / ".." / "cuda-skill-efficiency" / "tasks" / "row_reduce"

PROMPT = """You are a CUDA/Triton kernel engineer.

Implement `run(x)` in `candidate.py` for the computation in `reference.py`: sum
each row of a contiguous CUDA float32 matrix, returning one float32 value per
row, within relative and absolute tolerance 1e-4.

Use a Triton GPU kernel (`@triton.jit` with `tl.load` / `tl.store`). A plain
Python loop, a list comprehension, or a torch/numpy implementation is NOT
acceptable.

Reply with the complete contents of `candidate.py` inside a single fenced code
block. Do not use any tools; write the code in your reply.
"""


def api_key() -> str:
    path = Path(os.path.expanduser("~/.local/share/opencode/auth.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["deepseek"]["key"]


def call(model: str, prompt: str, no_proxy: bool, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": 20,
    }).encode()
    request = urllib.request.Request(
        API, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key()}"},
    )
    if no_proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        opener = urllib.request.build_opener()
    with opener.open(request, timeout=300) as response:
        return json.loads(response.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="reasoning models need headroom before content appears; "
                         "4000 was consumed entirely by reasoning and left content empty")
    ap.add_argument("--no-proxy", action="store_true",
                    help="bypass the system proxy (try this first)")
    ap.add_argument("--out", type=Path,
                    default=Path("lowprob-attribution/results/plaintext-check.json"))
    args = ap.parse_args()

    print(f"[probe] model={args.model} no_proxy={args.no_proxy} max_tokens={args.max_tokens}")
    try:
        data = call(args.model, PROMPT, args.no_proxy, args.max_tokens)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        print(f"[probe] HTTP {exc.code}: {detail}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] {type(exc).__name__}: {exc}")
        return 1

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    logprobs = choice.get("logprobs") or {}
    content_lp = logprobs.get("content") or []
    reasoning_lp = logprobs.get("reasoning_content") or []

    print(f"[probe] finish_reason      : {choice.get('finish_reason')}")
    print(f"[probe] content chars      : {len(content)}")
    print(f"[probe] reasoning chars    : {len(reasoning)}")
    print(f"[probe] logprobs keys      : {sorted(logprobs.keys())}")
    print(f"[probe] content tokens     : {len(content_lp)}")
    print(f"[probe] reasoning tokens   : {len(reasoning_lp)}")

    # does the logged text reproduce the code, and is the code inside it?
    logged = "".join(t.get("token", "") for t in content_lp)
    fenced = re.findall(r"```(?:python)?\n([\s\S]*?)```", content)
    code = fenced[0] if fenced else ""

    print()
    print(f"[probe] code block found   : {bool(code)} ({len(code)} chars)")
    print(f"[probe] has @triton.jit    : {'@triton.jit' in code}")
    print(f"[probe] has tl.load        : {'tl.load' in code}")
    print(f"[probe] logged==content    : {logged == content}")

    # the decisive test: every code token carries a logprob
    code_tokens = 0
    code_with_lp = 0
    cursor = 0
    for entry in content_lp:
        token = entry.get("token", "")
        index = content.find(token, cursor)
        cursor = index + len(token) if index >= 0 else cursor
        if code and index >= 0:
            start = content.find(code)
            if start >= 0 and start <= index < start + len(code):
                code_tokens += 1
                if entry.get("logprob") is not None:
                    code_with_lp += 1

    print()
    print(f"[probe] tokens inside code : {code_tokens}")
    print(f"[probe] of those, with lp  : {code_with_lp}")
    verdict = code_tokens > 0 and code_tokens == code_with_lp
    print(f"[probe] VERDICT            : {'VIABLE' if verdict else 'NOT VIABLE'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "no_proxy": args.no_proxy,
        "finish_reason": choice.get("finish_reason"),
        "content_chars": len(content), "reasoning_chars": len(reasoning),
        "content_tokens": len(content_lp), "reasoning_tokens": len(reasoning_lp),
        "code_chars": len(code), "code_tokens": code_tokens,
        "code_tokens_with_logprob": code_with_lp, "viable": verdict,
        "has_triton": "@triton.jit" in code,
        "content": content, "reasoning": reasoning,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[probe] wrote {args.out}")
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())
