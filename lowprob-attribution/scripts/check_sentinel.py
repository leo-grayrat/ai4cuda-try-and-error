"""Phase 0 sanity check: does the -9999 sentinel fire on real generation?

`check_logprobs.py` confirms the API returns logprobs at all, but a hello-world
reply is too predictable to contain a low-probability tail (it reported 0
sentinels). This script sends one realistic code-generation request through the
official DeepSeek API with logprobs enabled and reports the sentinel rate, so
the 0%-30% band in PLAN.md can be checked against an actual generation.

Usage:
    python lowprob-attribution/scripts/check_sentinel.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

AUTH_PATH = os.path.expanduser(r"~\.local\share\opencode\auth.json")
URL = "https://api.deepseek.com/chat/completions"
SENTINEL = -9998.0  # -9999.0 is the documented value; compare with slack

PROMPT = (
    "Write a Triton kernel that computes a row-wise sum reduction over a "
    "2D float32 tensor of shape (4096, 4096), returning shape (4096,). "
    "Define BLOCK_SIZE as a constexpr, use tl.load/tl.store, and show the "
    "launch wrapper. Output only code, no explanation."
)

# A context shaped like a real agent turn: previous attempt + compiler output.
PROMPT_AFTER_ERROR = """The Triton kernel below fails to compile. Fix it and output the
corrected kernel only.

```python
import triton
import triton.language as tl

@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + cols, mask=cols < N, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + row, s)
```

Compile error:
```
triton.compiler.errors.CompilationError: at 8:8:
    s = tl.sum(x, axis=0)
                   ^
ValueError: `axis` must be a constexpr, got `0`
```
"""

# The same task in CUDA C++, which is rarer in pretraining than Triton/python.
PROMPT_CUDA_CPP = (
    "Write a CUDA C++ kernel that computes a row-wise sum reduction over a "
    "float32 matrix of shape (4096, 4096) into a (4096,) output. Use shared "
    "memory or warp shuffles for the reduction. Output only the kernel and "
    "launch code, no explanation."
)

SCENARIOS = {
    "cold": PROMPT,
    "after_error": PROMPT_AFTER_ERROR,
    "cuda_cpp": PROMPT_CUDA_CPP,
}

# Coarse bucketing, only for eyeballing where the tail lands.
KEYWORDS_TRITON = (
    "tl.", "triton", "num_warps", "BLOCK_SIZE", "num_stages", "constexpr",
    "@triton.jit", "program_id", "grid",
)


def bucket(token: str) -> str:
    t = token.strip()
    if not t:
        return "whitespace"
    if re.fullmatch(r"[0-9._]+", t):
        return "number"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
        low = t.lower()
        if any(k.lower() in low for k in KEYWORDS_TRITON):
            return "triton_specific"
        return "identifier"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
        return "identifier"
    if t in ("(", ")", "[", "]", "{", "}", ",", ":", ".", "=", "+", "-", "*", "/"):
        return "punctuation"
    if re.search(r"[A-Za-z]", t):
        return "NL_or_mixed"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--temperature", type=float, default=None,
                    help="omit to use the API default; 0.0 changes logprobs behaviour")
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="cold")
    ap.add_argument("--model", default="deepseek-chat")
    args = ap.parse_args()

    try:
        key = json.load(open(AUTH_PATH, encoding="utf-8"))["deepseek"]["key"]
    except Exception as exc:  # noqa: BLE001
        print(f"cannot read key from {AUTH_PATH}: {exc}")
        return 1

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": SCENARIOS[args.scenario]}],
        "max_tokens": args.max_tokens,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
        "stream": False,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    body = json.dumps(payload).encode()

    req = urllib.request.Request(
        URL, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    print(f"[request] scenario={args.scenario} temperature={args.temperature} "
          f"top_logprobs={args.top_logprobs}")
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=180).read())
    except Exception as exc:  # noqa: BLE001
        print(f"REQUEST FAILED: {exc}")
        if hasattr(exc, "read"):
            try:
                print(exc.read().decode()[:400])
            except Exception:
                pass
        return 1

    choice = data["choices"][0]
    lp = choice.get("logprobs") or {}
    msg = choice.get("message") or {}
    # Reasoning models (deepseek-flash / deepseek-v4-pro) expose logprobs on
    # reasoning_content; the chat alias exposes them on content.
    toks = lp.get("content") or []
    field_used = "content"
    if not toks:
        toks = lp.get("reasoning_content") or []
        field_used = "reasoning_content"
    text = msg.get("content") or msg.get("reasoning_content") or ""

    if not toks:
        print(f"no logprobs in either field; logprobs keys={list(lp.keys())} "
              f"message keys={list(msg.keys())}")
        return 1

    sentinels = [t for t in toks if t["logprob"] <= SENTINEL]
    low = [t for t in toks if t["logprob"] <= -5.0]
    rate = len(sentinels) / len(toks)

    print(f"logprobs field   : {field_used}")

    print(f"reply chars      : {len(text)}")
    print(f"tokens w/ logprob: {len(toks)}")
    print(f"sentinel (-9999) : {len(sentinels)}  ({rate:.1%})")
    print(f"logprob <= -5    : {len(low)}  ({len(low)/len(toks):.1%})")
    print(f"min logprob      : {min(t['logprob'] for t in toks):.3f}")

    per_bucket: dict[str, list[int]] = {}
    for t in toks:
        b = bucket(t["token"])
        row = per_bucket.setdefault(b, [0, 0])
        row[0] += 1
        if t["logprob"] <= SENTINEL:
            row[1] += 1
    print("\nsentinel rate by coarse bucket:")
    for b, (n, s) in sorted(per_bucket.items(), key=lambda kv: -kv[1][0]):
        print(f"  {b:16s} n={n:4d} sentinel={s:3d} ({s/n:.1%})")

    print("\nsample of the least likely tokens:")
    for t in sorted(toks, key=lambda x: x["logprob"])[:12]:
        top = (t.get("top_logprobs") or [{}])[0]
        print(f"  {t['token']!r:24s} logprob={t['logprob']:9.3f} "
              f"top={top.get('token')!r}@{top.get('logprob', float('nan')):.3f}")

    ok = 0.0 <= rate <= 0.30
    print(f"\nsanity band 0%-30%: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
