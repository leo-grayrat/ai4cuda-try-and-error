"""Does the API return logprobs for tool-call arguments?

The first trial recorded 349 visible output tokens with zero content_logprobs,
which was read as "tool-call tokens carry no probability". That inference came
from token counts, not from a direct look at a tool-calling response. This sends
a request that must trigger a tool call and prints the raw structure, so the
question is settled by observation rather than by arithmetic.

Usage:
    python lowprob-attribution/scripts/check_tool_logprobs.py
"""
from __future__ import annotations

import json
import os
import urllib.request

AUTH_PATH = os.path.expanduser(r"~\.local\share\opencode\auth.json")
URL = "https://api.deepseek.com/chat/completions"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write text content to a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
        },
    },
}]

PROMPT = (
    "Use the write_file tool to create a Triton kernel file. "
    "First write out the complete file content in your reply, verbatim, "
    "then call the tool with that same content. "
    "The kernel should compute a row-wise sum over a (4096, 4096) float32 "
    "matrix into a (4096,) output."
)


def main() -> int:
    key = json.load(open(AUTH_PATH, encoding="utf-8"))["deepseek"]["key"]
    body = json.dumps({
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS,
        "max_tokens": 4000,
        "logprobs": True,
        "top_logprobs": 5,
        "stream": False,
    }).encode()

    request = urllib.request.Request(
        URL, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    data = json.loads(urllib.request.urlopen(request, timeout=240).read())

    choice = data["choices"][0]
    message = choice.get("message") or {}
    logprobs = choice.get("logprobs") or {}

    print("finish_reason     :", choice.get("finish_reason"))
    print("message keys      :", sorted(message.keys()))
    print("logprobs keys     :", sorted(logprobs.keys()))

    calls = message.get("tool_calls") or []
    print(f"tool_calls        : {len(calls)}")
    for call in calls:
        fn = call.get("function") or {}
        args = fn.get("arguments") or ""
        print(f"  function={fn.get('name')!r} arguments_len={len(args)}")

    content_tokens = logprobs.get("content") or []
    reasoning_tokens = logprobs.get("reasoning_content") or []
    print(f"logprobs.content        : {len(content_tokens)} tokens")
    print(f"logprobs.reasoning      : {len(reasoning_tokens)} tokens")

    usage = data.get("usage") or {}
    print("usage             :", {k: v for k, v in usage.items() if "token" in k})

    if content_tokens:
        joined = "".join(t.get("token", "") for t in content_tokens)
        print(f"\nlogprobs.content, joined length {len(joined)} chars:")
        print(repr(joined[:400]))
        code_markers = ("tl.load", "tl.store", "triton.jit", "num_warps", "BLOCK_SIZE")
        print(f"code markers present in content logprobs: "
              f"{[m for m in code_markers if m in joined]}")
        in_args = None
        for call in calls:
            args = (call.get("function") or {}).get("arguments") or ""
            if args and args[:60] in joined:
                in_args = True
        print(f"arguments text appears inside logprobs.content: {in_args}")
    else:
        print("\nlogprobs.content is EMPTY while tool_calls is non-empty")

    if reasoning_tokens:
        print(f"reasoning sample: {reasoning_tokens[0].get('token')!r} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
