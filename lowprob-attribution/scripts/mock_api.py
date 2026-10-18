"""Mock upstream API server for testing the record proxy + OpenCode pipeline
without needing DeepSeek API online.

Returns a canned response with logprobs for /chat/completions,
and echoes /models. Simulates both streaming and non-streaming.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CANNED_CONTENT = "Here is a Triton kernel:\n```python\n@triton.jit\ndef kernel(x_ptr, n, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < n\n    x = tl.load(x_ptr + offs, mask=mask)\n    tl.store(x_ptr + offs, x * 2.0, mask=mask)\n```\nThe kernel uses BLOCK=256 and num_warps=4."

# Fake tokens for logprobs — mix of types
FAKE_TOKENS = [
    {"token": "Here", "logprob": -0.8},
    {"token": " is", "logprob": -0.3},
    {"token": " a", "logprob": -0.5},
    {"token": " Triton", "logprob": -1.2},
    {"token": " kernel", "logprob": -0.4},
    {"token": ":\n", "logprob": -0.6},
    {"token": "```", "logprob": -0.1},
    {"token": "python", "logprob": -0.2},
    {"token": "\n", "logprob": -0.05},
    {"token": "@", "logprob": -3.1},
    {"token": "triton", "logprob": -0.3},
    {"token": ".", "logprob": -0.1},
    {"token": "jit", "logprob": -0.4},
    {"token": "\n", "logprob": -0.1},
    {"token": "def", "logprob": -0.05},
    {"token": " kernel", "logprob": -0.2},
    {"token": "(", "logprob": -0.1},
    {"token": "x", "logprob": -0.3},
    {"token": "_", "logprob": -0.2},
    {"token": "ptr", "logprob": -0.5},
    {"token": ",", "logprob": -0.1},
    {"token": " n", "logprob": -0.2},
    {"token": ",", "logprob": -0.1},
    {"token": " BLOCK", "logprob": -0.3},
    {"token": ":", "logprob": -0.1},
    {"token": " tl", "logprob": -0.4},
    {"token": ".", "logprob": -0.1},
    {"token": "constexpr", "logprob": -0.3},
    {"token": ")", "logprob": -0.1},
    {"token": ":", "logprob": -0.1},
    {"token": "\n", "logprob": -0.05},
    {"token": " pid", "logprob": -0.3},
    {"token": " =", "logprob": -0.1},
    {"token": " tl", "logprob": -0.2},
    {"token": ".", "logprob": -0.1},
    {"token": "program", "logprob": -0.3},
    {"token": "_", "logprob": -0.2},
    {"token": "id", "logprob": -0.1},
    {"token": "(", "logprob": -0.1},
    {"token": "0", "logprob": -0.05},
    {"token": ")", "logprob": -0.1},
    {"token": "\n", "logprob": -0.05},
    {"token": " offs", "logprob": -0.3},
    {"token": " =", "logprob": -0.1},
    {"token": " pid", "logprob": -0.2},
    {"token": " *", "logprob": -0.1},
    {"token": " BLOCK", "logprob": -0.2},
    {"token": " +", "logprob": -0.1},
    {"token": " tl", "logprob": -0.2},
    {"token": ".", "logprob": -0.1},
    {"token": "arange", "logprob": -0.3},
    {"token": "(", "logprob": -0.1},
    {"token": "0", "logprob": -0.05},
    {"token": ",", "logprob": -0.1},
    {"token": " BLOCK", "logprob": -0.2},
    {"token": ")", "logprob": -0.1},
    {"token": "\n", "logprob": -0.05},
    {"token": " mask", "logprob": -0.3},
    {"token": " =", "logprob": -0.1},
    {"token": " offs", "logprob": -0.2},
    {"token": " <", "logprob": -0.1},
    {"token": " n", "logprob": -0.2},
    {"token": "\n", "logprob": -0.05},
    {"token": " x", "logprob": -0.3},
    {"token": " =", "logprob": -0.1},
    {"token": " tl", "logprob": -0.2},
    {"token": ".", "logprob": -0.1},
    {"token": "load", "logprob": -0.3},
    {"token": "(", "logprob": -0.1},
    {"token": "x", "logprob": -0.2},
    {"token": "_", "logprob": -0.2},
    {"token": "ptr", "logprob": -0.3},
    {"token": " +", "logprob": -0.1},
    {"token": " offs", "logprob": -0.2},
    {"token": ",", "logprob": -0.1},
    {"token": " mask", "logprob": -0.2},
    {"token": "=", "logprob": -0.1},
    {"token": "mask", "logprob": -0.2},
    {"token": ")", "logprob": -0.1},
    {"token": "\n", "logprob": -0.05},
    {"token": " tl", "logprob": -0.2},
    {"token": ".", "logprob": -0.1},
    {"token": "store", "logprob": -0.3},
    {"token": "(", "logprob": -0.1},
    {"token": "x", "logprob": -0.2},
    {"token": "_", "logprob": -0.2},
    {"token": "ptr", "logprob": -0.3},
    {"token": " +", "logprob": -0.1},
    {"token": " offs", "logprob": -0.2},
    {"token": ",", "logprob": -0.1},
    {"token": " x", "logprob": -0.2},
    {"token": " *", "logprob": -0.1},
    {"token": " 2", "logprob": -0.2},
    {"token": ".", "logprob": -0.1},
    {"token": "0", "logprob": -0.05},
    {"token": ",", "logprob": -0.1},
    {"token": " mask", "logprob": -0.2},
    {"token": "=", "logprob": -0.1},
    {"token": "mask", "logprob": -0.2},
    {"token": ")", "logprob": -0.1},
    {"token": "\n", "logprob": -0.05},
    {"token": "```", "logprob": -0.1},
    {"token": "\n", "logprob": -0.05},
    {"token": "The", "logprob": -0.3},
    {"token": " kernel", "logprob": -0.2},
    {"token": " uses", "logprob": -0.3},
    {"token": " BLOCK", "logprob": -0.2},
    {"token": "=", "logprob": -0.1},
    {"token": "256", "logprob": -0.2},
    {"token": " and", "logprob": -0.3},
    {"token": " num", "logprob": -0.3},
    {"token": "_", "logprob": -0.2},
    {"token": "warps", "logprob": -0.3},
    {"token": "=", "logprob": -0.1},
    {"token": "4", "logprob": -0.05},
    {"token": ".", "logprob": -0.05},
    # insert a few sentinels to simulate low-probability tokens
    {"token": " weird", "logprob": -9999.0},
    {"token": " path", "logprob": -9999.0},
    {"token": "/tmp", "logprob": -9999.0},
    {"token": "/cache", "logprob": -9999.0},
    {"token": " 3", "logprob": -9999.0},
    {"token": ".", "logprob": -0.05},
    {"token": "141", "logprob": -9999.0},
    {"token": "59", "logprob": -9999.0},
]


def make_top_logprobs(token: str, logprob: float) -> list[dict]:
    return [{"token": token, "logprob": logprob}] if logprob > -9998 else [
        {"token": "the", "logprob": -0.5},
        {"token": token, "logprob": -9999.0},
    ]


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        if self.path.rstrip("/").endswith("models"):
            data = json.dumps({"data": [{"id": "deepseek-chat"}]}).encode()
        else:
            data = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = self._read_body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
        is_stream = payload.get("stream", False)
        model = payload.get("model", "deepseek-chat")

        if not is_stream:
            resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": CANNED_CONTENT},
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {**t, "top_logprobs": make_top_logprobs(t["token"], t["logprob"])}
                            for t in FAKE_TOKENS
                        ],
                        "reasoning_content": None,
                    },
                }],
                "usage": {"prompt_tokens": 50, "completion_tokens": len(FAKE_TOKENS), "total_tokens": 50 + len(FAKE_TOKENS)},
            }
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # streaming: send chunks
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        for i, tok in enumerate(FAKE_TOKENS):
            chunk = {
                "id": f"chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": tok["token"]},
                    "finish_reason": None if i < len(FAKE_TOKENS) - 1 else "stop",
                    "logprobs": {
                        "content": [{**tok, "top_logprobs": make_top_logprobs(tok["token"], tok["logprob"])}],
                        "reasoning_content": None,
                    },
                }],
            }
            raw = f"data: {json.dumps(chunk)}\n\n".encode()
            self.wfile.write(b"%x\r\n" % len(raw) + raw + b"\r\n")
            self.wfile.flush()
            time.sleep(0.01)

        done = b"data: [DONE]\n\n"
        self.wfile.write(b"%x\r\n" % len(done) + done + b"\r\n")
        usage_chunk = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [],
            "usage": {"prompt_tokens": 50, "completion_tokens": len(FAKE_TOKENS), "total_tokens": 50 + len(FAKE_TOKENS)},
        }
        udata = f"data: {json.dumps(usage_chunk)}\n\n".encode()
        self.wfile.write(b"%x\r\n" % len(udata) + udata + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:8766")
    args = ap.parse_args()
    host, port = args.listen.split(":")
    server = ThreadingHTTPServer((host, int(port)), MockHandler)
    print(f"[mock-api] listening on {args.listen}")
    server.serve_forever()


if __name__ == "__main__":
    main()
