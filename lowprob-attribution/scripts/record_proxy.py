"""Transparent recording proxy for OpenAI-compatible APIs.

Sits between OpenCode and the upstream chat-completions endpoint.
Injects `logprobs: true, top_logprobs: N` into every request and
side-records the full response (content + reasoning_content logprobs,
aggregated across SSE stream chunks) to a JSONL file.

Usage:
    python record_proxy.py --upstream https://api.deepseek.com/v1 \
        --listen 127.0.0.1:8765 --out lowprob-attribution/proxy-records \
        --top-logprobs 20

    # then point OpenCode's provider baseURL at http://127.0.0.1:8765/v1

Records one JSON line per assistant response:
    {ts, request_id, model, request_messages, content_text,
     reasoning_text, content_logprobs, reasoning_logprobs, finish_reason}
Requests are echoed with their own id so the caller can correlate.
The side-record is the alignment key source for the token attribution
pipeline: content_text / reasoning_text is matched against OpenCode
events.jsonl text and tool_use parts.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.request
import urllib.error

RECORD_FIELDS = (
    "ts",
    "request_id",
    "model",
    "content_text",
    "reasoning_text",
    "content_logprobs",
    "reasoning_logprobs",
    "finish_reason",
    "usage",
    "temperature",
)


class ProxyState:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = out_dir / "logprobs.jsonl"
        self.n_requests = 0
        self.n_recorded = 0

    def record(self, entry: dict) -> None:
        slim = {k: entry.get(k) for k in RECORD_FIELDS if k in entry}
        with self.records_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
        self.n_recorded += 1


def build_handler(state: ProxyState, upstream_base: str, top_logprobs: int,
                  temperature_override: float | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        # ---- helpers ----------------------------------------------------

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def _forward(self, body: bytes, is_stream: bool):
            url = upstream_base.rstrip("/") + self.path
            req = urllib.request.Request(url, data=body, method=self.command)
            for h in ("Authorization", "Content-Type", "Accept", "User-Agent"):
                if self.headers.get(h):
                    req.add_header(h, self.headers[h])
            req.add_header("Connection", "close")
            try:
                resp = urllib.request.urlopen(req, timeout=600)
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                self.send_response(exc.code)
                self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return None
            return resp

        # ---- recording --------------------------------------------------

        def _record_nonstream(self, data: dict, request_id: str, temperature=None):
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            lp = choice.get("logprobs") or {}
            state.record({
                "ts": time.time(),
                "request_id": request_id,
                "model": data.get("model"),
                "content_text": msg.get("content"),
                "reasoning_text": msg.get("reasoning_content"),
                "content_logprobs": lp.get("content") or [],
                "reasoning_logprobs": lp.get("reasoning_content") or [],
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage"),
                "temperature": temperature,
            })

        def _handle_chat(self, body: bytes):
            request_id = str(uuid.uuid4())
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = None
            is_stream = bool(payload and payload.get("stream"))
            sent_temperature = None
            if payload is not None:
                payload["logprobs"] = True
                payload["top_logprobs"] = top_logprobs
                # DeepSeek degenerates logprobs to all-zero when temperature is
                # 0, so the probe is only meaningful off that point. Record what
                # is actually sent, and lift it only if explicitly asked to.
                if temperature_override is not None and payload.get("temperature") in (None, 0):
                    payload["temperature"] = temperature_override
                sent_temperature = payload.get("temperature")
                body = json.dumps(payload).encode("utf-8")

            resp = self._forward(body, is_stream)
            if resp is None:
                return

            if not is_stream:
                data = json.loads(resp.read())
                out = json.dumps(data).encode("utf-8")
                self.send_response(resp.status if hasattr(resp, "status") else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                self._record_nonstream(data, request_id, sent_temperature)
                return

            # ---- stream: aggregate SSE chunks, replay to caller -----------
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            content_text = []
            reasoning_text = []
            content_lp = []
            reasoning_lp = []
            finish_reason = None
            usage = None
            model = None
            got_done = False
            try:
                for raw in resp:
                    # Forward every line verbatim, blank separator lines
                    # included: the caller's SSE parser needs the blank line to
                    # terminate an event, and dropping it (as an earlier version
                    # did) makes OpenCode fail with a JSON parsing error.
                    self._write_chunk(raw)
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        got_done = True
                        continue
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    model = chunk.get("model") or model
                    for choice in chunk.get("choices") or []:
                        lp = choice.get("logprobs") or {}
                        if lp.get("content") is not None:
                            content_lp.extend(lp["content"])
                        if lp.get("reasoning_content") is not None:
                            reasoning_lp.extend(lp["reasoning_content"])
                        delta = choice.get("delta") or {}
                        if delta.get("content") is not None:
                            content_text.append(delta["content"])
                        if delta.get("reasoning_content") is not None:
                            reasoning_text.append(delta["reasoning_content"])
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                    if chunk.get("usage"):
                        usage = chunk["usage"]
            finally:
                if not got_done:
                    self._write_chunk(b"data: [DONE]\n\n")
                self._write_final_chunk()
                state.record({
                    "ts": time.time(),
                    "request_id": request_id,
                    "model": model,
                    "content_text": "".join(content_text) or None,
                    "reasoning_text": "".join(reasoning_text) or None,
                    "content_logprobs": content_lp,
                    "reasoning_logprobs": reasoning_lp,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "temperature": sent_temperature,
                })

        def _write_chunk(self, raw: bytes):
            self.wfile.write(b"%x\r\n" % len(raw) + raw + b"\r\n")
            self.wfile.flush()

        def _write_final_chunk(self):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        # ---- dispatch ---------------------------------------------------

        def do_POST(self):
            body = self._read_body()
            state.n_requests += 1
            if self.path.rstrip("/").endswith("chat/completions"):
                self._handle_chat(body)
            else:
                resp = self._forward(body, False)
                if resp is None:
                    return
                data = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def do_GET(self):
            resp = self._forward(b"", False)
            if resp is None:
                return
            data = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", required=True, help="e.g. https://api.deepseek.com/v1")
    ap.add_argument("--listen", default="127.0.0.1:8765")
    ap.add_argument("--out", default="lowprob-attribution/proxy-records")
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--temperature-override", type=float, default=None,
                    help="value to use when the request carries temperature 0 or none; "
                         "DeepSeek degenerates logprobs to all-zero at temperature 0")
    args = ap.parse_args()

    state = ProxyState(Path(args.out))
    handler = build_handler(state, args.upstream, args.top_logprobs,
                            args.temperature_override)
    host, port = args.listen.split(":")
    server = ThreadingHTTPServer((host, int(port)), handler)
    print(f"[record-proxy] listening on {args.listen} -> {args.upstream}")
    print(f"[record-proxy] recording to {state.records_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[record-proxy] stopped; {state.n_requests} requests, {state.n_recorded} recorded")


if __name__ == "__main__":
    main()
