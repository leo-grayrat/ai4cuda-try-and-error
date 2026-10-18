"""Phase 0 check: does the DeepSeek official API support logprobs, and what does the -9999 sentinel look like?"""
import json
import sys
import urllib.request

auth = json.load(open(r"C:\Users\34788\.local\share\opencode\auth.json"))
key = auth["deepseek"]["key"]

body = json.dumps({
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 20,
    "logprobs": True,
    "top_logprobs": 5,
    "stream": False,
}).encode()

req = urllib.request.Request(
    "https://api.deepseek.com/chat/completions",
    data=body,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)

try:
    resp = urllib.request.urlopen(req, timeout=30)
except Exception as exc:  # noqa: BLE001
    print(f"REQUEST FAILED: {exc}")
    if hasattr(exc, "read"):
        try:
            print(exc.read().decode())
        except Exception:
            pass
    sys.exit(1)

data = json.loads(resp.read())
msg = data["choices"][0]["message"]
lp = data["choices"][0].get("logprobs") or {}
content_lp = lp.get("content") or []

print(f"content: {msg.get('content')!r}")
print(f"num tokens with logprobs: {len(content_lp)}")
sentinel = sum(1 for t in content_lp if t["logprob"] <= -9998)
print(f"sentinel count: {sentinel}")
for t in content_lp[:10]:
    print(f"  token={t['token']!r} logprob={t['logprob']:.4f} top0={t['top_logprobs'][0]['token']!r}@{t['top_logprobs'][0]['logprob']:.3f}" if t.get("top_logprobs") else f"  token={t['token']!r} logprob={t['logprob']:.4f}")
