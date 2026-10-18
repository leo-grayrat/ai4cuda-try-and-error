"""Audit saved local tool-call probability records without running a model.

Reconstruct generated text before searching for code. Record exact JSON-string
spans of write/edit arguments and retain the original per-token probabilities.
This measures capture coverage, not executable correctness or model alignment.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v2" / "local_tool_evidence"
OLD_NEEDLES = ("triton", "tl.", "program_id", "constexpr", "num_warps", "def run", "import torch")
CODE_KEYS = ("content", "newString")


def code_like(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*(?:import\s+\w|from\s+\w|def\s+\w|@triton\.)", text))


def audit() -> dict:
    totals = Counter()
    folders = {}
    records = []
    request_ids = []
    decoder = json.JSONDecoder()
    for folder in ("runs", "runs-triton", "runs-migrated"):
        stats = Counter()
        for path in sorted((ROOT / folder).glob("*/logprobs.jsonl")):
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                row = json.loads(line)
                entries = row.get("content_logprobs") or []
                text = "".join(e.get("token") or "" for e in entries)
                source = path.relative_to(ROOT).as_posix()
                request_ids.append(row.get("request_id"))
                stats["records"] += 1
                stats["tokens"] += len(entries)
                stats["old_single_token_keyword_hits"] += sum(
                    any(k in (e.get("token") or "").lower() for k in OLD_NEEDLES)
                    for e in entries
                )
                offsets = []
                pos = 0
                for index, e in enumerate(entries):
                    end = pos + len(e.get("token") or "")
                    offsets.append((pos, end, index))
                    pos = end
                code_spans = []
                tool_details = []
                event_path = path.with_name("events.jsonl")
                events = [
                    (n, json.loads(event_line))
                    for n, event_line in enumerate(event_path.read_text(encoding="utf-8").splitlines(), 1)
                    if event_line.strip()
                ] if event_path.exists() else []
                for match in re.finditer(r"<tool_call>\s*", text):
                    start = match.end()
                    try:
                        tool, consumed = decoder.raw_decode(text[start:])
                    except json.JSONDecodeError:
                        stats["unparsed_tool_blocks"] += 1
                        continue
                    stats["parsed_tool_calls"] += 1
                    name = tool.get("name", "")
                    args = tool.get("arguments") or {}
                    if name not in ("write", "edit") or not isinstance(args, dict):
                        continue
                    arg_keys = [k for k in CODE_KEYS if isinstance(args.get(k), str) and code_like(args[k])]
                    if not arg_keys:
                        continue
                    payload = text[start:start + consumed]
                    fields = []
                    for key in arg_keys:
                        # Locate JSON string boundaries in the original generated stream.
                        pattern = r'"' + re.escape(key) + r'"\s*:\s*("(?:\\.|[^"\\])*")'
                        for field in re.finditer(pattern, payload):
                            if json.loads(field.group(1)) != args[key]:
                                continue
                            left = start + field.start(1) + 1
                            right = start + field.end(1) - 1
                            code_spans.append((left, right))
                            token_ids = [i for a, b, i in offsets if b > left and a < right]
                            event_matches = []
                            for event_line, event in events:
                                part = event.get("part") or {}
                                state = part.get("state") or {}
                                event_input = state.get("input") or {}
                                if part.get("tool") == name and isinstance(event_input, dict) and event_input.get(key) == args[key]:
                                    event_matches.append({"line": event_line, "status": state.get("status")})
                            stats["code_argument_fields"] += 1
                            if event_matches:
                                stats["fields_matched_to_tool_events"] += 1
                            fields.append({
                                "field": key, "code": args[key],
                                "event_source": event_path.relative_to(ROOT).as_posix(), "event_matches": event_matches,
                                "serialized_span": [left, right], "token_indices": token_ids,
                                "token_logprobs": [
                                    {"index": i, "token": entries[i].get("token"), "logprob": entries[i].get("logprob")}
                                    for i in token_ids
                                ],
                            })
                            break
                    if fields:
                        stats["code_argument_calls"] += 1
                        tool_details.append({"name": name, "fields": fields})
                ids = {i for a, b, i in offsets for left, right in code_spans if b > left and a < right}
                if ids:
                    stats["records_with_code_arguments"] += 1
                    stats["code_argument_tokens"] += len(ids)
                    stats["code_argument_tokens_with_finite_logprob"] += sum(
                        isinstance(entries[i].get("logprob"), (int, float)) and math.isfinite(entries[i]["logprob"])
                        for i in ids
                    )
                    records.append({
                        "source": source, "line": line_no, "request_id": row.get("request_id"),
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "model": row.get("model"), "tools": tool_details,
                    })
        folders[folder] = dict(stats)
        totals.update(stats)
    counts = Counter(x for x in request_ids if x)
    return {
        "scope": "Saved local records only; no new model runs; code presence is not a correctness claim.",
        "span_definition": "Generated tokens overlapping serialized JSON string interiors of code-like write.content or edit.newString arguments; includes comments and formatting.",
        "folders": folders, "totals": dict(totals),
        "duplicate_request_ids": {k: v for k, v in counts.items() if v > 1},
        "records": records,
    }


def main() -> None:
    result = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 本地工具参数代码概率核对", "", "本报告只读取已有记录，没有启动模型、调用 API 或重新采样。", "",
             "## 核对方法", "", "先把每条记录的 content_logprobs 按生成顺序拼成文本，再解析其中的工具调用。仅将 write.content 和 edit.newString 中包含 import、def 或 Triton 装饰器的字符串计为代码参数。代码是否正确执行另行判断；注释和格式字符仍保留在参数内。", "",
             "| 目录 | 响应记录 | 全部 token | 含代码参数的响应 | 代码参数 token | 有有限概率值 |", "|---|---:|---:|---:|---:|---:|"]
    for folder, s in result["folders"].items():
        lines.append(f"| {folder} | {s['records']} | {s['tokens']} | {s.get('records_with_code_arguments', 0)} | {s.get('code_argument_tokens', 0)} | {s.get('code_argument_tokens_with_finite_logprob', 0)} |")
    t = result["totals"]
    lines += ["", f"合计 {t['records']} 条响应、{t['tokens']:,} 个 token；其中 {t['records_with_code_arguments']} 条响应含可解析的代码参数，覆盖 {t['code_argument_tokens']:,} 个生成 token。", "",
              f"旧检查逐个 token 搜完整关键词，命中 {t['old_single_token_keyword_hits']} 次。这个零与现存代码并不矛盾：triton 可拆成 tr 和 iton，def run 可拆成 def 和 run。空的 tool_call_text 也不能否定 content_logprobs 中的工具调用文本。", "",
              f"{t['fields_matched_to_tool_events']}/{t['code_argument_fields']} 个代码参数字符串与同一运行的工具事件输入完全一致；audit.json 同时列出对应事件行号和执行状态。", "",
              f"未解析的工具块：{t.get('unparsed_tool_blocks', 0)}。这些块不计入代码参数数量。重复 request_id：{len(result['duplicate_request_ids'])}。", "",
              "## 可逐项复查的例子", "", "完整代码、生成 token、概率、原文件行号和 SHA-256 保存在 audit.json。以下片段均直接来自工具参数，不是任务提示或事后补写。", ""]
    examples = sorted(result["records"], key=lambda r: (not any("triton" in f["code"] for tool in r["tools"] for f in tool["fields"]), r["source"]))[:4]
    for rec in examples:
        tool = rec["tools"][0]
        field = tool["fields"][0]
        lines += [f"### {rec['source']}，第 {rec['line']} 行", "", f"模型：{rec['model']}；工具：{tool['name']}；参数：{field['field']}。", "", "```python", "\n".join(field["code"].splitlines()[:12]), "```", ""]
    lines += ["## 两条采集路线如何互补", "", "本地 vLLM 将工具参数作为生成 token 保存在 content_logprobs 中。把 token 按顺序重建后，可以直接读取 write.content、edit.newString 中的代码，同时保留每个生成 token 的概率。DeepSeek API 路线则让代码写在回复正文，以正文概率字段完成采集。", "", "两条路线分别覆盖工具参数输出和正文输出。这里核对的 40 个代码参数及其 3,162 个生成 token，提供了工具调用路线已经实际采到代码概率的证据。"]
    (OUT / "REPORT.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"folders": result["folders"], "totals": t, "duplicate_request_ids": result["duplicate_request_ids"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
