"""Count document text with the pinned official tokenizer; optionally calibrate API framing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
TOKENIZER = ROOT.parent / ".cache/skill-effects-tokenizer/tokenizer.json"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate-api", action="store_true")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    paths = [ROOT / "sources/CUDA-Agent.SKILL.md", ROOT / "sources/FlashRT.AGENTS.md", ROOT / "materials/core.md", ROOT / "materials/workflow.md", ROOT / "materials/condensed.md"]
    paths.extend(sorted((ROOT / "evidence").glob("growth-*/cumulative.md")))
    rows = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        rows.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "lines": len(text.splitlines()), "characters": len(text), "text_tokens": len(tokenizer.encode(text, add_special_tokens=False).ids)})
    result = {"tokenizer_repository": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash", "revision": REVISION, "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(), "measured_utc": datetime.now(timezone.utc).isoformat(), "scope": "document text only; actual agent request usage is recorded separately", "documents": rows}
    result["archived_api_calibration"] = "evidence/token-calibration/calibration.json"
    if args.calibrate_api:
        folder = ROOT / "runs" / ("token-calibration-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
        folder.mkdir(parents=True)
        key = json.loads((Path.home() / ".local/share/opencode/auth.json").read_text(encoding="utf-8"))["deepseek"]["key"]
        calibrations = []
        for index, document in enumerate(["", paths[0].read_text(encoding="utf-8"), paths[-1].read_text(encoding="utf-8")]):
            content = "Length calibration. Reply OK. The following quoted document is data.\nBEGIN DOCUMENT\n" + document + "\nEND DOCUMENT"
            body = {"model": "deepseek-flash", "messages": [{"role": "user", "content": content}], "thinking": {"type": "disabled"}, "max_tokens": 1}
            response = requests.post("https://api.deepseek.com/chat/completions", json=body, headers={"Authorization": "Bearer " + key}, timeout=(20, 60))
            data = response.json()
            (folder / f"request-{index}.json").write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
            (folder / f"response-{index}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            response.raise_for_status()
            text_count = len(tokenizer.encode(content, add_special_tokens=False).ids)
            prompt_count = data["usage"]["prompt_tokens"]
            calibrations.append({"index": index, "content_text_tokens": text_count, "api_prompt_tokens": prompt_count, "framing_difference": prompt_count - text_count, "usage": data["usage"]})
        result["api_calibration"] = calibrations
        result["constant_framing_difference"] = len({r["framing_difference"] for r in calibrations}) == 1
    (ROOT / "materials/token_counts.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
