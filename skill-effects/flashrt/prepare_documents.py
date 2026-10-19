"""Freeze the real FlashRT instruction family and split it without rewriting.

The files are experimental data. Logical AGENTS.md locations and precedence
are retained in the manifest; generated files are not active repo guidance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer

DOCUMENTS = ["AGENTS.md", "flashrt/apps/AGENTS.md", "flashrt/ir/AGENTS.md", "assets/AGENTS.md"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def split_document(raw):
    lines = raw.decode("utf-8").splitlines(keepends=True)
    boundaries = [(0, "entry")]
    fence = None
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            token = marker.group(1)[0]
            if fence is None:
                fence = token
            elif fence == token:
                fence = None
            continue
        if fence is not None:
            continue
        if index and (re.match(r"#{1,3}\s", line) or line.startswith("**")):
            boundaries.append((index, line.strip()))
    result = []
    for position, (start, title) in enumerate(boundaries):
        stop = boundaries[position + 1][0] if position + 1 < len(boundaries) else len(lines)
        result.append((start + 1, stop, title, "".join(lines[start:stop]).encode("utf-8")))
    assert b"".join(part[3] for part in result) == raw
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    manifest = {"basis": "Pinned original FlashRT instructions; no new rules, no synthetic growth",
                "treatment": "Verbatim modules. Concatenating each document's modules reconstructs its original bytes. This does not yet define a selective-loading policy.",
                "precedence": "Nearest logical AGENTS.md applies, per the original root document; scopes must remain the same in every condition.",
                "tokenizer_sha256": sha(args.tokenizer.read_bytes()), "documents": []}
    for number, logical_path in enumerate(DOCUMENTS):
        raw = (args.repository / logical_path).read_bytes()
        document_id = f"d{number:02d}"
        original = args.output / "original" / (document_id + ".md")
        original.parent.mkdir(exist_ok=True)
        original.write_bytes(raw)
        modules = []
        for part, (start, stop, title, data) in enumerate(split_document(raw)):
            path = args.output / "modules" / f"{document_id}-{part:03d}.md"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
            modules.append({"path": path.relative_to(args.output).as_posix(), "source_start": start,
                            "source_end": stop, "title": title, "sha256": sha(data), "bytes": len(data),
                            "text_tokens": len(tokenizer.encode(data.decode("utf-8"), add_special_tokens=False).ids)})
        row = {"logical_path": logical_path, "original": original.relative_to(args.output).as_posix(),
               "source_sha256": sha(raw), "bytes": len(raw), "lines": len(raw.splitlines()),
               "text_tokens": len(tokenizer.encode(raw.decode("utf-8"), add_special_tokens=False).ids), "modules": modules}
        assert b"".join((args.output / item["path"]).read_bytes() for item in modules) == raw
        manifest["documents"].append(row)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"documents": [{key: row[key] for key in ["logical_path", "lines", "text_tokens"]}
                                    for row in manifest["documents"]],
                      "modules": sum(len(row["modules"]) for row in manifest["documents"]),
                      "all_original_bytes_reconstructed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
