"""Build a reviewable partial rewrite while preserving every other source byte."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from tokenizers import Tokenizer


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent / 'materials'
    source_dir = base / 'verbatim-v1'
    target = base / 'phase4-draft-v1'
    manifest = json.loads((source_dir / 'manifest.json').read_text(encoding='utf-8'))
    root = (source_dir / manifest['documents'][0]['original']).read_bytes()
    assert sha(root) == manifest['documents'][0]['source_sha256']
    lines = root.splitlines(keepends=True)
    assert lines[546].startswith(b'### Phase 4:')
    assert lines[1181].startswith(b'### Verification and benchmarking')
    prefix, replaced, suffix = b''.join(lines[:546]), b''.join(lines[546:1181]), b''.join(lines[1181:])
    replacement = (target / 'replacement.md').read_bytes()
    newline = b'\r\n' if b'\r\n' in root else b'\n'
    replacement = replacement.replace(b'\r\n', b'\n').rstrip(b'\n').replace(b'\n', newline) + newline * 2
    combined = prefix + replacement + suffix
    assert combined[:len(prefix)] == prefix
    assert combined[len(prefix) + len(replacement):] == suffix
    ranges = [(int(a), int(b)) for a, b in re.findall(r'^\| (\d+)—(\d+) \|',
              (target / 'AUDIT.zh.md').read_text(encoding='utf-8'), flags=re.M)]
    assert ranges[0][0] == 547 and ranges[-1][1] == 1181
    assert all(b + 1 == c for (_, b), (c, _) in zip(ranges, ranges[1:]))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    count = lambda raw: len(tokenizer.encode(raw.decode('utf-8'), add_special_tokens=False).ids)
    (target / 'root-draft.md').write_bytes(combined)
    result = {'status': 'Partial rewrite draft; not a frozen condition or demonstrated effective compression',
              'source_root_sha256': sha(root), 'source_lines_replaced': [547, 1181],
              'untouched_prefix_sha256': sha(prefix), 'untouched_suffix_sha256': sha(suffix),
              'outside_replacement_identical': True, 'source_ranges_contiguous': True,
              'tokenizer_sha256': sha(args.tokenizer.read_bytes()),
              'tokens': {'root_original': count(root), 'section_original': count(replaced),
                         'section_draft': count(replacement), 'root_draft': count(combined)},
              'root_draft_sha256': sha(combined),
              'unchanged_documents': [{key: row[key] for key in ['logical_path', 'source_sha256', 'text_tokens']}
                                      for row in manifest['documents'][1:]],
              'semantic_review': 'Author source comparison only; no independent review or task-outcome validation'}
    (target / 'build.json').write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    print(json.dumps({'tokens': result['tokens'], 'outside_replacement_identical': True}))


if __name__ == '__main__':
    main()
