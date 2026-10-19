"""Reflow eligible prose only; verify wording, examples and rendered structure."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from markdown_it import MarkdownIt
from tokenizers import Tokenizer


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reflow(raw):
    output, pending = [], []
    fence = None
    changed = 0

    def flush():
        nonlocal changed
        if len(pending) > 1:
            last = pending[-1]
            ending = b'\r\n' if last.endswith(b'\r\n') else b'\n' if last.endswith(b'\n') else b''
            output.append(b' '.join(line.rstrip(b'\r\n') for line in pending) + ending)
            changed += 1
        else:
            output.extend(pending)
        pending.clear()

    for line in raw.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = re.match(rb'(`{3,}|~{3,})', stripped)
        if marker:
            flush()
            symbol = marker.group(1)[:1]
            if fence is None:
                fence = symbol
            elif fence == symbol:
                fence = None
            output.append(line)
            continue
        protected = (fence is not None or not line.strip() or line[:1].isspace()
                     or b'`' in line or line.rstrip(b'\r\n').endswith((b'  ', b'\\'))
                     or re.match(rb'^(#{1,6}\s|[>*+-]\s|\d+[.)]\s|\||<)', line))
        if protected:
            flush()
            output.append(line)
        else:
            pending.append(line)
    flush()
    return b''.join(output), changed


def canonical_html(text):
    rendered = MarkdownIt('commonmark').render(text)
    # Preserve code contents exactly; normalize insignificant whitespace elsewhere.
    parts = re.split(r'(<pre\b[^>]*>.*?</pre>|<code\b[^>]*>.*?</code>)', rendered, flags=re.S)
    return ''.join(part if index % 2 else re.sub(r'\s+', ' ', part)
                   for index, part in enumerate(parts))


def code_blocks(text):
    parser = MarkdownIt('commonmark')
    lines = text.splitlines(keepends=True)
    return [''.join(lines[token.map[0]:token.map[1]])
            for token in parser.parse(text) if token.type in {'fence', 'code_block'}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent / 'materials/verbatim-v1'
    source_manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows, files = [], []
    for index, source in enumerate(source_manifest['documents']):
        raw = (base / source['original']).read_bytes()
        assert sha(raw) == source['source_sha256']
        formatted, changed = reflow(raw)
        original_text, formatted_text = raw.decode('utf-8'), formatted.decode('utf-8')
        assert re.findall(r'\S+', original_text) == re.findall(r'\S+', formatted_text)
        assert code_blocks(original_text) == code_blocks(formatted_text)
        assert canonical_html(original_text) == canonical_html(formatted_text), source['logical_path']
        filename = f'd{index:02d}.md'
        files.append((filename, formatted))
        rows.append({'logical_path': source['logical_path'], 'path': filename,
                     'source_sha256': sha(raw), 'formatted_sha256': sha(formatted),
                     'source_tokens': source['text_tokens'],
                     'formatted_tokens': len(tokenizer.encode(formatted_text, add_special_tokens=False).ids),
                     'reflowed_prose_groups': changed, 'code_blocks': len(code_blocks(original_text)),
                     'identical_nonwhitespace_sequence': True, 'identical_code_blocks': True,
                     'same_commonmark_rendering_except_noncode_whitespace': True})
    result = {'treatment': 'Prose line reflow only; original wording, order, scopes and examples retained',
              'tokenizer_sha256': sha(args.tokenizer.read_bytes()), 'documents': rows}
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, formatted in files:
        (args.output / filename).write_bytes(formatted)
    (args.output / 'manifest.json').write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'documents': len(rows), 'reflowed_groups': sum(row['reflowed_prose_groups'] for row in rows),
                      'source_tokens': sum(row['source_tokens'] for row in rows),
                      'formatted_tokens': sum(row['formatted_tokens'] for row in rows),
                      'all_invariants_verified': True}))


if __name__ == '__main__':
    main()
