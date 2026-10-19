"""Index the human source ledger; verify ranges and pin exact source excerpts."""
import hashlib
import json
from pathlib import Path
import re


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    base = Path(__file__).resolve().parent
    source = base / 'materials/verbatim-v1'
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    docs = {row['logical_path']: row for row in manifest['documents']}
    ledger = (base / 'REQUIREMENTS.zh.md').read_text(encoding='utf-8')
    logical_paths = {'R': 'AGENTS.md', 'A': 'flashrt/apps/AGENTS.md',
                     'I': 'flashrt/ir/AGENTS.md', 'S': 'assets/AGENTS.md'}
    rows = []
    for line in ledger.splitlines():
        columns = [column.strip() for column in line.strip('|').split('|')]
        if len(columns) != 4 or not re.fullmatch(r'[RAIS]\d{2}', columns[0]):
            continue
        identifier, location, requirement, evaluation = columns
        logical_path = logical_paths[identifier[0]]
        bounds = re.search(r'(\d+)—(\d+)', location)
        assert bounds, identifier
        start, end = map(int, bounds.groups())
        doc = docs[logical_path]
        raw = (source / doc['original']).read_bytes()
        assert sha(raw) == doc['source_sha256']
        lines = raw.splitlines(keepends=True)
        assert 1 <= start <= end <= len(lines), identifier
        excerpt = b''.join(lines[start - 1:end])
        anchor = next(text.strip() for text in excerpt.decode('utf-8').splitlines() if text.strip())
        rows.append({'id': identifier, 'logical_path': logical_path, 'source_start': start,
                     'source_end': end, 'source_sha256': sha(raw), 'excerpt_sha256': sha(excerpt),
                     'first_nonempty_line': anchor, 'requirement_zh': requirement,
                     'evaluation_proposal_zh': evaluation})
    expected = re.findall(r'^\| ([RAIS]\d{2}) \|', ledger, flags=re.M)
    assert len(rows) == len(expected) == len({row['id'] for row in rows})
    assert set(expected) == {row['id'] for row in rows}
    result = {'status': 'Human-authored source ledger, not an atomic rule count or automated semantic proof',
              'ledger_sha256': sha((base / 'REQUIREMENTS.zh.md').read_bytes()),
              'source_ranges_verified': True, 'entries': rows}
    path = base / 'materials/requirements-v1.json'
    path.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    print(json.dumps({'entries': len(rows), 'source_documents': len(logical_paths),
                      'range_and_source_hash_checks': 'passed'}))


if __name__ == '__main__':
    main()
