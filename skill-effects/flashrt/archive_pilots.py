"""Archive complete pilot traces and controller evidence, excluding source clones."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for run in args.runs:
            assert (run / 'result.json').exists(), 'Archive terminal pilots only'
            selected = [path for path in run.iterdir() if path.is_file()]
            for subtree in ['control', 'work/flashrt/apps/causal_features/backend']:
                selected += [path for path in (run / subtree).rglob('*')
                             if path.is_file() and '__pycache__' not in path.parts]
            for path in sorted(set(selected)):
                name = (Path(run.name) / path.relative_to(run)).as_posix()
                raw = path.read_bytes()
                archive.writestr(name, raw)
                rows.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for row in rows:
            assert hashlib.sha256(archive.read(row['path'])).hexdigest() == row['sha256']
    manifest = {'archive': args.output.name, 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'bytes': args.output.stat().st_size, 'files': rows,
                'excluded': 'Public pinned source clones; reconstruct with the existing repository provenance and instruction manifests',
                'verified_every_entry': True}
    args.output.with_suffix('.manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'files': len(rows), 'archive_bytes': args.output.stat().st_size, 'all_entries_verified': True}))


if __name__ == '__main__':
    main()
