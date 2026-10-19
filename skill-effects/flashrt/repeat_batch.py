"""Plan repeated trials; execute only with --execute, never replay partial runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from threading import Event

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
MANIFEST = BASE / 'materials/repeat-batch-v1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def plan():
    arms = [(name, False, False) for name in ['task_only', 'source_core', 'original', 'formatted', 'phase4']]
    arms += [('phase4', True, False), ('phase4', False, True), ('phase4', True, True)]
    rng = random.Random(20261019)
    rows = []
    for repeat in range(3):
        tasks = ['causal_features', 'branch_features']
        if repeat % 2:
            tasks.reverse()
        for task in tasks:
            order = arms.copy()
            rng.shuffle(order)
            for condition, permissions, checker in order:
                rows.append({'id': f't{len(rows)+1:03d}', 'repeat': repeat+1, 'task': task,
                             'condition': condition, 'permissions': permissions, 'checker': checker})
    files = list(BASE.glob('*.py')) + list((BASE / 'workloads').rglob('*.py'))
    for folder in ['verbatim-v1', 'format-v1', 'phase4-draft-v1']:
        files += [p for p in (BASE / 'materials' / folder).rglob('*') if p.is_file()]
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': 20261019,
            'description': 'Eight conditions, two tasks, three repetitions; wider common budget; all failures retained',
            'output_budget': 96000, 'steps': 48, 'report_budget': 8000, 'report_requests': 6,
            'concurrency': 3, 'run_root': 'skill-effects/runs/flashrt-repeat-v1',
            'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))},
            'trials': rows}


def command(manifest, row):
    args = [sys.executable, str(BASE / 'pilot_trial.py'), '--repository',
            str(ROOT / '.cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository'),
            '--output', str(ROOT / manifest['run_root'] / row['id']), '--purpose', 'formal',
            '--condition', row['condition'], '--task', row['task']]
    for key in ['output_budget', 'steps', 'report_budget', 'report_requests']:
        args += ['--' + key.replace('_', '-'), str(manifest[key])]
    for key in ['permissions', 'checker']:
        if row[key]:
            args.append('--' + key)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not MANIFEST.exists():
        save(MANIFEST, plan())
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for path, expected in manifest['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    print(json.dumps({'trials': len(manifest['trials']), 'concurrency': manifest['concurrency'],
                      'output_budget_per_trial': manifest['output_budget'], 'execute': args.execute}), flush=True)
    if not args.execute:
        return
    output = ROOT / manifest['run_root']
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in manifest['trials']:
        folder = output / row['id']
        if folder.exists():
            if not (folder / 'result.json').exists():
                raise RuntimeError('Partial trial requires inspection; will not replay: ' + row['id'])
        else:
            pending.append(row)

    stop = Event()

    def run(row):
        if stop.is_set():
            return {'id': row['id'], 'not_started': True, 'reason': 'API/controller error in another trial'}
        with (output / (row['id'] + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command(manifest, row), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result_path = output / row['id'] / 'result.json'
        state = json.loads(result_path.read_text(encoding='utf-8')).get('state') if result_path.exists() else None
        if state in [None, 'api_or_trace_error']:
            stop.set()
        return {'id': row['id'], 'exitcode': result.returncode, 'state': state,
                'terminal_result_available': result_path.exists()}

    with ThreadPoolExecutor(max_workers=manifest['concurrency']) as pool:
        for future in as_completed([pool.submit(run, row) for row in pending]):
            result = future.result()
            if not result.get('not_started'):
                save(output / (result['id'] + '-controller.json'), result)
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
