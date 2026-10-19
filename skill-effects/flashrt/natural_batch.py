"""Prepare the approved natural-work comparison; never overlap the old batch."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
from threading import Event

from repeat_batch import BASE, ROOT, save, sha

MANIFEST = BASE / 'materials/natural-batch-v1.json'


def plan():
    arms = [(name, False, False) for name in ['task_only', 'source_core', 'original', 'formatted', 'phase4']]
    arms += [('phase4', True, False), ('phase4', False, True), ('phase4', True, True)]
    rng = random.Random(20261020)
    trials = []
    for repeat in range(3):
        for task in (['causal_features', 'branch_features'] if repeat % 2 == 0
                     else ['branch_features', 'causal_features']):
            order = arms.copy()
            rng.shuffle(order)
            for condition, permissions, checker in order:
                trials.append({'id': f'n{len(trials)+1:03d}', 'repeat': repeat+1, 'task': task,
                               'condition': condition, 'permissions': permissions, 'checker': checker})
    names = ['natural_batch.py', 'natural_trial.py', 'pilot_trial.py', 'benchmark_local.py',
             'run_local_reference.py', 'local_session.py', '../runtime.py', '../gpu_lock.py']
    files = [(BASE / name).resolve() for name in names] + list((BASE / 'workloads').rglob('*.py'))
    for folder in ['verbatim-v1', 'format-v1', 'phase4-draft-v1']:
        files += [p for p in (BASE / 'materials' / folder).rglob('*') if p.is_file()]
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': 20261020,
            'purpose': 'natural_work', 'run_root': 'skill-effects/runs/flashrt-natural-v1',
            'external_output_ceiling': 320000, 'external_request_ceiling': 128,
            'external_limits_shown_to_actor': False, 'concurrency': 3,
            'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))},
            'trials': trials}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not MANIFEST.exists():
        save(MANIFEST, plan())
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for path, expected in manifest['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Frozen natural-work input changed: ' + path)
    print(json.dumps({'planned': len(manifest['trials']), 'execute': args.execute,
                      'external_limits_shown_to_actor': False}), flush=True)
    if not args.execute:
        return
    old = json.loads((BASE / 'materials/repeat-batch-recovery-v1.json').read_text(encoding='utf-8'))
    unfinished = [row['id'] for row in old['trials']
                  if not (ROOT / old['run_root'] / row['id'] / 'result.json').exists()]
    if unfinished:
        raise RuntimeError(f'Current batch must finish first; {len(unfinished)} slots incomplete')
    output = ROOT / manifest['run_root']
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in manifest['trials']:
        folder = output / row['id']
        if folder.exists() and not (folder / 'result.json').exists():
            raise RuntimeError('Partial trial requires inspection; will not replay: ' + row['id'])
        if not folder.exists():
            pending.append(row)
    stop = Event()

    def run(row):
        if stop.is_set():
            return {'id': row['id'], 'not_started': True}
        command = [sys.executable, str(BASE / 'natural_trial.py'), '--repository',
                   str(ROOT / '.cache/flashrt/dd9dae0072a72964a2bdc3ca3f06f68afa122a7f/repository'),
                   '--output', str(output / row['id']), '--condition', row['condition'], '--task', row['task'],
                   '--external-output-ceiling', str(manifest['external_output_ceiling']),
                   '--external-request-ceiling', str(manifest['external_request_ceiling'])]
        for flag in ['permissions', 'checker']:
            if row[flag]:
                command.append('--' + flag)
        with (output / (row['id']+'.log')).open('w', encoding='utf-8') as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        path = output / row['id'] / 'result.json'
        state = json.loads(path.read_text(encoding='utf-8')).get('state') if path.exists() else None
        if state in [None, 'api_or_trace_error']:
            stop.set()
        return {'id': row['id'], 'state': state, 'exitcode': proc.returncode,
                'terminal_result_available': path.exists()}

    with ThreadPoolExecutor(max_workers=manifest['concurrency']) as pool:
        for future in as_completed([pool.submit(run, row) for row in pending]):
            result = future.result()
            if not result.get('not_started'):
                save(output / (result['id']+'-controller.json'), result)
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
