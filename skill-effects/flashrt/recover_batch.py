"""Retain externally interrupted trials, then run unopened and appended slots.

Run only after confirming the original batch and children are no longer live.
Candidate code, budgets and original source fingerprints stay unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from threading import Event

from repeat_batch import BASE, ROOT, MANIFEST, command, save, sha

sys.path.insert(0, str(BASE.parent))
from runtime import summarize_usage

RECOVERY = BASE / 'materials/repeat-batch-recovery-v1.json'


def prepare():
    if RECOVERY.exists():
        return json.loads(RECOVERY.read_text(encoding='utf-8'))
    original = json.loads(MANIFEST.read_text(encoding='utf-8'))
    trials = list(original['trials'])
    interruptions = []
    for trial in original['trials']:
        run = ROOT / original['run_root'] / trial['id']
        if not run.exists() or (run / 'result.json').exists():
            continue
        config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        responses = []
        for path in sorted(run.glob('response-*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            if 'usage' in data:
                responses.append(data['usage'])
        protected = json.loads((run / 'protected-inventory.json').read_text(encoding='utf-8'))
        changed = [name for name, expected in protected.items()
                   if not (run / 'work' / name).is_file() or sha(run / 'work' / name) != expected]
        backend = f"flashrt/apps/{trial['task']}/backend/"
        added = [p.relative_to(run / 'work').as_posix() for p in (run / 'work').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts
                 and not p.relative_to(run / 'work').as_posix().startswith(backend)
                 and p.relative_to(run / 'work').as_posix() not in protected]
        last = datetime.fromisoformat(events[-1]['utc'])
        started = datetime.fromisoformat(config['started_utc'])
        attempted = len(list(run.glob('request-*.json')))
        save(run / 'result.json', {'state': 'infrastructure_interrupted',
             'interruption_evidence': 'Original exec handle missing; host-level read-only process check found no matching Python process',
             'detected_utc': datetime.now(timezone.utc).isoformat(), 'last_event_utc': events[-1]['utc'],
             'seconds': (last-started).total_seconds(), 'wall_seconds_is_lower_bound': True,
             'requests': len(responses), 'requests_attempted': attempted,
             'usage_records_complete': len(responses) == attempted, 'usage': summarize_usage(responses),
             'report_available': (run / 'report.md').exists(), 'protected_files_changed': changed,
             'files_added_outside_backend': added, 'final_evaluation': {},
             'model_failure': False, 'original_trace_retained': True})
        added_trial = {**trial, 'id': f't{len(trials)+1:03d}', 'supplement_for': trial['id'],
                       'supplement_reason': 'External process interruption; original remains in cost ledger'}
        trials.append(added_trial)
        interruptions.append({'original': trial['id'], 'supplement': added_trial['id']})
    recovery = {**original, 'original_manifest_sha256': sha(MANIFEST), 'trials': trials,
                'recovery_created_utc': datetime.now(timezone.utc).isoformat(),
                'interrupted_and_supplementary_slots': interruptions,
                'recovery_policy': 'Only external interruptions receive appended slots; ordinary model failures are not replaced'}
    save(RECOVERY, recovery)
    return recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    manifest = prepare()
    for path, expected in manifest['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    output = ROOT / manifest['run_root']
    pending = [row for row in manifest['trials'] if not (output / row['id']).exists()]
    print(json.dumps({'pending': len(pending), 'interrupted': manifest['interrupted_and_supplementary_slots'],
                      'execute': args.execute}), flush=True)
    if not args.execute:
        return
    stop = Event()

    def run(row):
        if stop.is_set():
            return {'id': row['id'], 'not_started': True}
        with (output / (row['id']+'.log')).open('w', encoding='utf-8') as log:
            proc = subprocess.run(command(manifest, row), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result_path = output / row['id'] / 'result.json'
        state = json.loads(result_path.read_text(encoding='utf-8')).get('state') if result_path.exists() else None
        if state in [None, 'api_or_trace_error']:
            stop.set()
        return {'id': row['id'], 'exitcode': proc.returncode, 'state': state,
                'terminal_result_available': result_path.exists()}

    with ThreadPoolExecutor(max_workers=manifest['concurrency']) as pool:
        for future in as_completed([pool.submit(run, row) for row in pending]):
            result = future.result()
            if not result.get('not_started'):
                save(output / (result['id']+'-controller.json'), result)
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
