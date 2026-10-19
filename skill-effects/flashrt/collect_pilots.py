"""Summarize saved API pilots without inventing scores for absent artifacts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(run):
    result_path, config_path = run / 'result.json', run / 'config.json'
    if not result_path.exists() or not config_path.exists():
        return {'run': str(run), 'terminal_result_available': False}
    result = json.loads(result_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    operations = Counter()
    failed_tools, writes, responses = [], [], []
    for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['type'] == 'response':
            responses.append({'index': row['index'], 'usage': row['usage'], 'finish_reason': row['finish_reason']})
        elif row['type'] == 'tool':
            try:
                arguments = json.loads(row['call']['function']['arguments'])
            except (ValueError, TypeError):
                failed_tools.append({'index': row['index'], 'operation': 'invalid_arguments'})
                continue
            operation = arguments.get('operation', '?')
            operations[operation] += 1
            outcome = row['result']
            if outcome.get('ok') is False or outcome.get('exitcode', 0) != 0:
                failed_tools.append({'index': row['index'], 'operation': operation, 'path': arguments.get('path'),
                                     'error': outcome.get('error'), 'exitcode': outcome.get('exitcode')})
            if operation == 'write' and outcome.get('ok'):
                writes.append({'index': row['index'], 'path': arguments['path']})
    payload = json.loads((run / 'task-and-materials.json').read_text(encoding='utf-8'))
    prefix = payload['task'].split('The following four instruction documents retain their logical directory scopes:')[0]
    generated = (result.get('usage') or {}).get('output_tokens', 0)
    limit_kind = ('total_generation_budget' if generated >= config['output_budget'] else
                  'per_response_cap' if result['state'] == 'generation_truncated' else result['state'])
    backend = run / 'work/flashrt/apps/causal_features/backend'
    final = result.get('final_evaluation', {})
    report = run / 'report.md'
    return {'run': str(run), 'terminal_result_available': True, 'condition': config['condition'],
            'requests_attempted': result.get('requests_attempted', len(list(run.glob('request-*.json')))),
            'usage_records_complete': result.get('usage_records_complete', result['requests'] == len(list(run.glob('request-*.json')))),
            'permissions': config['permissions'], 'checker': config['checker'],
            'runner_sha256': config['runner_sha256'], 'common_task_prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
            'steps': config['steps'], 'output_budget': config['output_budget'],
            'max_response_tokens': config.get('max_response_tokens', 8192),
            'state': result['state'], 'termination_classification': limit_kind,
            'usage': result.get('usage'), 'seconds': result['seconds'], 'responses': responses,
            'tool_operations': dict(operations), 'failed_tools': failed_tools, 'successful_writes': writes,
            'backend_files': [{'path': path.relative_to(backend).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}
                              for path in sorted(backend.rglob('*')) if path.is_file() and '__pycache__' not in path.parts],
            'report_available': report.exists(), 'report_sha256': sha(report) if report.exists() else None,
            'prose_review': None, 'final_correctness': final.get('correctness', {}).get('ok'),
            'final_benchmark': final.get('benchmark'), 'protected_files_changed': result['protected_files_changed'],
            'files_added_outside_backend': result.get('files_added_outside_backend'), 'result_sha256': sha(result_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [collect(path) for path in args.runs]
    terminal = [row for row in rows if row['terminal_result_available']]
    comparable_groups = {}
    for row in terminal:
        key = (row['runner_sha256'], row['common_task_prefix_sha256'], row['output_budget'], row['steps'],
               row['max_response_tokens'], row['permissions'], row['checker'])
        comparable_groups.setdefault(key, []).append(row['run'])
    result = {'scope': 'Development pilots; descriptive records only, no population effect estimate',
              'cache_note': 'Prior calibration requests can change cache hits; cached and uncached tokens remain separate',
              'missing_report_policy': 'No prose-quality score is assigned to an absent report',
              'runs': rows, 'same_runner_and_task_groups': list(comparable_groups.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'runs': len(rows), 'terminal_results': len(terminal), 'groups': len(comparable_groups)}))


if __name__ == '__main__':
    main()
