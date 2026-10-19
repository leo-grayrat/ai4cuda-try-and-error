"""Read-only workflow timing and final raw measurement decomposition.

These observations do not score qualitative adherence or infer thought themes.
Only terminal runs are opened. All timestamps and measurement hashes remain
traceable to the original records; old gate decisions are never overwritten.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values, budget=20.0):
    if not values:
        return None
    return {'count': len(values), 'median_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)), 'max_ms': float(max(values)),
            'above_budget_count': sum(value > budget for value in values),
            'above_budget_fraction': sum(value > budget for value in values) / len(values)}


def decompose(load, budget=20.0):
    requests = load['requests']
    within = [(b-a)/1e6 for row in requests for a, b in zip(row['output_ns'], row['output_ns'][1:])]
    boundary = [(b['output_ns'][0]-a['output_ns'][-1])/1e6 for a, b in zip(requests, requests[1:])]
    flat = [value for row in requests for value in row['output_ns']]
    combined = [(b-a)/1e6 for a, b in zip(flat, flat[1:])]
    assert len(combined) == len(within) + len(boundary)
    assert sum(value > budget for value in combined) == sum(value > budget for value in within+boundary)
    return {'offered_requests_per_second': load['offered_requests_per_second'],
            'completed_requests_per_second': load['completed_requests_per_second'],
            'first_output_ms': load['first_output_ms'],
            'peak_external_queue_requests': load['peak_external_queue_requests'],
            'original_rate_gate_passed': load['rate_gate_passed'],
            'original_smoothness_gate_passed': load['smoothness_gate_passed'],
            'within_request_gaps': distribution(within, budget),
            'request_boundary_gaps': distribution(boundary, budget),
            'combined_gaps': distribution(combined, budget)}


def review(run):
    result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    responses = [row for row in events if row['type'] == 'response']
    started = datetime.fromisoformat(config['started_utc'])
    tools = []
    for event in events:
        if event['type'] != 'tool':
            continue
        try:
            arguments = json.loads(event['call']['function']['arguments'])
        except (ValueError, TypeError):
            continue
        outcome = event['result']
        tools.append({'index': event['index'], 'operation': arguments.get('operation'),
                      'path': arguments.get('path'), 'ok': outcome.get('ok'),
                      'exitcode': outcome.get('exitcode'), 'error': outcome.get('error'),
                      'elapsed_from_config_seconds': (datetime.fromisoformat(event['utc'])-started).total_seconds(),
                      'tool_seconds_including_queue': outcome.get('seconds'),
                      'evidence_directory': outcome.get('evidence_directory'),
                      'cumulative_generated_tokens': sum(row['usage']['completion_tokens']
                           for row in responses if row['index'] <= event['index'])})
    successful = [row for row in tools if row['operation'] == 'verify' and row['ok'] is True]
    writes = [row for row in tools if row['operation'] == 'write' and row['ok'] is True]
    first_verified = successful[0] if successful else None
    source_hashes = {name: sha(run / name) for name in ['config.json', 'events.jsonl', 'result.json']}
    paired = {}
    for name, outcome in result.get('final_evaluation', {}).get('benchmark', {}).get('paired_measurements', {}).items():
        directory = outcome.get('execution', {}).get('evidence_directory')
        path = run / 'control' / Path(directory).parent / name / 'result.json' if directory else None
        if not path or not path.is_file():
            paired[name] = {'raw_measurement_available': False}
            continue
        raw = json.loads(path.read_text(encoding='utf-8'))
        source_hashes[path.relative_to(run).as_posix()] = sha(path)
        loads = [decompose(load, raw['chunk_gap_budget_ms']) for load in raw.get('loads', [])]
        rates = [row['offered_requests_per_second'] for row in loads if row['original_rate_gate_passed']]
        paired[name] = {'raw_measurement_available': True, 'status': raw.get('status'),
                        'original_highest_tested_passing_rate': raw.get('highest_tested_passing_rate'),
                        'highest_tested_rate_without_backlog': max(rates) if rates else None,
                        'loads': loads}
    return {'run': run.relative_to(ROOT).as_posix(), 'state': result['state'],
            'model_observation_eligible': result['state'] not in ['infrastructure_interrupted', 'api_or_trace_error'],
            'task': config['task'], 'condition': config['condition'], 'permissions': config['permissions'],
            'checker': config['checker'], 'api_response_seconds': sum(row['seconds'] for row in responses),
            'wall_seconds': result['seconds'], 'first_successful_write': writes[0] if writes else None,
            'first_actor_requested_independent_correctness_pass': first_verified,
            'successful_writes_after_first_independent_pass': [row for row in writes
                if first_verified and row['elapsed_from_config_seconds'] > first_verified['elapsed_from_config_seconds']],
            'tool_events': tools, 'final_correctness': result.get('final_evaluation', {}).get('correctness', {}).get('ok'),
            'final_paired_measurements': paired, 'source_sha256': source_hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    runs = [(row, ROOT / manifest['run_root'] / row['id']) for row in manifest['trials']]
    rows = [{**trial, **review(run)} for trial, run in runs if (run / 'result.json').is_file()]
    payload = {'scope': 'Descriptive terminal-run evidence; no inferred thought themes or adherence scores',
               'timing_note': 'First actor-requested independent pass is not the first self-test pass. Wall and tool time include waiting; API time includes network and server time.',
               'gap_note': 'Post-hoc decomposition retains original combined gate. Boundary gaps can include waiting for input; separating them does not establish all within-request timing is optimal.',
               'trials': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'terminal_runs_reviewed': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
