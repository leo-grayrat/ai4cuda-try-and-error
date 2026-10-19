"""Read finished repeated trials; preserve missingness and raw per-run records."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from collect_pilots import collect

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent


def arm(row):
    return row['condition'] + ('+permissions' if row['permissions'] else '') + ('+checker' if row['checker'] else '')


def summarize(manifest):
    rows = []
    for trial in manifest['trials']:
        run = ROOT / manifest['run_root'] / trial['id']
        if not (run / 'result.json').exists():
            rows.append({**trial, 'arm': arm(trial), 'terminal_result_available': False,
                         'started': (run / 'config.json').exists()})
            continue
        row = {**trial, **collect(run), 'arm': arm(trial), 'started': True}
        result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        row['api_response_seconds'] = sum(e.get('seconds', 0) for e in events if e['type'] == 'response')
        row['model_observation_eligible'] = result['state'] != 'api_or_trace_error'
        row['independent_benchmark_completed'] = bool(result.get('final_evaluation', {}).get('benchmark', {}).get('ok'))
        row['protected_source_unchanged'] = not row['protected_files_changed']
        row['valid_backend'] = row['final_correctness'] is True and row['protected_source_unchanged']
        row['submission_rejections'] = sum(e['type'] == 'tool'
            and e.get('result', {}).get('ok') is False
            and 'Submission requires' in e.get('result', {}).get('error', '') for e in events)
        usage = row.get('usage') or {}
        if row['usage_records_complete'] and all(usage.get(key) is not None for key in
                ['cached_input_tokens', 'uncached_input_tokens', 'output_tokens']):
            row['current_offpeak_estimate_usd'] = (usage['cached_input_tokens'] * .003
                + usage['uncached_input_tokens'] * .15 + usage['output_tokens'] * .6) / 1e6
            row['cache_neutral_offpeak_estimate_usd'] = (usage['input_tokens'] * .15
                + usage['output_tokens'] * .6) / 1e6
        else:
            row['current_offpeak_estimate_usd'] = None
            row['cache_neutral_offpeak_estimate_usd'] = None
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row['arm']].append(row)
    summaries = []
    for name, group in groups.items():
        done = [r for r in group if r['terminal_result_available']]
        eligible = [r for r in done if r['model_observation_eligible']]
        summary = {'arm': name, 'planned': len(group), 'terminal': len(done),
                   'started': sum(r['started'] for r in group), 'eligible_model_observations': len(eligible),
                   'api_or_trace_errors': len(done) - len(eligible)}
        for key in ['valid_backend', 'independent_benchmark_completed', 'report_available',
                    'prose_available', 'protected_source_unchanged']:
            summary[key + '_count'] = sum(r[key] for r in eligible)
        for field in ['seconds', 'current_offpeak_estimate_usd', 'cache_neutral_offpeak_estimate_usd']:
            values = [r[field] for r in done if r[field] is not None]
            summary[field + '_median'] = statistics.median(values) if values else None
        values = [(r['usage'] or {}).get('output_tokens') for r in done]
        values = [v for v in values if v is not None]
        summary['generated_tokens_median'] = statistics.median(values) if values else None
        summaries.append(summary)
    return {'generated_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Repeated observations in two fixed local tasks with one model; not a population claim',
            'pricing_source': 'https://api-docs.deepseek.com/quick_start/pricing/',
            'pricing_snapshot_date': '2026-10-19',
            'pricing_note': 'USD estimates at uniform off-peak rates; peak sensitivity is twice these values. Not account invoices.',
            'missing_prose_policy': 'Absent prose remains missing; no invented quality score',
            'trials': rows, 'conditions': summaries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((BASE / 'materials/repeat-batch-v1.json').read_text(encoding='utf-8'))
    result = summarize(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'planned': len(result['trials']),
                      'terminal': sum(r['terminal_result_available'] for r in result['trials'])}))


if __name__ == '__main__':
    main()
