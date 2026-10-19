"""Finite offered-load sweep with explicit backpressure and output-gap evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--correctness-result', type=Path, required=True)
    parser.add_argument('--backend-file', type=Path)
    parser.add_argument('--backend-repository', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rates', default='10,25,50,100,200')
    parser.add_argument('--seconds', type=float, default=1.0)
    parser.add_argument('--chunk-budget-ms', type=float, default=20.0)
    args = parser.parse_args()
    assert args.seconds > 0 and args.chunk_budget_ms > 0
    base = Path(__file__).resolve().parent
    sys.path[:0] = [str(base / 'workloads'), str(args.repository.resolve()), str(base.parent)]
    import numpy as np
    from gpu_lock import gpu_lock
    from local_session import LocalSession, file_hash
    rates = [float(value) for value in args.rates.split(',')]
    assert rates and all(value > 0 for value in rates) and len(set(rates)) == len(rates)
    correctness = json.loads(args.correctness_result.read_text(encoding='utf-8'))
    required = 'candidate_run_passed' if args.backend_file else 'reference_run_passed'
    assert correctness['status'] == required and correctness['cleanup_verified']
    if args.backend_file:
        assert file_hash(args.backend_file) == correctness['candidate_entry_sha256'], 'Candidate changed after correctness'
    for path, expected in correctness['source_sha256'].items():
        if path.replace('\\', '/').startswith('workloads/'):
            assert file_hash(base / path) == expected, 'Shared task source changed after correctness'
    for path, expected in correctness['upstream_source_sha256'].items():
        assert file_hash(args.repository / path) == expected, 'Upstream source changed after correctness'
    config = dict(correctness['config'], prefix='flashrt_load_' + uuid.uuid4().hex)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'benchmark-snapshot.py').write_bytes(Path(__file__).read_bytes())
    initial_benchmark_hash = file_hash(Path(__file__))
    result = {'status': 'started', 'scope': 'Finite-window load calibration, not an infinite sustainability proof',
              'correctness_controller_sha256': correctness['source_sha256'].get('run_local_reference.py'),
              'correctness_result_sha256': file_hash(args.correctness_result), 'config': config,
              'requested_rates': rates, 'window_seconds': args.seconds, 'chunk_gap_budget_ms': args.chunk_budget_ms,
              'admission_contract': 'One request in flight; scheduled arrivals beyond it remain in the external offered queue',
              'rate_gate': 'Final admission lateness <= one input period and peak offered queue <= one request',
              'smoothness_gate': 'At most 1 percent of output gaps exceed the configured chunk budget',
              'cold_and_warmup': [], 'loads': []}
    session = LocalSession(args.repository, config, args.output / 'backend', args.backend_file, args.backend_repository)
    random = np.random.default_rng(1017)
    n_chunks = math.ceil(config['length'] / config['chunk'])

    def request(features):
        identifier, published = session.adapter.submit(features)
        observed = []
        for _ in range(n_chunks):
            values, stamp = session.adapter.receive()
            assert np.isfinite(values).all(), 'Nonfinite benchmark output'
            observed.append(stamp)
        return {'request_id': identifier, 'published_ns': published, 'output_ns': observed,
                'input_sha256': hashlib.sha256(features.tobytes()).hexdigest()}

    try:
        with gpu_lock(base.parent / 'runs/native-gpu.lock', timeout=1800) as waited:
            result['gpu_lock_wait_seconds'] = waited
            with session:
                for _ in range(2):
                    result['cold_and_warmup'].append(request(random.standard_normal((config['channels'], config['length']), dtype=np.float32)))
                for rate in rates:
                    session.adapter.reset()
                    count = max(5, math.ceil(args.seconds * rate))
                    fixtures = [random.standard_normal((config['channels'], config['length']), dtype=np.float32)
                                for _ in range(count)]
                    start = time.perf_counter_ns()
                    period_ns = 1e9 / rate
                    requests, lateness, queue_sizes = [], [], []
                    for index in range(count):
                        due = start + int(index * period_ns)
                        remaining = (due - time.perf_counter_ns()) / 1e9
                        if remaining > 0:
                            time.sleep(remaining)
                        admission = time.perf_counter_ns()
                        lateness.append(max(0, admission - due) / 1e6)
                        offered = min(count, int((admission - start) / period_ns) + 1)
                        queue_sizes.append(max(0, offered - index - 1))
                        measured = request(fixtures[index])
                        measured.update(scheduled_ns=due, admission_ns=admission, external_queue_before_admission=queue_sizes[-1])
                        requests.append(measured)
                    end = time.perf_counter_ns()
                    stamps = [stamp for row in requests for stamp in row['output_ns']]
                    gaps = np.diff(stamps) / 1e6
                    first = [(row['output_ns'][0] - row['published_ns']) / 1e6 for row in requests]
                    rate_stable = lateness[-1] <= 1000 / rate and max(queue_sizes) <= 1
                    exceed = float(np.mean(gaps > args.chunk_budget_ms))
                    row = {'offered_requests_per_second': rate, 'requests': requests,
                           'elapsed_seconds': (end - start) / 1e9,
                           'completed_requests_per_second': count * 1e9 / (end - start),
                           'output_chunks_per_second': len(stamps) * 1e9 / (end - start),
                           'admission_lateness_ms': lateness, 'peak_external_queue_requests': max(queue_sizes),
                           'first_output_ms': {'median': float(np.median(first)), 'p95': float(np.percentile(first, 95))},
                           'output_gap_ms': {'median': float(np.median(gaps)), 'p95': float(np.percentile(gaps, 95)),
                                             'over_budget_fraction': exceed},
                           'rate_gate_passed': rate_stable, 'smoothness_gate_passed': exceed <= 0.01,
                           'both_gates_passed': rate_stable and exceed <= 0.01}
                    result['loads'].append(row)
                    (args.output / 'progress.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
                    print(json.dumps({key: row[key] for key in ['offered_requests_per_second', 'peak_external_queue_requests',
                                                               'rate_gate_passed', 'smoothness_gate_passed']}), flush=True)
            result['status'] = 'load_sweep_completed' if session.evidence['valid_exit'] else 'invalid_backend_exit'
    except Exception as error:
        result.update(status='load_sweep_failed', error=f'{type(error).__name__}: {error}')
    result['session'] = session.evidence
    result['upstream_changed'] = [path for path, expected in correctness['upstream_source_sha256'].items()
                                  if file_hash(args.repository / path) != expected]
    if result['upstream_changed'] or file_hash(Path(__file__)) != initial_benchmark_hash:
        result['status'] = 'invalid_source_changed'
    passing = [row for row in result['loads'] if row['both_gates_passed']]
    result['highest_tested_passing_rate'] = max((row['offered_requests_per_second'] for row in passing), default=None)
    result['benchmark_sha256'] = file_hash(Path(__file__))
    (args.output / 'result.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps({'status': result['status'], 'highest_tested_passing_rate': result['highest_tested_passing_rate']}))
    return 0 if result['status'] == 'load_sweep_completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
