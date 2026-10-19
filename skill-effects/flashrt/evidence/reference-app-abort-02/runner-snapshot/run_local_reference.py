"""Independently drive the local reference through IPC and retain raw evidence."""
import argparse
import hashlib
import json
import math
from multiprocessing import shared_memory
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--length', type=int, default=1043)
    parser.add_argument('--chunk', type=int, default=127)
    parser.add_argument('--requests', type=int, default=6)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--fixture-seed', type=int, default=917)
    parser.add_argument('--abort-after-chunks', type=int, default=0)
    parser.add_argument('--backend-file', type=Path, help='Candidate backend with the same repository/config CLI and IPC contract')
    args = parser.parse_args()
    assert args.requests >= 3 and min(args.channels, args.hidden, args.length, args.chunk) > 0
    repository = args.repository.resolve()
    base = Path(__file__).resolve().parent
    sys.path[:0] = [str(base / 'workloads'), str(repository)]
    import numpy as np
    import torch
    from causal_features.adapter import FeatureAdapter
    from causal_features.model import full_sequence_oracle

    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    config = {'channels': args.channels, 'hidden': args.hidden, 'length': args.length,
              'chunk': args.chunk, 'weight_seed': 703, 'device': args.device,
              'prefix': 'flashrt_features_' + uuid.uuid4().hex}
    (output / 'config.json').write_bytes((json.dumps(config, indent=2) + '\n').encode())
    random = np.random.default_rng(args.fixture_seed)
    inputs = [random.standard_normal((args.channels, args.length), dtype=np.float32)
              for _ in range(args.requests)]
    inputs[2] = inputs[0].copy()
    # The second request continues the first session. The third explicitly
    # resets and repeats the first input, detecting cross-session leakage.
    expected = [full_sequence_oracle(repository, config, values) for values in inputs]
    expected[1] = full_sequence_oracle(repository, config, np.concatenate(inputs[:2], axis=-1))[:, args.length:]
    np.savez_compressed(output / 'fixtures.npz', **{f'input_{i}': row for i, row in enumerate(inputs)})

    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = '0'
    environment['PYTHONPATH'] = str(base / 'workloads')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    stdout = (output / 'backend.stdout.log').open('w', encoding='utf-8')
    stderr = (output / 'backend.stderr.log').open('w', encoding='utf-8')
    command = [sys.executable, '-u', '-m', 'causal_features.reference', '--repository', str(repository),
               '--config', str(output / 'config.json')]
    if args.backend_file:
        command = [sys.executable, '-u', str(args.backend_file.resolve()), '--repository', str(repository),
                   '--config', str(output / 'config.json')]
    process, adapter, rows, collected = None, None, [], {}
    result = {'scope': 'Reference application development run, not a skill-treatment comparison',
              'schema': 2,
              'config': config, 'status': 'started', 'requests': rows,
              'timing_scope': 'Adapter-observed finite request bursts; not maximum sustainable-rate evidence',
              'oracle': 'CPU full-sequence evaluation; reference worker runs chunk caches through original IPC',
              'source_sha256': {str(path.relative_to(base)): digest(path.read_bytes())
                                for path in [Path(__file__), *(base / 'workloads/causal_features').glob('*.py')]}}
    if args.backend_file:
        result['candidate_entry_sha256'] = digest(args.backend_file.read_bytes())
        result['scope'] = 'Independent candidate evaluation development run, not a skill-treatment comparison'
    result['fixture_seed'] = args.fixture_seed
    result['upstream_source_sha256'] = {name: digest((repository / name).read_bytes()) for name in [
        'flashrt/layers/audio.py', 'flashrt/channels/shm_channel/tensor_buffer.py',
        'flashrt/channels/shm_channel/control_buffer.py']}
    for name, expected_hash in result['source_sha256'].items():
        snapshot = output / 'runner-snapshot' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((base / name).read_bytes())
        assert digest(snapshot.read_bytes()) == expected_hash
    for name, expected_hash in result['upstream_source_sha256'].items():
        snapshot = output / 'upstream-snapshot' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((repository / name).read_bytes())
        assert digest(snapshot.read_bytes()) == expected_hash
    if args.backend_file:
        (output / 'candidate-entry.py').write_bytes(args.backend_file.read_bytes())
    started = time.monotonic()
    chunks_seen = 0
    try:
        process = subprocess.Popen(command, cwd=repository, env=environment, stdout=stdout, stderr=stderr)
        result['backend_pid'] = process.pid
        adapter = FeatureAdapter(config)
        adapter.start()
        result['startup_seconds'] = time.monotonic() - started
        for index, features in enumerate(inputs):
            reset = index >= 2
            if reset:
                adapter.reset()
            request_id, published_ns = adapter.submit(features)
            pieces, timestamps = [], []
            for _ in range(math.ceil(args.length / args.chunk)):
                data, observed_ns = adapter.receive()
                pieces.append(data)
                timestamps.append(observed_ns)
                chunks_seen += 1
                if args.abort_after_chunks and chunks_seen >= args.abort_after_chunks:
                    raise RuntimeError('Injected harness interruption after receiving a real chunk')
            actual = np.concatenate(pieces, axis=-1)
            assert actual.shape == expected[index].shape
            match = bool(np.allclose(actual, expected[index], atol=3e-5, rtol=3e-4))
            row = {'index': index, 'request_id': request_id, 'reset_before': reset,
                   'timing_role': 'cold_initial_request' if index == 0 else 'after_initial_request',
                   'published_ns': published_ns, 'observed_chunk_ns': timestamps,
                   'input_sha256': digest(features.tobytes()), 'expected_sha256': digest(expected[index].tobytes()),
                   'output_sha256': digest(actual.tobytes()), 'numerical_match': match,
                   'max_abs_difference': float(np.max(np.abs(actual - expected[index]))),
                   'atol': 3e-5, 'rtol': 3e-4, 'first_output_ms': (timestamps[0] - published_ns) / 1e6,
                   'completion_ms': (timestamps[-1] - published_ns) / 1e6,
                   'inter_chunk_gap_ms': (np.diff(timestamps) / 1e6).tolist()}
            if row['first_output_ms'] < 0:
                raise RuntimeError('Output preceded the recorded input publication')
            rows.append(row)
            collected[f'output_{index}'] = actual
            assert match, f'Request {index} differs from the full-sequence oracle'
        assert rows[0]['output_sha256'] == rows[2]['output_sha256'], 'Reset did not reproduce the first session'
        result['status'] = 'numerical_checks_passed_pending_cleanup'
        np.savez_compressed(output / 'outputs.npz', **collected)
    except Exception as exc:
        result.update(status='invalid_run', error=f'{type(exc).__name__}: {exc}')
    finally:
        if adapter is not None:
            try:
                adapter.terminate()
            except Exception as exc:
                result['termination_error'] = f'{type(exc).__name__}: {exc}'
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            result['backend_exitcode'] = process.returncode
            result['backend_alive_after_cleanup'] = process.poll() is None
        if adapter is not None:
            adapter.close()
        stdout.close()
        stderr.close()
        leftovers = []
        names = [config['prefix'] + '__ctrl'] + [config['prefix'] + '_' + kind + suffix
                  for kind in ['input', 'output', 'metadata'] for suffix in ['__meta', '__tensor']]
        for name in names:
            try:
                handle = shared_memory.SharedMemory(name=name, create=False)
            except FileNotFoundError:
                continue
            leftovers.append(name)
            handle.unlink()
            handle.close()
        result['shared_memory_leftovers_after_cleanup'] = leftovers
        stderr_text = (output / 'backend.stderr.log').read_text(encoding='utf-8')
        result['backend_stderr'] = stderr_text
        # This exact interpreter-startup diagnostic is also emitted by the
        # parent interpreter before task code runs. Preserve it, but do not
        # misclassify it as a leaked process or a backend computation error.
        known_launcher_line = 'Failed to find real location of ' + sys.executable
        stderr_lines = [line for line in stderr_text.splitlines() if line.strip()]
        result['known_interpreter_diagnostics'] = [line for line in stderr_lines if line == known_launcher_line]
        unexpected_stderr = [line for line in stderr_lines if line != known_launcher_line]
        result['unexpected_backend_stderr'] = unexpected_stderr
        cleanup_ok = process is not None and process.returncode == 0 and not leftovers
        result['cleanup_verified'] = cleanup_ok
        if result['status'] == 'numerical_checks_passed_pending_cleanup':
            passed_status = 'candidate_run_passed' if args.backend_file else 'reference_run_passed'
            result['status'] = (passed_status if cleanup_ok and not unexpected_stderr
                                else 'invalid_cleanup' if not cleanup_ok else 'invalid_backend_stderr')
        changed = [name for name, expected_hash in result['source_sha256'].items()
                   if digest((base / name).read_bytes()) != expected_hash]
        changed += [name for name, expected_hash in result['upstream_source_sha256'].items()
                    if digest((repository / name).read_bytes()) != expected_hash]
        if args.backend_file and digest(args.backend_file.read_bytes()) != result['candidate_entry_sha256']:
            changed.append(str(args.backend_file))
        result['source_changed_during_run'] = changed
        if changed:
            result['status'] = 'invalid_source_changed'
        result['artifacts'] = {path.name: digest(path.read_bytes()) for path in output.iterdir()
                               if path.is_file() and path.name != 'result.json'}
        result['elapsed_seconds'] = time.monotonic() - started
        (output / 'result.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps({'status': result['status'], 'requests_completed': len(rows),
                      'cleanup_verified': result['cleanup_verified'], 'output': str(output)}))
    if args.abort_after_chunks:
        return 0 if result['status'] == 'invalid_run' and result['cleanup_verified'] and 'Injected' in result.get('error', '') else 1
    return 0 if result['status'] in {'reference_run_passed', 'candidate_run_passed'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
