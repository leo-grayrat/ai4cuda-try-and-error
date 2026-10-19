"""Observable API development trial for real FlashRT-derived local tasks.

This is a pilot, not a frozen formal campaign. Tool permissions are enforced
on file operations, not an OS sandbox for Python executed by a candidate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import requests

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from gpu_lock import gpu_lock
from runtime import FileTools, summarize_usage

BACKEND = 'flashrt/apps/causal_features/backend/'
TOOL = {'type': 'function', 'function': {'name': 'workspace',
        'description': 'Inspect/edit trial-local files; execute backend scripts; independently verify and benchmark a backend; finish with a report. No shell or dependency installation tool.',
        'parameters': {'type': 'object', 'properties': {
            'operation': {'type': 'string', 'enum': ['list', 'read', 'write', 'delete', 'execute', 'verify', 'benchmark', 'finish']},
            'path': {'type': 'string', 'description': 'Relative file path. execute requires an existing backend/*.py file written first; shell commands, -c code and inline scripts are not paths.'},
            'content': {'type': 'string', 'description': 'Complete UTF-8 file contents for write. To run Python, write a backend script then execute its path.'},
            'report': {'type': 'string'}},
            'required': ['operation']}}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def code_hashes(directory):
    return {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob('*.py') if '__pycache__' not in p.parts}


class Workspace:
    def __init__(self, run, repository, permissions, checker):
        self.run, self.repository = run, repository
        self.work, self.control = run / 'work', run / 'control'
        self.control.mkdir()
        shutil.copytree(repository, self.work, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(BASE / 'workloads/causal_features', self.work / 'flashrt/apps/causal_features',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (self.work / BACKEND).mkdir()
        self.files = FileTools(self.work, permissions=False)
        self.permissions, self.checker = permissions, checker
        self.counter, self.finished, self.last_correctness, self.last_benchmark = 0, False, None, None
        self.entry = BACKEND + 'worker.py'

    def process(self, command, folder, timeout=180, lock=False):
        folder.mkdir(parents=True, exist_ok=False)
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(BASE / 'workloads') + os.pathsep + str(self.work),
                           PYTHONDONTWRITEBYTECODE='1')
        started = time.monotonic()
        def invoke():
            with (folder / 'stdout.log').open('w', encoding='utf-8') as out, (folder / 'stderr.log').open('w', encoding='utf-8') as err:
                proc = subprocess.Popen(command, cwd=self.work, env=environment, stdout=out, stderr=err)
                try:
                    return proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Research-controller-owned process tree only.
                    import psutil
                    children = psutil.Process(proc.pid).children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    proc.kill()
                    proc.wait()
                    psutil.wait_procs(children, timeout=5)
                    return -999
        if lock:
            with gpu_lock(BASE.parent / 'runs/native-gpu.lock', timeout=60):
                exitcode = invoke()
        else:
            exitcode = invoke()
        return {'exitcode': exitcode, 'seconds': time.monotonic() - started,
                'stdout': (folder / 'stdout.log').read_text(encoding='utf-8')[-14000:],
                'stderr': (folder / 'stderr.log').read_text(encoding='utf-8')[-8000:],
                'evidence_directory': folder.relative_to(self.control).as_posix()}

    def verify(self, path, hidden=False):
        entry = self.files.path(path)
        if not path.startswith(BACKEND) or entry.suffix != '.py':
            raise ValueError('Backend entry must be a Python script under the backend directory')
        self.counter += 1
        self.last_correctness = None
        before = code_hashes(self.work / BACKEND)
        folder = self.control / f'verify-{self.counter:03d}'
        command = [sys.executable, str(BASE / 'run_local_reference.py'), '--repository', str(self.repository),
                   '--backend-file', str(entry), '--output', str(folder / 'evaluation'),
                   '--fixture-seed', '1429' if hidden else '917']
        execution = self.process(command, folder, lock=True)
        result_path = folder / 'evaluation/result.json'
        verdict = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
        passed = execution['exitcode'] == 0 and verdict.get('status') == 'candidate_run_passed'
        passed = passed and code_hashes(self.work / BACKEND) == before
        if passed:
            self.last_correctness = {'result': result_path, 'code': code_hashes(self.work / BACKEND), 'entry': path}
            self.entry = path
        return {'ok': passed, 'execution': execution, 'status': verdict.get('status'),
                'request_checks': [{key: row[key] for key in ['index', 'numerical_match', 'max_abs_difference']}
                                   for row in verdict.get('requests', [])],
                'cleanup_verified': verdict.get('cleanup_verified')}

    def benchmark(self, path):
        current = code_hashes(self.work / BACKEND)
        if not self.last_correctness or self.last_correctness['entry'] != path or current != self.last_correctness['code']:
            raise ValueError('Verify the current backend code before benchmarking')
        self.counter += 1
        self.last_benchmark = None
        folder = self.control / f'benchmark-{self.counter:03d}'
        folder.mkdir()
        results = {}
        for name in ['reference', 'candidate']:
            command = [sys.executable, str(BASE / 'benchmark_local.py'), '--repository', str(self.repository),
                       '--correctness-result', str(BASE / 'evidence/reference-app-03/result.json') if name == 'reference'
                       else str(self.last_correctness['result']), '--output', str(folder / name), '--rates', '50,100,200']
            if name == 'candidate':
                command += ['--backend-file', str(self.files.path(path))]
            execution = self.process(command, folder / (name + '-controller'), timeout=180)
            data_path = folder / name / 'result.json'
            data = json.loads(data_path.read_text(encoding='utf-8')) if data_path.exists() else {}
            results[name] = {'execution': execution, 'status': data.get('status'),
                             'highest_tested_passing_rate': data.get('highest_tested_passing_rate'),
                             'loads': [{key: row[key] for key in ['offered_requests_per_second', 'first_output_ms', 'output_gap_ms',
                                                                 'rate_gate_passed', 'smoothness_gate_passed']}
                                       for row in data.get('loads', [])]}
        passed = all(row['status'] == 'load_sweep_completed' and row['execution']['exitcode'] == 0 for row in results.values())
        passed = passed and code_hashes(self.work / BACKEND) == current
        if passed:
            self.last_benchmark = {'code': current, 'entry': path}
        return {'ok': passed, 'paired_measurements': results,
                'scope': 'Pilot finite windows, fixed reference-then-candidate order; no aggregate effect claim'}

    def execute(self, arguments):
        operation = arguments['operation']
        path = arguments.get('path', '')
        if operation == 'list':
            if path.rstrip('/') == 'evidence' or path.startswith('evidence/'):
                relative = path[len('evidence'):].lstrip('/')
                target = (self.control / relative).resolve()
                if not target.is_relative_to(self.control):
                    raise ValueError('Evidence path escapes the controller')
                return {'files': ['evidence/' + p.relative_to(self.control).as_posix()
                                  for p in target.rglob('*') if p.is_file()]} if target.is_dir() else {'files': []}
            return {'files': [p.relative_to(self.work).as_posix() for p in self.work.rglob('*')
                              if p.is_file() and '__pycache__' not in p.parts and (not path or p.relative_to(self.work).as_posix().startswith(path))]}
        if operation == 'read':
            if path.startswith('evidence/'):
                target = (self.control / path.removeprefix('evidence/')).resolve()
                if not target.is_relative_to(self.control):
                    raise ValueError('Evidence path escapes the controller')
                return {'content': target.read_text(encoding='utf-8')}
            return {'content': self.files.read(path)}
        if operation in {'write', 'delete'}:
            self.files.path(path, write=True)
            if self.permissions and not path.startswith(BACKEND):
                raise PermissionError('Write denied: only the assigned backend directory is writable')
            if operation == 'write':
                self.files.write(path, arguments['content'])
            else:
                self.files.delete(path)
            return {'ok': True}
        if operation == 'execute':
            target = self.files.path(path)
            if not path.startswith(BACKEND) or target.suffix != '.py':
                raise ValueError('Execute only Python scripts under the backend directory')
            self.counter += 1
            return self.process([sys.executable, str(target)], self.control / f'execute-{self.counter:03d}', lock=True)
        if operation == 'verify':
            return self.verify(path)
        if operation == 'benchmark':
            return self.benchmark(path)
        if operation == 'finish':
            current = code_hashes(self.work / BACKEND)
            if self.checker and (not self.last_correctness or not self.last_benchmark
                                 or self.last_correctness['code'] != current or self.last_benchmark['code'] != current
                                 or self.last_correctness['entry'] != self.last_benchmark['entry']
                                 or self.last_correctness['entry'] != self.entry):
                raise ValueError('Submission requires correctness and load measurements for the current code')
            report = arguments.get('report', '')
            if not report.strip():
                raise ValueError('Submit a report explaining decisions and evidence')
            (self.run / 'report.md').write_bytes(report.encode('utf-8'))
            self.finished = True
            return {'ok': True, 'submitted': True}
        raise ValueError('Unknown workspace operation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', choices=['original', 'formatted', 'phase4'], required=True)
    parser.add_argument('--permissions', action='store_true')
    parser.add_argument('--checker', action='store_true')
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--output-budget', type=int, default=24000)
    parser.add_argument('--max-response-tokens', type=int, default=24000)
    args = parser.parse_args()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    workspace = Workspace(run, args.repository.resolve(), args.permissions, args.checker)
    manifest = json.loads((BASE / 'materials/verbatim-v1/manifest.json').read_text(encoding='utf-8'))
    instructions = []
    for index, row in enumerate(manifest['documents']):
        source = BASE / 'materials/verbatim-v1' / row['original']
        if args.condition == 'formatted':
            source = BASE / 'materials/format-v1' / f'd{index:02d}.md'
        elif args.condition == 'phase4' and index == 0:
            source = BASE / 'materials/phase4-draft-v1/root-draft.md'
        data = source.read_bytes()
        destination = workspace.work / row['logical_path']
        destination.write_bytes(data)
        instructions.append({'logical_path': row['logical_path'], 'sha256': sha(source), 'content': data.decode('utf-8')})
    task = f'''Optimize the assigned causal_features application on one RTX 5060 8GB GPU, Windows, PyTorch 2.11.0+cu128.
The researcher-supplied sequential reference, fixed adapter and task specification are under flashrt/apps/causal_features/. Read CONTRACT.zh.md, model.py, reference.py and adapter.py. Your workspace is {BACKEND}. Supply a standalone backend/worker.py accepting --repository and --config, honoring exactly the supplied adapter protocol. The task is continuous feature processing, not speech-recognition quality. Weights are fixed, no downloads or installation are needed.
Use the original IR tools under flashrt/ir to express and validate state dependencies, then explore meaningful local optimizations. Multi-GPU requirements do not apply on this assigned single GPU. Root run-script/path examples map to this local app's backend; do not edit shared or reference files. The independent controller supplies module paths; candidate scripts can import causal_features.adapter and causal_features.model, and flashrt.ir. Verification compares six complete requests with a CPU full-sequence oracle, including continuation, reset and a tail chunk. Benchmark runs paired reference and candidate finite offered-load sweeps with a 20ms chunk-gap budget. All code and measurements must concern the supplied task.
workspace.execute runs a backend Python script from repo root, with the pinned FlashRT package and causal_features importable, for your IR and diagnostic scripts. workspace.verify and benchmark take the relative path of your standalone backend entry. Their evidence lives outside your writable workspace and can be read under evidence/. No shell tool is supplied. Write notes and variants under your backend directory. Finish using workspace.finish with a Chinese report in connected prose: explain hypotheses, evidence, failures, trade-offs and the remaining unfinished requirements precisely.
This development observation has an external cap of {args.steps} API requests and {args.output_budget} generated tokens including reasoning. The cap overrides any unlimited-session language in the supplied documents. Finish submits the observed state and evidence, not a claim that every upstream completion criterion was achieved. Submit an honest report within the observation budget, explicitly listing unmet requirements instead of inventing completion. This is a pilot; no minimum variant requirement is silently removed. The assigned writable area is the backend directory. Independent correctness always uses the protected research reference.
The following four instruction documents retain their logical directory scopes:
''' + '\n\n'.join('FILE: ' + row['logical_path'] + '\n' + row['content'] for row in instructions)
    save(run / 'task-and-materials.json', {'task': task, 'documents': [{k: v for k, v in row.items() if k != 'content'} for row in instructions]})
    protected = {p.relative_to(workspace.work).as_posix(): sha(p) for p in workspace.work.rglob('*') if p.is_file() and not p.relative_to(workspace.work).as_posix().startswith(BACKEND)}
    save(run / 'protected-inventory.json', protected)
    auth = json.loads((Path.home() / '.local/share/opencode/auth.json').read_text(encoding='utf-8'))
    headers = {'Authorization': 'Bearer ' + auth['deepseek']['key']}
    messages = [{'role': 'system', 'content': 'You are a coding agent conducting the assigned streaming-backend optimization using workspace tools.'},
                {'role': 'user', 'content': task}]
    save(run / 'config.json', {'purpose': 'development pilot, not formal effectiveness evidence', 'condition': args.condition,
                              'permissions': args.permissions, 'checker': args.checker, 'steps': args.steps,
                              'output_budget': args.output_budget, 'max_response_tokens': args.max_response_tokens,
                              'model': 'deepseek-flash', 'runner_sha256': sha(Path(__file__)),
                              'started_utc': datetime.now(timezone.utc).isoformat()})
    (run / 'runner-snapshot.py').write_bytes(Path(__file__).read_bytes())
    usage, state, started = [], 'step_budget', time.monotonic()
    with (run / 'events.jsonl').open('w', encoding='utf-8') as events:
        def event(row):
            events.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), **row}, ensure_ascii=False) + '\n')
            events.flush()
        for index in range(args.steps):
            remaining = args.output_budget - sum(row['completion_tokens'] for row in usage)
            if remaining < 256:
                state = 'output_budget'
                break
            body = {'model': 'deepseek-flash', 'messages': messages, 'tools': [TOOL], 'max_tokens': min(args.max_response_tokens, remaining),
                    'temperature': 1.0, 'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
            save(run / f'request-{index:03d}.json', body)
            print(f'API request {index + 1}, remaining generated tokens {remaining}', flush=True)
            api_started = time.monotonic()
            try:
                response = requests.post('https://api.deepseek.com/chat/completions', headers=headers, json=body, timeout=(20, 600))
                data = response.json()
                save(run / f'response-{index:03d}.json', data)
                response.raise_for_status()
                usage.append(data['usage'])
                choice = data['choices'][0]
                messages.append(choice['message'])
                event({'type': 'response', 'index': index, 'usage': data['usage'], 'model': data.get('model'),
                       'finish_reason': choice['finish_reason'], 'seconds': time.monotonic() - api_started})
                if choice['finish_reason'] == 'length':
                    state = 'output_budget' if sum(row['completion_tokens'] for row in usage) >= args.output_budget else 'generation_truncated'
                    break
                calls = choice['message'].get('tool_calls') or []
                if not calls:
                    messages.append({'role': 'user', 'content': 'Continue with workspace tools or submit your report using finish. '
                                     + json.dumps({'remaining_api_requests': args.steps - index - 1,
                                                   'remaining_generated_tokens': args.output_budget - sum(row['completion_tokens'] for row in usage)})})
                for call in calls:
                    arguments = {}
                    try:
                        if call['function']['name'] != 'workspace':
                            raise ValueError('Unknown tool')
                        arguments = json.loads(call['function']['arguments'])
                        result = workspace.execute(arguments)
                    except (ValueError, OSError, KeyError, TypeError) as error:
                        result = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
                    result['observation_budget'] = {'remaining_api_requests': args.steps - index - 1,
                                                    'remaining_generated_tokens': args.output_budget - sum(row['completion_tokens'] for row in usage)}
                    event({'type': 'tool', 'index': index, 'call': call, 'result': result})
                    messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result, ensure_ascii=False)})
                    print(f"tool {arguments.get('operation', '?')}: {str(result)[:180]}", flush=True)
                if workspace.finished:
                    state = 'submitted'
                    break
            except (requests.RequestException, ValueError, KeyError) as error:
                state = 'api_or_trace_error'
                event({'type': 'error', 'index': index, 'error': str(error)})
                break
        save(run / 'messages.json', messages)
        final = {}
        entry = workspace.work / workspace.entry
        if entry.exists():
            final['correctness'] = workspace.verify(workspace.entry, hidden=True)
            if final['correctness']['ok']:
                final['benchmark'] = workspace.benchmark(workspace.entry)
        changed = [path for path, expected in protected.items() if not (workspace.work / path).exists() or sha(workspace.work / path) != expected]
        added = [p.relative_to(workspace.work).as_posix() for p in workspace.work.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and not p.relative_to(workspace.work).as_posix().startswith(BACKEND)
                 and p.relative_to(workspace.work).as_posix() not in protected]
        result = {'state': state, 'seconds': time.monotonic() - started, 'requests': len(usage),
                  'requests_attempted': len(list(run.glob('request-*.json'))),
                  'usage_records_complete': len(usage) == len(list(run.glob('request-*.json'))),
                  'usage': summarize_usage(usage) if usage else None, 'protected_files_changed': changed,
                  'files_added_outside_backend': added,
                  'report_available': (run / 'report.md').exists(), 'final_evaluation': final}
        save(run / 'result.json', result)
        print(json.dumps({'state': state, 'requests': len(usage), 'usage': result['usage'], 'final_correctness': final.get('correctness', {}).get('ok')}), flush=True)
    return 0 if state == 'submitted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
