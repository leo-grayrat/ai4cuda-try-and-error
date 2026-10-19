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
    def __init__(self, run, repository, permissions, checker, task='causal_features'):
        self.run, self.repository = run, repository
        self.task, self.backend = task, f'flashrt/apps/{task}/backend/'
        self.work, self.control = run / 'work', run / 'control'
        self.control.mkdir()
        shutil.copytree(repository, self.work, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for package in sorted({'causal_features', task}):
            shutil.copytree(BASE / 'workloads' / package, self.work / 'flashrt/apps' / package,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (self.work / self.backend).mkdir()
        self.files = FileTools(self.work, permissions=False)
        self.permissions, self.checker = permissions, checker
        self.counter, self.finished, self.last_correctness, self.last_benchmark = 0, False, None, None
        self.entry = self.backend + 'worker.py'
        self.submission_attempts = 0

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
            with gpu_lock(BASE.parent / 'runs/native-gpu.lock', timeout=1800):
                exitcode = invoke()
        else:
            exitcode = invoke()
        return {'exitcode': exitcode, 'seconds': time.monotonic() - started,
                'stdout': (folder / 'stdout.log').read_text(encoding='utf-8')[-14000:],
                'stderr': (folder / 'stderr.log').read_text(encoding='utf-8')[-8000:],
                'evidence_directory': folder.relative_to(self.control).as_posix()}

    def verify(self, path, hidden=False):
        entry = self.files.path(path)
        if not path.startswith(self.backend) or entry.suffix != '.py':
            raise ValueError('Backend entry must be a Python script under the backend directory')
        if not entry.is_file():
            raise FileNotFoundError('Write the backend entry before requesting verification')
        self.counter += 1
        self.last_correctness = None
        before = code_hashes(self.work / self.backend)
        folder = self.control / f'verify-{self.counter:03d}'
        command = [sys.executable, str(BASE / 'run_local_reference.py'), '--repository', str(self.repository),
                   '--backend-file', str(entry), '--output', str(folder / 'evaluation'),
                   '--task', self.task, '--backend-repository', str(self.work),
                   '--fixture-seed', '1429' if hidden else '917']
        execution = self.process(command, folder, lock=True)
        result_path = folder / 'evaluation/result.json'
        verdict = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
        passed = execution['exitcode'] == 0 and verdict.get('status') == 'candidate_run_passed'
        passed = passed and code_hashes(self.work / self.backend) == before
        if passed:
            self.last_correctness = {'result': result_path, 'code': code_hashes(self.work / self.backend), 'entry': path}
            self.entry = path
        return {'ok': passed, 'execution': execution, 'status': verdict.get('status'),
                'request_checks': [{key: row[key] for key in ['index', 'numerical_match', 'max_abs_difference']}
                                   for row in verdict.get('requests', [])],
                'cleanup_verified': verdict.get('cleanup_verified')}

    def benchmark(self, path):
        current = code_hashes(self.work / self.backend)
        if not self.last_correctness or self.last_correctness['entry'] != path or current != self.last_correctness['code']:
            raise ValueError('Verify the current backend code before benchmarking')
        self.counter += 1
        self.last_benchmark = None
        folder = self.control / f'benchmark-{self.counter:03d}'
        folder.mkdir()
        results = {}
        for name in ['reference', 'candidate']:
            command = [sys.executable, str(BASE / 'benchmark_local.py'), '--repository', str(self.repository),
                       '--correctness-result', str(BASE / 'evidence' / ('reference-v3-' + self.task) / 'result.json') if name == 'reference'
                       else str(self.last_correctness['result']), '--output', str(folder / name), '--rates', '50,100,200']
            if name == 'candidate':
                command += ['--backend-file', str(self.files.path(path)), '--backend-repository', str(self.work)]
            execution = self.process(command, folder / (name + '-controller'), timeout=180)
            data_path = folder / name / 'result.json'
            data = json.loads(data_path.read_text(encoding='utf-8')) if data_path.exists() else {}
            results[name] = {'execution': execution, 'status': data.get('status'),
                             'highest_tested_passing_rate': data.get('highest_tested_passing_rate'),
                             'loads': [{key: row[key] for key in ['offered_requests_per_second', 'first_output_ms', 'output_gap_ms',
                                                                 'rate_gate_passed', 'smoothness_gate_passed']}
                                       for row in data.get('loads', [])]}
        passed = all(row['status'] == 'load_sweep_completed' and row['execution']['exitcode'] == 0 for row in results.values())
        passed = passed and code_hashes(self.work / self.backend) == current
        if passed:
            self.last_benchmark = {'code': current, 'entry': path}
        return {'ok': passed, 'paired_measurements': results,
                'scope': 'Pilot finite windows, fixed reference-then-candidate order; no aggregate effect claim'}

    def execute(self, arguments):
        if not isinstance(arguments, dict):
            raise ValueError('Workspace arguments must be an object')
        operation = arguments.get('operation')
        if not isinstance(operation, str) or operation not in {'list', 'read', 'write', 'delete', 'execute', 'verify', 'benchmark', 'finish'}:
            raise ValueError('Unknown workspace operation')
        path = arguments.get('path', '')
        if not isinstance(path, str):
            raise ValueError('path must be a relative path string')
        if operation in {'read', 'write', 'delete', 'execute', 'verify', 'benchmark'} and not path:
            raise ValueError('This operation requires path')
        if operation == 'write' and not isinstance(arguments.get('content'), str):
            raise ValueError('write requires string content')
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
            if self.permissions and not path.startswith(self.backend):
                raise PermissionError('Write denied: only the assigned backend directory is writable')
            if operation == 'write':
                self.files.write(path, arguments['content'])
            else:
                self.files.delete(path)
            return {'ok': True}
        if operation == 'execute':
            target = self.files.path(path)
            if not path.startswith(self.backend) or target.suffix != '.py':
                raise ValueError('Execute only Python scripts under the backend directory')
            self.counter += 1
            return self.process([sys.executable, str(target)], self.control / f'execute-{self.counter:03d}', lock=True)
        if operation == 'verify':
            return self.verify(path)
        if operation == 'benchmark':
            return self.benchmark(path)
        if operation == 'finish':
            report = arguments.get('report', '')
            if not isinstance(report, str) or not report.strip():
                raise ValueError('Submit a report explaining decisions and evidence')
            self.submission_attempts += 1
            attempt = self.control / f'submission-attempt-{self.submission_attempts:03d}.md'
            attempt.write_bytes(report.encode('utf-8'))
            current = code_hashes(self.work / self.backend)
            if self.checker and (not self.last_correctness or not self.last_benchmark
                                 or self.last_correctness['code'] != current or self.last_benchmark['code'] != current
                                 or self.last_correctness['entry'] != self.last_benchmark['entry']
                                 or self.last_correctness['entry'] != self.entry):
                raise ValueError('Submission requires correctness and load measurements for the current code')
            (self.run / 'report.md').write_bytes(report.encode('utf-8'))
            self.finished = True
            return {'ok': True, 'submitted': True}
        raise ValueError('Unknown workspace operation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', choices=['task_only', 'source_core', 'original', 'formatted', 'phase4'], required=True)
    parser.add_argument('--task', choices=['causal_features', 'branch_features'], default='causal_features')
    parser.add_argument('--purpose', choices=['development', 'formal'], default='development')
    parser.add_argument('--permissions', action='store_true')
    parser.add_argument('--checker', action='store_true')
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--output-budget', type=int, default=24000)
    parser.add_argument('--max-response-tokens', type=int)
    parser.add_argument('--report-budget', type=int, default=4000)
    parser.add_argument('--report-requests', type=int, default=4)
    args = parser.parse_args()
    if args.max_response_tokens is None:
        args.max_response_tokens = args.output_budget
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    assert 0 <= args.report_budget < args.output_budget and 0 <= args.report_requests < args.steps
    assert bool(args.report_budget) == bool(args.report_requests)
    development_budget = args.output_budget - args.report_budget
    workspace = Workspace(run, args.repository.resolve(), args.permissions, args.checker, args.task)
    manifest = json.loads((BASE / 'materials/verbatim-v1/manifest.json').read_text(encoding='utf-8'))
    instructions = []
    for index, row in enumerate(manifest['documents']):
        destination = workspace.work / row['logical_path']
        if args.condition == 'task_only':
            destination.unlink()
            continue
        source = BASE / 'materials/verbatim-v1' / row['original']
        if args.condition == 'formatted':
            source = BASE / 'materials/format-v1' / f'd{index:02d}.md'
        elif args.condition == 'phase4' and index == 0:
            source = BASE / 'materials/phase4-draft-v1/root-draft.md'
        data = source.read_bytes()
        ranges = None
        if args.condition == 'source_core':
            lines = data.splitlines(keepends=True)
            ranges = {0: [(1, 519)], 1: [(1, 137), (165, 432)], 2: [(1, len(lines))], 3: [(1, len(lines))]}[index]
            data = b''.join(b''.join(lines[start-1:end]) for start, end in ranges)
        destination.write_bytes(data)
        instructions.append({'logical_path': row['logical_path'], 'sha256': hashlib.sha256(data).hexdigest(),
                             'source_file_sha256': sha(source), 'selected_source_ranges': ranges,
                             'content': data.decode('utf-8')})
    task = f'''Optimize the assigned {args.task} application on one RTX 5060 8GB GPU, Windows, PyTorch 2.11.0+cu128.
The researcher-supplied sequential reference, fixed adapter and task specification are under flashrt/apps/{args.task}/. Read CONTRACT.zh.md, model.py, reference.py and adapter.py. Your workspace is {workspace.backend}. Supply a standalone backend/worker.py accepting --repository and --config, honoring exactly the supplied adapter protocol. The task is continuous feature processing, not speech-recognition quality. Weights are fixed, no downloads or installation are needed.
Use the original IR tools under flashrt/ir to express and validate state dependencies, then explore meaningful local optimizations. Multi-GPU requirements do not apply on this assigned single GPU. Root run-script/path examples map to this local app's backend; do not edit shared or reference files. The independent controller supplies module paths; candidate scripts can import {args.task}.adapter and {args.task}.model, causal_features.adapter, and flashrt.ir. Verification compares six complete requests with a CPU full-sequence oracle, including continuation, repeated input without reset, reset and a tail chunk. Benchmark runs paired reference and candidate finite offered-load sweeps with varied inputs and a 20ms chunk-gap budget. All code and measurements must concern the supplied task.
workspace.execute runs a backend Python script from repo root, with the pinned FlashRT package and task packages importable, for your IR and diagnostic scripts. workspace.verify and benchmark take the relative path of your standalone backend entry. Their evidence lives outside your writable workspace and can be read under evidence/. Before submitting a backend as independently verified/measured, invoke these tools for the recommended entry after its last code change. Your own harnesses can supplement them. No shell tool is supplied. Write notes and variants under your backend directory. Finish using workspace.finish with a Chinese report in connected prose: explain hypotheses, evidence, failures, trade-offs and the remaining unfinished requirements precisely.
This controlled observation has an external cap of {args.steps} API requests and {args.output_budget} generated tokens including reasoning. Development is limited to {development_budget} generated tokens and {args.steps - args.report_requests} API requests, reserving {args.report_budget} tokens and {args.report_requests} requests for a common final-report stage with thinking disabled. Coding is closed during that stage: only evidence inspection, verification/measurement of existing code, and report submission are available. The cap overrides unlimited-session language in supplied documents. Finish records the observed report and attempts artifact admission, not a claim that every upstream completion criterion was achieved. Report text is retained even if the artifact is rejected. Submit an honest report listing unmet applicable requirements instead of inventing completion. In the final-report stage, plain prose is also treated as a submission attempt under the same checker. Only the supplied instruction text and common task govern this observation; references to unavailable sections are background pointers, not hidden requirements. The assigned writable area is the backend directory. Independent correctness always uses the protected research reference.
The following instruction documents retain their logical directory scopes:
''' + '\n\n'.join('FILE: ' + row['logical_path'] + '\n' + row['content'] for row in instructions)
    save(run / 'task-and-materials.json', {'task': task, 'documents': [{k: v for k, v in row.items() if k != 'content'} for row in instructions]})
    protected = {p.relative_to(workspace.work).as_posix(): sha(p) for p in workspace.work.rglob('*') if p.is_file() and not p.relative_to(workspace.work).as_posix().startswith(workspace.backend)}
    save(run / 'protected-inventory.json', protected)
    auth = json.loads((Path.home() / '.local/share/opencode/auth.json').read_text(encoding='utf-8'))
    headers = {'Authorization': 'Bearer ' + auth['deepseek']['key']}
    messages = [{'role': 'system', 'content': 'You are a coding agent conducting the assigned streaming-backend optimization using workspace tools.'},
                {'role': 'user', 'content': task}]
    save(run / 'config.json', {'purpose': args.purpose, 'condition': args.condition,
                              'task': args.task, 'report_budget': args.report_budget, 'report_requests': args.report_requests,
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
        phase, report_start, report_start_tokens = 'development', None, None
        development_end = None
        for index in range(args.steps):
            generated = sum(row['completion_tokens'] for row in usage)
            if phase == 'development' and (development_end or generated >= development_budget - 256
                                           or index >= args.steps - args.report_requests):
                development_end = development_end or ('development_token_budget' if generated >= development_budget - 256 else 'development_request_budget')
                if not args.report_budget:
                    state = development_end
                    break
                phase, report_start, report_start_tokens = 'report', index, generated
                state = 'report_request_budget'
                messages.append({'role': 'user', 'content':
                    f'Development observation has ended. You now have at most {args.report_budget} generated tokens and '
                    f'{args.report_requests} API requests, with thinking disabled, to submit the observed result. '
                    'Do not add new variants. Report actual evidence, failures and unmet applicable requirements in connected Chinese prose. '
                    'Use existing verification/benchmark tools if needed and workspace.finish to submit. Plain prose without a tool call '
                    'is also a submission attempt, subject to the same checker. Do not invent results or claim full completion.'})
                event({'type': 'phase_transition', 'phase': phase, 'index': index, 'development_end': development_end,
                       'generated_tokens_before_report': generated})
            phase_limit = development_budget if phase == 'development' else min(args.output_budget, report_start_tokens + args.report_budget)
            remaining = phase_limit - generated
            if remaining < (256 if phase == 'development' else 1):
                state = 'report_token_budget' if phase == 'report' else 'output_budget'
                break
            if phase == 'report' and index - report_start >= args.report_requests:
                break
            body = {'model': 'deepseek-flash', 'messages': messages, 'tools': [TOOL], 'max_tokens': min(args.max_response_tokens, remaining),
                    'temperature': 1.0, 'thinking': {'type': 'enabled' if phase == 'development' else 'disabled'},
                    'reasoning_effort': 'high' if phase == 'development' else 'none'}
            phase_requests_left = min(args.steps - index, (args.steps - args.report_requests - index)
                                      if phase == 'development' else (args.report_requests - (index - report_start)))
            report_operations = ['list', 'read', 'verify', 'benchmark', 'finish']
            if phase == 'report':
                report_tool = json.loads(json.dumps(TOOL))
                report_tool['function']['description'] = 'Final report stage: inspect evidence, verify/measure an existing backend, or submit report. No code changes.'
                report_tool['function']['parameters']['properties']['operation']['enum'] = report_operations if phase_requests_left > 1 else ['finish']
                body['tools'] = [report_tool]
            if phase == 'report' and phase_requests_left == 1:
                messages.append({'role': 'user', 'content': 'This is the last report request. Submit the observed report now '
                                 'using finish or plain prose, including failures and unfinished items. Do not request further diagnostics.'})
            save(run / f'request-{index:03d}.json', body)
            print(f'API request {index + 1}, phase {phase}, remaining phase tokens {remaining}', flush=True)
            api_started = time.monotonic()
            try:
                response = requests.post('https://api.deepseek.com/chat/completions', headers=headers, json=body, timeout=(20, 600))
                data = response.json()
                save(run / f'response-{index:03d}.json', data)
                response.raise_for_status()
                metered_usage = dict(data['usage'])
                details = metered_usage.get('completion_tokens_details') or {}
                inferred_report_zero = (phase == 'report' and details.get('reasoning_tokens') is None
                                        and not data['choices'][0]['message'].get('reasoning_content'))
                if inferred_report_zero:
                    metered_usage['completion_tokens_details'] = {**details, 'reasoning_tokens': 0}
                usage.append(metered_usage)
                choice = data['choices'][0]
                messages.append(choice['message'])
                event({'type': 'response', 'index': index, 'usage': data['usage'], 'model': data.get('model'),
                       'finish_reason': choice['finish_reason'], 'seconds': time.monotonic() - api_started, 'phase': phase,
                       'report_zero_reasoning_basis': 'Requested non-thinking mode; raw provider response retained' if inferred_report_zero else None})
                if choice['finish_reason'] == 'length':
                    for call in choice['message'].get('tool_calls') or []:
                        messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                         'content': json.dumps({'ok': False, 'error': 'Generation was truncated; no partial operation was executed'})})
                        event({'type': 'truncated_call_not_executed', 'index': index, 'phase': phase, 'call': call})
                    if phase == 'development':
                        development_end = 'development_generation_truncated'
                        continue
                    content = choice['message'].get('content') or ''
                    if content:
                        (workspace.control / f'truncated-report-{index:03d}.md').write_bytes(content.encode('utf-8'))
                    state = 'report_generation_truncated'
                    break
                calls = choice['message'].get('tool_calls') or []
                budget = {'phase': phase, 'remaining_api_requests': phase_requests_left - 1,
                          'overall_api_requests_left': args.steps - index - 1,
                          'remaining_generated_tokens': max(0, phase_limit - sum(row['completion_tokens'] for row in usage)),
                          'reserved_report_tokens': args.report_budget if phase == 'development' else 0}
                if not calls:
                    content = choice['message'].get('content') or ''
                    if phase == 'report' and content.strip():
                        arguments = {'operation': 'finish', 'report': content}
                        try:
                            result = workspace.execute(arguments)
                        except (ValueError, OSError, KeyError, TypeError) as error:
                            result = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
                        event({'type': 'tool', 'index': index, 'phase': phase, 'origin': 'controller_from_plain_report',
                               'call': {'function': {'name': 'workspace', 'arguments': json.dumps(arguments, ensure_ascii=False)}},
                               'result': result})
                        messages.append({'role': 'user', 'content': 'Submission result: ' + json.dumps(result, ensure_ascii=False)
                                         + '\nRemaining report budget: ' + json.dumps(budget)})
                    else:
                        messages.append({'role': 'user', 'content': 'Continue with workspace tools or submit your report using finish. '
                                         + json.dumps(budget)})
                for call in calls:
                    arguments = {}
                    try:
                        if call['function']['name'] != 'workspace':
                            raise ValueError('Unknown tool')
                        parsed_arguments = json.loads(call['function']['arguments'])
                        if not isinstance(parsed_arguments, dict):
                            raise ValueError('Workspace arguments must be an object')
                        arguments = parsed_arguments
                        if phase == 'report' and arguments.get('operation') not in (report_operations if phase_requests_left > 1 else ['finish']):
                            raise ValueError('Development is closed; submit the report or inspect existing evidence')
                        result = workspace.execute(arguments)
                    except (ValueError, OSError, KeyError, TypeError) as error:
                        result = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
                    result['observation_budget'] = budget
                    event({'type': 'tool', 'index': index, 'phase': phase, 'call': call, 'result': result})
                    messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result, ensure_ascii=False)})
                    print(f"tool {arguments.get('operation', '?')}: {str(result)[:180]}", flush=True)
                    if workspace.finished:
                        break
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
                 if p.is_file() and '__pycache__' not in p.parts and not p.relative_to(workspace.work).as_posix().startswith(workspace.backend)
                 and p.relative_to(workspace.work).as_posix() not in protected]
        result = {'state': state, 'seconds': time.monotonic() - started, 'requests': len(usage),
                  'development_end_state': development_end, 'report_stage_entered': report_start is not None,
                  'submission_attempts': workspace.submission_attempts,
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
