"""Natural completion with researcher-only ceilings and no budget messages.

Shares the fixed task, source materials and independent evaluators with the
earlier observations. The common user request and stopping protocol differ.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import requests

from pilot_trial import BASE, TOOL, Workspace, save, sha
from runtime import summarize_usage

NATURAL_RATES = [50, 100, 200, 500, 1000, 2000]


class NaturalWorkspace(Workspace):
    def process(self, command, folder, timeout=180, lock=False):
        # Waiting for another trial's GPU must not consume the short parent
        # measurement deadline. Candidate scripts retain a separate run cap.
        if str(BASE / 'benchmark_local.py') in command:
            timeout = 2100
            command = list(command)
            command[command.index('--rates') + 1] = ','.join(map(str, NATURAL_RATES))
        return super().process(command, folder, timeout=timeout, lock=lock)

    def execute(self, arguments):
        if arguments.get('operation') == 'execute':
            path = arguments.get('path', '')
            if not isinstance(path, str):
                raise ValueError('path must be a relative path string')
            target = self.files.path(path)
            if not path.startswith(self.backend) or target.suffix != '.py' or not target.is_file():
                raise ValueError('Execute an existing Python script under the backend directory')
            extra = arguments.get('args', [])
            if not isinstance(extra, list) or not all(isinstance(value, str) for value in extra):
                raise ValueError('args must be an array of strings')
            self.counter += 1
            return self.process([sys.executable, str(target), *extra],
                                self.control / f'execute-{self.counter:03d}', timeout=600, lock=True)
        return super().execute(arguments)

    def benchmark(self, path):
        result = super().benchmark(path)
        paired = {}
        for name, outcome in result.get('paired_measurements', {}).items():
            raw_path = self.control / f'benchmark-{self.counter:03d}' / name / 'result.json'
            raw = json.loads(raw_path.read_text(encoding='utf-8')) if raw_path.exists() else {}
            loads = []
            for load in raw.get('loads', []):
                requests = load['requests']
                within = [(b-a)/1e6 for request in requests
                          for a, b in zip(request['output_ns'], request['output_ns'][1:])]
                boundary = [(b['output_ns'][0]-a['output_ns'][-1])/1e6
                            for a, b in zip(requests, requests[1:])]
                def describe(values):
                    return {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
                            'max': float(max(values))} if values else None
                loads.append({'offered_requests_per_second': load['offered_requests_per_second'],
                              'completed_requests_per_second': load['completed_requests_per_second'],
                              'first_output_ms': load['first_output_ms'],
                              'within_request_output_gap_ms': describe(within),
                              'request_boundary_gap_ms': describe(boundary),
                              'peak_external_queue_requests': load['peak_external_queue_requests'],
                              'offered_rate_kept_up': load['rate_gate_passed']})
            kept = [row['offered_requests_per_second'] for row in loads if row['offered_rate_kept_up']]
            paired[name] = {'status': outcome.get('status'), 'execution': outcome.get('execution'),
                            'highest_tested_rate_without_backlog': max(kept) if kept else None, 'loads': loads}
        return {'ok': result['ok'], 'paired_measurements': paired,
                'timing_definition': 'Within-request gaps describe chunk production. Request-boundary gaps also include waiting for scheduled new input; they are reported separately.'}


def instructions(workspace, condition):
    manifest = json.loads((BASE / 'materials/verbatim-v1/manifest.json').read_text(encoding='utf-8'))
    rows = []
    for index, row in enumerate(manifest['documents']):
        destination = workspace.work / row['logical_path']
        if condition == 'task_only':
            destination.unlink()
            continue
        source = BASE / 'materials/verbatim-v1' / row['original']
        if condition == 'formatted':
            source = BASE / 'materials/format-v1' / f'd{index:02d}.md'
        elif condition == 'phase4' and index == 0:
            source = BASE / 'materials/phase4-draft-v1/root-draft.md'
        data = source.read_bytes()
        ranges = None
        if condition == 'source_core':
            lines = data.splitlines(keepends=True)
            ranges = {0: [(1, 519)], 1: [(1, 137), (165, 432)],
                      2: [(1, len(lines))], 3: [(1, len(lines))]}[index]
            data = b''.join(b''.join(lines[start-1:end]) for start, end in ranges)
        destination.write_bytes(data)
        rows.append({'logical_path': row['logical_path'], 'sha256': sha(destination),
                     'selected_source_ranges': ranges, 'initially_in_context': index == 0})
    root = workspace.work / 'AGENTS.md'
    return rows, root.read_text(encoding='utf-8') if root.exists() else ''


def common_request(task):
    return f'''请优化 flashrt/apps/{task}/ 里的特征处理应用，让它在本机 RTX 5060 8GB、Windows、PyTorch 2.11.0+cu128 上处理更快，兼顾首块输出等待和连续处理速度。保持参考实现的计算结果、跨请求状态、重置行为及现有输入输出接口兼容。
参考实现、接口约定和固定适配器都在该目录中。权重由代码中的固定种子构建，无需下载或安装。实现放在该应用的 backend/ 下，入口 worker.py 接受 --repository 和 --config；共享文件保持只读。请按仓库中适用的说明工作。完成后向我说明改动、验证结果、性能取舍及尚未解决的问题。'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', choices=['task_only', 'source_core', 'original', 'formatted', 'phase4'], required=True)
    parser.add_argument('--task', choices=['causal_features', 'branch_features'], required=True)
    parser.add_argument('--permissions', action='store_true')
    parser.add_argument('--checker', action='store_true')
    parser.add_argument('--external-output-ceiling', type=int, default=320000)
    parser.add_argument('--external-request-ceiling', type=int, default=128)
    args = parser.parse_args()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    workspace = NaturalWorkspace(run, args.repository.resolve(), args.permissions, args.checker, args.task)
    documents, root = instructions(workspace, args.condition)
    request = common_request(args.task)
    user_text = request + ('\n\n项目根目录 AGENTS.md：\n' + root if root else '')
    messages = [{'role': 'system', 'content': 'You are a coding agent working in the provided repository with workspace tools.'},
                {'role': 'user', 'content': user_text}]
    tool = json.loads(json.dumps(TOOL))
    tool['function']['description'] = 'Read and edit workspace files, execute Python scripts, independently verify or benchmark an application. You may finish with a normal final answer or use finish.'
    tool['function']['parameters']['properties']['args'] = {'type': 'array', 'items': {'type': 'string'},
        'description': 'Optional command-line arguments for an existing Python script passed to execute.'}
    assert all(marker not in request for marker in ['token', '调用次数', '报告阶段', '预算'])
    save(run / 'task-and-materials.json', {'task': user_text, 'common_request': request, 'documents': documents,
                                         'directory_loading': 'Root initially supplied; nested documents available at original paths'})
    save(run / 'config.json', {'purpose': 'natural_work', 'condition': args.condition, 'task': args.task,
        'permissions': args.permissions, 'checker': args.checker,
        'steps': args.external_request_ceiling, 'output_budget': args.external_output_ceiling,
        'max_response_tokens': args.external_output_ceiling, 'report_budget': 0, 'report_requests': 0,
        'model': 'deepseek-flash', 'runner_sha256': sha(Path(__file__)),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'external_limits_shown_to_actor': False})
    (run / 'runner-snapshot.py').write_bytes(Path(__file__).read_bytes())
    protected = {p.relative_to(workspace.work).as_posix(): sha(p) for p in workspace.work.rglob('*')
                 if p.is_file() and not p.relative_to(workspace.work).as_posix().startswith(workspace.backend)}
    save(run / 'protected-inventory.json', protected)
    auth = json.loads((Path.home() / '.local/share/opencode/auth.json').read_text(encoding='utf-8'))
    headers = {'Authorization': 'Bearer ' + auth['deepseek']['key']}
    usage, state, started = [], 'external_request_ceiling', time.monotonic()
    with (run / 'events.jsonl').open('w', encoding='utf-8') as events:
        def event(row):
            events.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), **row}, ensure_ascii=False) + '\n')
            events.flush()

        for index in range(args.external_request_ceiling):
            remaining = args.external_output_ceiling - sum(row['completion_tokens'] for row in usage)
            if remaining < 1:
                state = 'external_output_ceiling'
                break
            body = {'model': 'deepseek-flash', 'messages': messages, 'tools': [tool], 'max_tokens': remaining,
                    'temperature': 1.0, 'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
            save(run / f'request-{index:03d}.json', body)
            api_started = time.monotonic()
            try:
                response = requests.post('https://api.deepseek.com/chat/completions', headers=headers,
                                         json=body, timeout=(20, 1800))
                data = response.json()
                save(run / f'response-{index:03d}.json', data)
                response.raise_for_status()
                usage.append(data['usage'])
                choice = data['choices'][0]
                messages.append(choice['message'])
                event({'type': 'response', 'index': index, 'usage': data['usage'],
                       'model': data.get('model'), 'finish_reason': choice['finish_reason'],
                       'seconds': time.monotonic()-api_started})
                print(f'Response {index+1}: {choice["finish_reason"]}', flush=True)
                if choice['finish_reason'] == 'length':
                    state = 'external_generation_truncated'
                    break
                calls = choice['message'].get('tool_calls') or []
                if not calls:
                    content = choice['message'].get('content') or ''
                    if not content.strip():
                        state = 'empty_final_response'
                        break
                    try:
                        result = workspace.execute({'operation': 'finish', 'report': content})
                    except (ValueError, OSError, KeyError, TypeError) as error:
                        result = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
                    event({'type': 'plain_final_submission', 'index': index, 'result': result})
                    if workspace.finished:
                        state = 'submitted'
                        break
                    messages.append({'role': 'user', 'content': '交付检查：' + json.dumps(result, ensure_ascii=False)})
                for call in calls:
                    try:
                        if call['function']['name'] != 'workspace':
                            raise ValueError('Unknown tool')
                        arguments = json.loads(call['function']['arguments'])
                        if not isinstance(arguments, dict):
                            raise ValueError('Workspace arguments must be an object')
                        result = workspace.execute(arguments)
                    except (ValueError, OSError, KeyError, TypeError) as error:
                        result = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
                    event({'type': 'tool', 'index': index, 'call': call, 'result': result})
                    messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                     'content': json.dumps(result, ensure_ascii=False)})
                    if workspace.finished:
                        state = 'submitted'
                        break
                save(run / 'messages.json', messages)
                if workspace.finished:
                    break
            except (requests.RequestException, ValueError, KeyError) as error:
                state = 'api_or_trace_error'
                event({'type': 'error', 'index': index, 'error': str(error)})
                break
        save(run / 'messages.json', messages)
    final = {}
    if (workspace.work / workspace.entry).is_file():
        final['correctness'] = workspace.verify(workspace.entry, hidden=True)
        if final['correctness']['ok']:
            final['benchmark'] = workspace.benchmark(workspace.entry)
    changed = [name for name, expected in protected.items()
               if not (workspace.work / name).is_file() or sha(workspace.work / name) != expected]
    added = [p.relative_to(workspace.work).as_posix() for p in workspace.work.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts
             and not p.relative_to(workspace.work).as_posix().startswith(workspace.backend)
             and p.relative_to(workspace.work).as_posix() not in protected]
    save(run / 'result.json', {'state': state, 'seconds': time.monotonic()-started, 'requests': len(usage),
         'requests_attempted': len(list(run.glob('request-*.json'))),
         'usage_records_complete': len(usage) == len(list(run.glob('request-*.json'))),
         'usage': summarize_usage(usage) if usage else None, 'protected_files_changed': changed,
         'files_added_outside_backend': added, 'report_available': (run / 'report.md').exists(),
         'submission_attempts': workspace.submission_attempts, 'final_evaluation': final,
         'externally_censored': state.startswith('external_'), 'external_limits_shown_to_actor': False})
    print(json.dumps({'state': state, 'correctness': final.get('correctness', {}).get('ok')}), flush=True)
    return 0 if state == 'submitted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
