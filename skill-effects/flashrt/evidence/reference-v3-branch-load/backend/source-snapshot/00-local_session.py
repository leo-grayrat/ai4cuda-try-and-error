"""Owned backend lifecycle for local application measurements."""
import hashlib
import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import subprocess
import sys

import psutil


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LocalSession:
    def __init__(self, repository, config, output, backend_file=None, backend_repository=None):
        self.repository = Path(repository).resolve()
        self.config, self.output = config, Path(output).resolve()
        self.backend_file = Path(backend_file).resolve() if backend_file else None
        self.backend_repository = Path(backend_repository).resolve() if backend_repository else self.repository
        self.process = self.adapter = None
        self.files = []
        self.evidence = {}

    def __enter__(self):
        from causal_features.adapter import FeatureAdapter
        self.output.mkdir(parents=True, exist_ok=False)
        config_path = self.output / 'config.json'
        config_path.write_bytes((json.dumps(self.config, indent=2) + '\n').encode())
        base = Path(__file__).resolve().parent
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(base / 'workloads') + os.pathsep + str(self.backend_repository),
                           PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-u']
        task = self.config.get('task', 'causal_features')
        command += [str(self.backend_file)] if self.backend_file else ['-m', task + '.reference']
        command += ['--repository', str(self.backend_repository if self.backend_file else self.repository), '--config', str(config_path)]
        sources = sorted({Path(__file__), *(base / 'workloads/causal_features').glob('*.py'),
                          *(base / 'workloads' / task).glob('*.py')})
        if self.backend_file:
            sources.append(self.backend_file)
        self.source_hashes = {str(path): file_hash(path) for path in sources}
        for index, path in enumerate(sources):
            target = self.output / 'source-snapshot' / f'{index:02d}-{path.name}'
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(path.read_bytes())
        try:
            stdout = (self.output / 'backend.stdout.log').open('w', encoding='utf-8')
            self.files.append(stdout)
            stderr = (self.output / 'backend.stderr.log').open('w', encoding='utf-8')
            self.files.append(stderr)
            self.process = subprocess.Popen(command, cwd=self.repository, env=environment, stdout=stdout, stderr=stderr)
            self.evidence['backend_pid'] = self.process.pid
            self.adapter = FeatureAdapter(self.config)
            self.adapter.start()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        descendants = []
        if self.process is not None and self.process.poll() is None:
            try:
                descendants = psutil.Process(self.process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                pass
        if self.adapter:
            try:
                self.adapter.terminate()
            except Exception as error:
                self.evidence['termination_error'] = str(error)
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        remaining = [child for child in descendants if child.is_running()]
        self.evidence['descendants_remaining_before_forced_cleanup'] = [child.pid for child in remaining]
        for child in remaining:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if remaining:
            psutil.wait_procs(remaining, timeout=5)
        if self.adapter:
            self.adapter.close()
        for stream in self.files:
            stream.close()
        prefix = self.config['prefix']
        names = [prefix + '__ctrl'] + [prefix + '_' + kind + suffix for kind in ['input', 'output', 'metadata']
                                       for suffix in ['__meta', '__tensor']]
        leftovers = []
        for name in names:
            try:
                handle = shared_memory.SharedMemory(name=name, create=False)
            except FileNotFoundError:
                continue
            leftovers.append(name)
            handle.unlink()
            handle.close()
        stderr_path = self.output / 'backend.stderr.log'
        stderr_text = stderr_path.read_text(encoding='utf-8') if stderr_path.exists() else ''
        known = 'Failed to find real location of ' + sys.executable
        unexpected = [line for line in stderr_text.splitlines() if line.strip() and line != known]
        changed = [path for path, expected in self.source_hashes.items() if not Path(path).exists() or file_hash(path) != expected]
        exitcode = self.process.returncode if self.process else None
        self.evidence.update(backend_exitcode=exitcode, shared_memory_leftovers=leftovers,
                             unexpected_stderr=unexpected, source_changed=changed, source_sha256=self.source_hashes,
                             cleanup_verified=exitcode == 0 and not leftovers and not remaining,
                             valid_exit=exitcode == 0 and not leftovers and not remaining and not unexpected and not changed)
