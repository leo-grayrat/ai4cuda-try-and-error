from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_gpu_lock_excludes_other_processes_and_releases(tmp_path):
    from gpu_lock import gpu_lock
    lock = tmp_path / "gpu.lock"
    module_path = str(Path(__file__).resolve().parents[1])
    script = "import sys; sys.path.insert(0,sys.argv[1]); from gpu_lock import gpu_lock; from pathlib import Path\ntry:\n with gpu_lock(Path(sys.argv[2]),timeout=.15): pass\nexcept TimeoutError: sys.exit(3)"
    with gpu_lock(lock):
        blocked = subprocess.run([sys.executable, "-c", script, module_path, str(lock)], timeout=10)
        assert blocked.returncode == 3
    admitted = subprocess.run([sys.executable, "-c", script, module_path, str(lock)], timeout=10)
    assert admitted.returncode == 0
