"""Cross-process GPU exclusion for serial correctness and timing operations."""
from contextlib import contextmanager
import os
from pathlib import Path
import time


@contextmanager
def gpu_lock(path: Path, timeout: float = 1800):
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        while True:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("GPU operation is already held by another process")
                time.sleep(0.05)
        try:
            yield time.monotonic() - started
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
