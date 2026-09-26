import io
import json
import tarfile

import pytest

from scripts.audit_adaexplore import audit


def make_archive(path, invalid_parent=False):
    prefix = "run/2_1/"
    with tarfile.open(path, "w:gz") as archive:
        for index in range(4):
            log = {"node_id": index, "parent_node_id": None if index == 0 else index - 1}
            if invalid_parent and index == 3:
                log["parent_node_id"] = 99
            metrics = {"correctness": index > 0, "runtime": 10 / index if index else -1,
                       "metadata": {"hardware": "GPU", "device": "0"}}
            for name, data in [(f"step_{index}_log.json", json.dumps(log).encode()),
                               (f"step_{index}_metrics.json", json.dumps(metrics).encode()),
                               (f"step_{index}.py", f"code {index}\n".encode())]:
                info = tarfile.TarInfo(prefix + name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))


def test_archive_audit_checks_parent_edges_and_code_changes(tmp_path):
    path = tmp_path / "trace.tar.gz"
    make_archive(path)
    counts = audit(path)["counts"]
    assert counts["candidate_nodes"] == 3
    assert counts["explicit_parent_edges"] == 3
    assert counts["same_device_edges"] == 2
    assert counts["two_step_chains_with_two_code_changes"] == 1


def test_archive_audit_rejects_missing_parent(tmp_path):
    path = tmp_path / "trace.tar.gz"
    make_archive(path, invalid_parent=True)
    with pytest.raises(ValueError, match="invalid parent"):
        audit(path)
