import io
import json
import tarfile

from scripts.index_credit_edges import index_edges


def write_archive(path):
    entries = [
        (0, None, False, "0", "root"),
        (1, 0, True, "0", "parent"),
        (2, 1, True, "0", "faster"),
        (3, 1, True, "0", "parent"),
        (4, 1, True, "1", "another_device"),
    ]
    with tarfile.open(path, "w:gz") as archive:
        for child, parent, correct, device, source in reversed(entries):
            base = f"run/2_2/step_{child}"
            objects = {
                f"{base}_log.json": {"node_id": child, "parent_node_id": parent},
                f"{base}_metrics.json": {
                    "correctness": correct, "runtime": 100 - child,
                    "metadata": {"hardware": "GPU", "device": device},
                },
            }
            for name, obj in objects.items():
                data = json.dumps(obj).encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            data = source.encode()
            info = tarfile.TarInfo(f"{base}.py")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_index_keeps_all_edges_and_exclusion_reasons_without_speed_ranking(tmp_path):
    path = tmp_path / "trace.tar.gz"
    write_archive(path)

    indexed = index_edges(path)

    assert [(row["parent_node"], row["child_node"], row["status"]) for row in indexed["rows"]] == [
        (0, 1, "root_parent"),
        (1, 2, "eligible"),
        (1, 3, "identical_source"),
        (1, 4, "different_device"),
    ]
    assert indexed["counts"]["eligible"] == 1
    assert all("runtime" not in row for row in indexed["rows"])
    assert indexed["rows"][1]["parent_code_sha256"] != indexed["rows"][1]["child_code_sha256"]
