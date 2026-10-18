"""Execute permission/checker fixtures and verify cuBLAS linking on Windows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from agent_trial import Executor, ROOT, save
from gpu_lock import gpu_lock

GEMM_BINDING = r'''
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include "../binding_registry.h"
#ifdef _MSC_VER
#pragma comment(lib, "cublas.lib")
#endif

torch::Tensor probe_gemm(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA inputs required");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "contiguous inputs required");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "float32 required");
    int m = a.size(0), k = a.size(1), n = b.size(1);
    TORCH_CHECK(k == b.size(0), "incompatible GEMM shapes");
    auto out = torch::empty({m, n}, a.options());
    cublasHandle_t handle;
    TORCH_CHECK(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
    cublasSetStream(handle, c10::cuda::getCurrentCUDAStream().stream());
    float alpha = 1.0f, beta = 0.0f;
    auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, b.data_ptr<float>(), n, a.data_ptr<float>(), k,
                            &beta, out.data_ptr<float>(), n);
    cublasDestroy(handle);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSgemm failed");
    return out;
}
void register_probe_gemm(pybind11::module& m) { m.def("probe_gemm", &probe_gemm); }
REGISTER_BINDING(probe_gemm, register_probe_gemm);
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a directory name")
    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=False)
    executor = Executor(run, permissions=True, checker=True)
    records = {}
    try:
        executor.execute({"operation": "write", "path": "model.py", "content": "invalid replacement"})
        raise AssertionError("reference write was not rejected")
    except PermissionError:
        records["reference_write_rejected"] = True
    source = ROOT / "sources/CUDA-Agent"
    executor.files.write("model_new.py", (source / "model_new.py").read_text(encoding="utf-8"))
    for path in (source / "kernels").iterdir():
        executor.files.write("kernels/" + path.name, path.read_text(encoding="utf-8"))
    executor.files.write("kernels/probe_gemm_binding.cpp", GEMM_BINDING)
    before = executor.execute({"operation": "finish", "report": "unverified"})
    assert before["finish_rejected"]
    records["unverified_finish_rejected"] = before
    for operation in ["compile", "verify", "benchmark"]:
        print("mechanism probe " + operation, flush=True)
        result = executor.execute({"operation": operation, "label": "trusted upstream CUDA fixture"})
        records[operation] = result
        save(run / "results.json", records)
        if not result["ok"]:
            print(json.dumps(result), flush=True)
            return 1
    command = [sys.executable, "-c", "import torch,cuda_extension,json; a=torch.randn(3,7,device='cuda'); b=torch.randn(7,5,device='cuda'); actual=cuda_extension.probe_gemm(a,b); torch.testing.assert_close(actual,a@b,atol=1e-5,rtol=1e-5); torch.cuda.synchronize(); print(json.dumps({'cuBLAS_GEMM_correct':True,'output_shape':list(actual.shape)}))"]
    with gpu_lock(ROOT / "runs/gpu.lock"):
        result = subprocess.run(command, cwd=executor.work, env=executor.env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    (run / "cublas.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    records["cublas"] = json.loads(result.stdout.strip().splitlines()[-1])
    # A legal candidate can submit. A later write invalidates previous evidence.
    assert executor.execute({"operation": "finish", "report": "verified"})["submitted"]
    executor.submitted = False
    text = executor.files.read("model_new.py")
    executor.files.write("model_new.py", text + "\n# final source edit\n")
    after = executor.execute({"operation": "finish", "report": "stale verification"})
    assert after["finish_rejected"] and after["validation_missing_or_stale"] == ["verify", "benchmark"]
    records["post_edit_finish_rejected"] = after
    assert not executor.integrity()
    save(run / "results.json", records)
    print(json.dumps({"permissions": "verified", "fresh_validation_gate": "verified", "cublas": records["cublas"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
