
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
