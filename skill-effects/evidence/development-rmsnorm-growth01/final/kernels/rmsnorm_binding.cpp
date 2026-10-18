// PyTorch binding for the RMSNorm CUDA kernel.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "../binding_registry.h"

extern "C" void rmsnorm_launcher(const float* x,
                                 float* y,
                                 int B,
                                 int F,
                                 int N,
                                 float eps,
                                 int config,
                                 cudaStream_t stream);

torch::Tensor rmsnorm_forward(torch::Tensor input, double eps, int64_t config) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dims");

    const int B = static_cast<int>(input.size(0));
    const int F = static_cast<int>(input.size(1));
    const long long total = static_cast<long long>(input.numel());
    const int N = static_cast<int>(total / (static_cast<long long>(B) * F));

    auto output = torch::empty_like(input);
    if (total == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    rmsnorm_launcher(input.data_ptr<float>(),
                     output.data_ptr<float>(),
                     B,
                     F,
                     N,
                     static_cast<float>(eps),
                     static_cast<int>(config),
                     stream);
    return output;
}

void register_rmsnorm(pybind11::module& m) {
    m.def("rmsnorm_forward",
          &rmsnorm_forward,
          "RMSNorm over dim=1 (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("eps") = 1e-5,
          pybind11::arg("config") = -1);
}

REGISTER_BINDING(rmsnorm, register_rmsnorm);
