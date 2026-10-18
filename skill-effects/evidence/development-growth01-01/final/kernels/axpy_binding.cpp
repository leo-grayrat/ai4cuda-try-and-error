// PyTorch binding for the fused alpha*a + b kernel.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void axpy_launcher(float* out, const float* a, const float* b,
                              int n, float alpha, int config, cudaStream_t stream);

torch::Tensor axpy_forward(torch::Tensor a, torch::Tensor b, double alpha, int64_t config) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(a.numel() == b.numel(), "Inputs must have the same numel");

    auto out = torch::empty_like(a);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    axpy_launcher(out.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(),
                  static_cast<int>(a.numel()), static_cast<float>(alpha),
                  static_cast<int>(config), stream);
    return out;
}

void register_axpy(pybind11::module& m) {
    m.def("axpy_forward", &axpy_forward, "Fused alpha*a + b",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("alpha"),
          pybind11::arg("config") = 0);
}

REGISTER_BINDING(axpy, register_axpy);
