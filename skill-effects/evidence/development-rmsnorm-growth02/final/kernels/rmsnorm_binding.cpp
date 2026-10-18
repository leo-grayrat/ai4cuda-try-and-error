// Binding for the fused RMSNorm kernel.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "../binding_registry.h"

extern "C" void rmsnorm_launch(const float *in, float *out, int B, int F, int M,
                               float invF, float eps, cudaStream_t stream);

torch::Tensor rmsnorm_forward(torch::Tensor x, double eps) {
  TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
  TORCH_CHECK(x.dim() >= 2, "Input must have at least 2 dimensions");

  auto out = torch::empty_like(x);
  if (x.numel() == 0) return out;

  const int64_t B = x.size(0);
  const int64_t F = x.size(1);
  const int64_t M = x.numel() / (B * F);
  const float invF = 1.0f / static_cast<float>(F);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  rmsnorm_launch(x.data_ptr<float>(), out.data_ptr<float>(),
                 static_cast<int>(B), static_cast<int>(F), static_cast<int>(M),
                 invF, static_cast<float>(eps), stream);
  return out;
}

void register_rmsnorm(pybind11::module &m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "Fused RMSNorm over dim=1",
        pybind11::arg("x"), pybind11::arg("eps") = 1e-5);
}

REGISTER_BINDING(rmsnorm, register_rmsnorm);
