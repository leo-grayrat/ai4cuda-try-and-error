#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void axpby_float_launcher(float* out, const float* a, const float* b,
                                     long long n, float alpha, int nblocks,
                                     int nthreads, cudaStream_t stream);

// Lean hot path: no dispatcher for allocation, no dtype dispatch, no defaults.
static at::Tensor axpby_forward(const at::Tensor& a, const at::Tensor& b,
                                const double alpha) {
    at::Tensor out(
        at::detail::empty_cuda(a.sizes(), at::kFloat, a.device(), std::nullopt));
    axpby_float_launcher(static_cast<float*>(out.data_ptr()),
                         static_cast<const float*>(a.data_ptr()),
                         static_cast<const float*>(b.data_ptr()),
                         static_cast<long long>(a.numel()),
                         static_cast<float>(alpha), 0, 0,
                         c10::cuda::getCurrentCUDAStream());
    return out;
}

void register_axpby(pybind11::module& m) {
    m.def("axpby", &axpby_forward, "out = alpha * a + b");
}

REGISTER_BINDING(axpby, register_axpby);
