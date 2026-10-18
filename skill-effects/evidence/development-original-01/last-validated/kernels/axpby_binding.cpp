#include <torch/types.h>
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void axpby_float_launcher(float* out, const float* a, const float* b,
                                     long long n, float alpha, int nblocks,
                                     int nthreads, cudaStream_t stream);

static torch::Tensor axpby_forward(const torch::Tensor& a, const torch::Tensor& b,
                                   double alpha, int64_t nblocks, int64_t nthreads) {
    auto out = at::empty_like(a);
    const long long n = (long long)a.numel();
    if (n == 0) return out;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    axpby_float_launcher(out.data_ptr<float>(), a.data_ptr<float>(),
                         b.data_ptr<float>(), n, (float)alpha, (int)nblocks,
                         (int)nthreads, stream);
    return out;
}

void register_axpby(pybind11::module& m) {
    m.def("axpby", &axpby_forward, "out = alpha * a + b", pybind11::arg("a"),
          pybind11::arg("b"), pybind11::arg("alpha"), pybind11::arg("nblocks") = 0,
          pybind11::arg("nthreads") = 0);
}

REGISTER_BINDING(axpby, register_axpby);
