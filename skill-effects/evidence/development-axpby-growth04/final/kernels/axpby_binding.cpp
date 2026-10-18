// PROBE ONLY -- diagnostics for pool recyclability, removed before final.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>
#include <vector>
#include "../binding_registry.h"

extern "C" void axpby_launcher(float* out,
                               const float* a,
                               const float* b,
                               float alpha,
                               int n,
                               cudaStream_t stream);

namespace {

struct PoolEntry {
    torch::Tensor buf;
    c10::DeviceIndex device;
};

std::vector<PoolEntry> g_pool;
long long g_calls = 0;

inline bool recyclable(const torch::Tensor& t) {
    return t.use_count() == 1 && t.storage().use_count() == 1;
}

torch::Tensor acquire_output(const torch::Tensor& a) {
    const c10::DeviceIndex dev = a.get_device();
    if (g_calls < 12) {
        std::printf("[probe] enter call=%lld pool=%zu :", g_calls, g_pool.size());
        for (size_t i = 0; i < g_pool.size() && i < 8; ++i) {
            std::printf(" (%zu:uc=%lld,suc=%lld)", i,
                        static_cast<long long>(g_pool[i].buf.use_count()),
                        static_cast<long long>(g_pool[i].buf.storage().use_count()));
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    for (auto& e : g_pool) {
        if (e.device == dev && e.buf.sizes() == a.sizes() && recyclable(e.buf)) {
            return e.buf;
        }
    }
    torch::Tensor buf = torch::empty_like(a);
    g_pool.push_back({buf, dev});
    return buf;
}

torch::Tensor axpby(torch::Tensor a, torch::Tensor b, double alpha) {
    torch::Tensor out = acquire_output(a);
    ++g_calls;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    axpby_launcher(out.data_ptr<float>(),
                   a.data_ptr<float>(),
                   b.data_ptr<float>(),
                   static_cast<float>(alpha),
                   static_cast<int>(a.numel()),
                   stream);
    return out;
}

void register_axpby(pybind11::module& m) {
    m.def("axpby", &axpby, "AXPBY: alpha * a + b",
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("alpha"));
}

}  // namespace

REGISTER_BINDING(axpby, register_axpby);
