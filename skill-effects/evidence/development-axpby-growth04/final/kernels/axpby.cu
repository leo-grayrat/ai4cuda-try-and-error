// AXPBY: out = alpha * a + b   (contiguous float32 CUDA tensors)
// Pure CUDA implementation: vectorized float4 main path + scalar tail/fallback.
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float4 axpby4(float alpha, float4 va, float4 vb) {
    float4 r;
    r.x = fmaf(alpha, va.x, vb.x);
    r.y = fmaf(alpha, va.y, vb.y);
    r.z = fmaf(alpha, va.z, vb.z);
    r.w = fmaf(alpha, va.w, vb.w);
    return r;
}

// Vectorized kernel: handles n4 float4 groups plus the remaining tail elements.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void axpby_vec4_kernel(float* __restrict__ out,
                                                           const float* __restrict__ a,
                                                           const float* __restrict__ b,
                                                           float alpha,
                                                           int n4,
                                                           int n) {
    const int tid = blockIdx.x * BLOCK + threadIdx.x;
    const int stride = gridDim.x * BLOCK;

    float4* __restrict__ o4 = reinterpret_cast<float4*>(out);
    const float4* __restrict__ a4 = reinterpret_cast<const float4*>(a);
    const float4* __restrict__ b4 = reinterpret_cast<const float4*>(b);

    for (int k = tid; k < n4; k += stride) {
        o4[k] = axpby4(alpha, a4[k], b4[k]);
    }

    const int tail = n4 << 2;
    for (int k = tail + tid; k < n; k += stride) {
        out[k] = fmaf(alpha, a[k], b[k]);
    }
}

// Scalar kernel: used for small / unaligned inputs.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void axpby_scalar_kernel(float* __restrict__ out,
                                                            const float* __restrict__ a,
                                                            const float* __restrict__ b,
                                                            float alpha,
                                                            int n) {
    const int stride = gridDim.x * BLOCK;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += stride) {
        out[i] = fmaf(alpha, a[i], b[i]);
    }
}

inline bool aligned16(const void* p) {
    return (reinterpret_cast<unsigned long long>(p) & 15ULL) == 0ULL;
}

inline int grid_for(int work, int block, int max_blocks) {
    int g = (work + block - 1) / block;
    if (g < 1) g = 1;
    if (g > max_blocks) g = max_blocks;
    return g;
}

}  // namespace

extern "C" void axpby_launcher(float* out,
                               const float* a,
                               const float* b,
                               float alpha,
                               int n,
                               cudaStream_t stream) {
    if (n <= 0) return;

    const bool vec_ok = (n >= 4) && aligned16(out) && aligned16(a) && aligned16(b);

    if (vec_ok) {
        const int n4 = n >> 2;
        constexpr int kBlock = 128;
        const int grid = grid_for(n4, kBlock, 65535);
        axpby_vec4_kernel<kBlock><<<grid, kBlock, 0, stream>>>(out, a, b, alpha, n4, n);
    } else {
        constexpr int kBlock = 128;
        const int grid = grid_for(n, kBlock, 65535);
        axpby_scalar_kernel<kBlock><<<grid, kBlock, 0, stream>>>(out, a, b, alpha, n);
    }
}
