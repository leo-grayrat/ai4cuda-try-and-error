// Fused alpha*a + b CUDA kernel (pure CUDA, no PyTorch dependencies).
//
// The workload is tiny (e.g. 128 floats), so the kernel is latency bound:
// a single warp performs one vectorized (float4) read of each input and one
// vectorized write, using FMA.  Block size is selectable at runtime ("config")
// for tuning without recompilation; config 0 (one warp) is the tuned default.
#include <cuda_runtime.h>

// ---- vectorized body: each thread handles 4 floats (float4) ----
__global__ void axpy_vec4_kernel(float4* __restrict__ out,
                                 const float4* __restrict__ a,
                                 const float4* __restrict__ b,
                                 int n4, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 va = __ldg(&a[i]);
        float4 vb = __ldg(&b[i]);
        float4 r;
        r.x = fmaf(alpha, va.x, vb.x);
        r.y = fmaf(alpha, va.y, vb.y);
        r.z = fmaf(alpha, va.z, vb.z);
        r.w = fmaf(alpha, va.w, vb.w);
        out[i] = r;
    }
}

// ---- scalar body: unaligned inputs / numel not divisible by 4 ----
__global__ void axpy_scalar_kernel(float* __restrict__ out,
                                   const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fmaf(alpha, __ldg(&a[i]), __ldg(&b[i]));
    }
}

extern "C" void axpy_launcher(float* out, const float* a, const float* b,
                              int n, float alpha, int config, cudaStream_t stream) {
    if (n <= 0) return;

    const bool aligned =
        ((reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(a) |
          reinterpret_cast<uintptr_t>(b)) & 15u) == 0u;

    if (aligned && (n & 3) == 0) {
        const int n4 = n >> 2;
        int threads;
        switch (config) {
            case 1:  threads = 128; break;
            case 2:  threads = 256; break;
            default: threads = 32;  break;  // one warp: minimal launch for tiny sizes
        }
        int blocks = (n4 + threads - 1) / threads;
        if (blocks < 1) blocks = 1;
        axpy_vec4_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4*>(out), reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b), n4, alpha);
        return;
    }

    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    axpy_scalar_kernel<<<blocks, threads, 0, stream>>>(out, a, b, n, alpha);
}
