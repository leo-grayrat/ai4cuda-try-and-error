// Fused elementwise kernel: out = alpha * a + b
#include <cuda_runtime.h>

__global__ void axpby_vec4_kernel(float4* __restrict__ out,
                                  const float4* __restrict__ a,
                                  const float4* __restrict__ b,
                                  int n4,
                                  int tail_offset,
                                  int n,
                                  float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 av = a[i];
        float4 bv = b[i];
        float4 r;
        r.x = fmaf(alpha, av.x, bv.x);
        r.y = fmaf(alpha, av.y, bv.y);
        r.z = fmaf(alpha, av.z, bv.z);
        r.w = fmaf(alpha, av.w, bv.w);
        out[i] = r;
    }
    if (tail_offset >= 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        const float* af = reinterpret_cast<const float*>(a);
        const float* bf = reinterpret_cast<const float*>(b);
        float* of = reinterpret_cast<float*>(out);
        for (int j = tail_offset; j < n; ++j) {
            of[j] = fmaf(alpha, af[j], bf[j]);
        }
    }
}

__global__ void axpby_scalar_float_kernel(float* __restrict__ out,
                                          const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          long long n,
                                          float alpha) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        out[i] = alpha * a[i] + b[i];
    }
}

extern "C" void axpby_float_launcher(float* out,
                                     const float* a,
                                     const float* b,
                                     long long n,
                                     float alpha,
                                     int nblocks,
                                     int nthreads,
                                     cudaStream_t stream) {
    bool aligned = ((reinterpret_cast<size_t>(out) | reinterpret_cast<size_t>(a) |
                     reinterpret_cast<size_t>(b)) & 15u) == 0u;
    if (aligned && n >= 4) {
        int n4 = (int)(n >> 2);
        int tail_offset = (int)(n4 << 2);
        int threads = nthreads > 0 ? (int)nthreads : 32;
        if (n4 < threads) threads = n4;
        int blocks = nblocks > 0 ? (int)nblocks : ((n4 + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        axpby_vec4_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4*>(out), reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b), n4, tail_offset, (int)n, alpha);
    } else {
        const int threads = 128;
        long long blocks = (n + threads - 1) / threads;
        if (blocks < 1) blocks = 1;
        if (blocks > 65535) blocks = 65535;
        axpby_scalar_float_kernel<<<(int)blocks, threads, 0, stream>>>(out, a, b, n,
                                                                      alpha);
    }
}
