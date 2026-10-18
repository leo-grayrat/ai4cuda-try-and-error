// RMSNorm over dim=1 for a contiguous tensor of shape (B, F, N) where N = numel/(B*F).
// Pure CUDA implementation - no PyTorch dependencies in this file.
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Single-pass kernel: one thread handles one column (b, n) and caches all F
// values of that column in registers.  F is a compile-time constant so the
// unrolled loops keep the cache entirely in registers.
// ---------------------------------------------------------------------------
template <int F, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rms_single_pass_f(const float* __restrict__ x,
                                                           float* __restrict__ y,
                                                           int N,
                                                           float inv_f,
                                                           float eps) {
    const int n = blockIdx.x * BLOCK + threadIdx.x;
    if (n >= N) return;
    const size_t base = (size_t)blockIdx.y * (size_t)F * (size_t)N + (size_t)n;
    const float* __restrict__ px = x + base;
    float* __restrict__ py = y + base;

    float v[F];
    float s = 0.0f;
#pragma unroll
    for (int f = 0; f < F; ++f) {
        const float t = px[(size_t)f * (size_t)N];
        v[f] = t;
        s = fmaf(t, t, s);
    }
    const float scale = rsqrtf(fmaf(s, inv_f, eps));
#pragma unroll
    for (int f = 0; f < F; ++f) {
        py[(size_t)f * (size_t)N] = v[f] * scale;
    }
}

// ---------------------------------------------------------------------------
// Split-F single pass: SPLIT threads cooperate on one column, each caching
// F/SPLIT values, and the partial sums are combined with warp shuffles.
// Higher thread-level parallelism / lower register pressure than the
// one-thread-per-column version.  Requires F % SPLIT == 0.
// ---------------------------------------------------------------------------
template <int F, int SPLIT, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rms_split_f(const float* __restrict__ x,
                                                     float* __restrict__ y,
                                                     int N,
                                                     float inv_f,
                                                     float eps) {
    static_assert(F % SPLIT == 0, "F must be divisible by SPLIT");
    constexpr int FP = F / SPLIT;
    constexpr int COLS = BLOCK / SPLIT;
    const int tid = threadIdx.x;
    const int part = tid & (SPLIT - 1);
    const int n = blockIdx.x * COLS + (tid / SPLIT);
    const bool valid = (n < N);
    const int nc = valid ? n : (N - 1);

    const size_t base = (size_t)blockIdx.y * (size_t)F * (size_t)N + (size_t)nc;
    const float* __restrict__ px = x + base;
    float* __restrict__ py = y + base;

    const int f0 = part * FP;
    float v[FP];
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < FP; ++i) {
        const float t = px[(size_t)(f0 + i) * (size_t)N];
        v[i] = t;
        s = fmaf(t, t, s);
    }
#pragma unroll
    for (int m = 1; m < SPLIT; m <<= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, m);
    }
    const float scale = rsqrtf(fmaf(s, inv_f, eps));
    if (!valid) return;
#pragma unroll
    for (int i = 0; i < FP; ++i) {
        py[(size_t)(f0 + i) * (size_t)N] = v[i] * scale;
    }
}

// Trivial kernel used to measure launch overhead (diagnostic only).
__global__ void rms_noop(float* y) {
    if (threadIdx.x == 1024) y[0] = 1.0f;
}

// Diagnostic: plain float4 copy (upper bound on achievable memory throughput).
template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void copy_vec4(const float4* __restrict__ x,
                                                   float4* __restrict__ y,
                                                   int nvec) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < nvec) y[i] = x[i];
}

// Diagnostic: plain scalar copy.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void copy_scalar(const float* __restrict__ x,
                                                     float* __restrict__ y,
                                                     int n) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) y[i] = x[i];
}

// ---------------------------------------------------------------------------
// Two-pass kernels (generic F): read x to compute the sum of squares, then
// read x again (usually from L2) and scale.  Scalar version.
// ---------------------------------------------------------------------------
template <int BLOCK, int UNROLL>
__global__ __launch_bounds__(BLOCK) void rms_two_pass_scalar(const float* __restrict__ x,
                                                            float* __restrict__ y,
                                                            int N,
                                                            int F,
                                                            float inv_f,
                                                            float eps) {
    const int n = blockIdx.x * BLOCK + threadIdx.x;
    if (n >= N) return;
    const size_t base = (size_t)blockIdx.y * (size_t)F * (size_t)N + (size_t)n;
    const float* __restrict__ px = x + base;
    float* __restrict__ py = y + base;

    float s = 0.0f;
#pragma unroll UNROLL
    for (int f = 0; f < F; ++f) {
        const float t = px[(size_t)f * (size_t)N];
        s = fmaf(t, t, s);
    }
    const float scale = rsqrtf(fmaf(s, inv_f, eps));
#pragma unroll UNROLL
    for (int f = 0; f < F; ++f) {
        const size_t o = (size_t)f * (size_t)N;
        py[o] = px[o] * scale;
    }
}

// ---------------------------------------------------------------------------
// Two-pass kernel, vectorized with float4 along the innermost (n) axis.
// Requires N % 4 == 0 (guaranteed by the launcher).
// ---------------------------------------------------------------------------
template <int BLOCK, int UNROLL>
__global__ __launch_bounds__(BLOCK) void rms_two_pass_vec4(const float* __restrict__ x,
                                                           float* __restrict__ y,
                                                           int N,
                                                           int F,
                                                           float inv_f,
                                                           float eps) {
    const int nv = blockIdx.x * BLOCK + threadIdx.x;
    const int nvec = N >> 2;
    if (nv >= nvec) return;
    const size_t base = (size_t)blockIdx.y * (size_t)F * (size_t)N + (size_t)nv * 4;
    const float4* __restrict__ px = reinterpret_cast<const float4*>(x + base);
    float4* __restrict__ py = reinterpret_cast<float4*>(y + base);
    const int stride = N >> 2;  // in float4 units

    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll UNROLL
    for (int f = 0; f < F; ++f) {
        const float4 t = px[(size_t)f * stride];
        s.x = fmaf(t.x, t.x, s.x);
        s.y = fmaf(t.y, t.y, s.y);
        s.z = fmaf(t.z, t.z, s.z);
        s.w = fmaf(t.w, t.w, s.w);
    }
    const float4 scale = make_float4(rsqrtf(fmaf(s.x, inv_f, eps)),
                                     rsqrtf(fmaf(s.y, inv_f, eps)),
                                     rsqrtf(fmaf(s.z, inv_f, eps)),
                                     rsqrtf(fmaf(s.w, inv_f, eps)));
#pragma unroll UNROLL
    for (int f = 0; f < F; ++f) {
        const size_t o = (size_t)f * stride;
        const float4 t = px[o];
        py[o] = make_float4(t.x * scale.x, t.y * scale.y, t.z * scale.z, t.w * scale.w);
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------
static inline int div_up(int a, int b) { return (a + b - 1) / b; }

#define DISPATCH_SINGLE(BLOCK, FVAL)                                                        \
    case FVAL:                                                                              \
        rms_single_pass_f<FVAL, BLOCK><<<dim3(div_up(N, BLOCK), B), BLOCK, 0, stream>>>(     \
            x, y, N, inv_f, eps);                                                           \
        return true;

static bool launch_single_pass(const float* x, float* y, int B, int F, int N, float inv_f, float eps,
                               cudaStream_t stream) {
    switch (F) {
        DISPATCH_SINGLE(256, 1)
        DISPATCH_SINGLE(256, 2)
        DISPATCH_SINGLE(256, 3)
        DISPATCH_SINGLE(256, 4)
        DISPATCH_SINGLE(256, 5)
        DISPATCH_SINGLE(256, 6)
        DISPATCH_SINGLE(256, 7)
        DISPATCH_SINGLE(256, 8)
        DISPATCH_SINGLE(256, 9)
        DISPATCH_SINGLE(256, 10)
        DISPATCH_SINGLE(256, 11)
        DISPATCH_SINGLE(256, 12)
        DISPATCH_SINGLE(256, 13)
        DISPATCH_SINGLE(256, 14)
        DISPATCH_SINGLE(256, 15)
        DISPATCH_SINGLE(256, 16)
        DISPATCH_SINGLE(256, 17)
        DISPATCH_SINGLE(256, 18)
        DISPATCH_SINGLE(256, 19)
        DISPATCH_SINGLE(256, 20)
        DISPATCH_SINGLE(256, 21)
        DISPATCH_SINGLE(256, 22)
        DISPATCH_SINGLE(256, 23)
        DISPATCH_SINGLE(256, 24)
        DISPATCH_SINGLE(256, 25)
        DISPATCH_SINGLE(256, 26)
        DISPATCH_SINGLE(256, 27)
        DISPATCH_SINGLE(256, 28)
        DISPATCH_SINGLE(256, 29)
        DISPATCH_SINGLE(256, 30)
        DISPATCH_SINGLE(256, 31)
        DISPATCH_SINGLE(256, 32)
        DISPATCH_SINGLE(256, 48)
        DISPATCH_SINGLE(256, 64)
        default:
            return false;
    }
}

// Alternate block sizes for tuning (only the shapes that matter).
template <int BLOCK>
static bool launch_single_pass_tuned(const float* x, float* y, int B, int F, int N, float inv_f,
                                     float eps, cudaStream_t stream) {
    switch (F) {
        DISPATCH_SINGLE(BLOCK, 16)
        DISPATCH_SINGLE(BLOCK, 17)
        DISPATCH_SINGLE(BLOCK, 32)
        DISPATCH_SINGLE(BLOCK, 64)
        default:
            return false;
    }
}

#define DISPATCH_SPLIT(SPLIT, FVAL)                                                          \
    case FVAL:                                                                               \
        rms_split_f<FVAL, SPLIT, 256><<<dim3(div_up(N, 256 / SPLIT), B), 256, 0, stream>>>(    \
            x, y, N, inv_f, eps);                                                            \
        return true;

template <int SPLIT>
static bool launch_split(const float* x, float* y, int B, int F, int N, float inv_f, float eps,
                         cudaStream_t stream) {
    if ((F % SPLIT) != 0) return false;
    switch (F) {
        DISPATCH_SPLIT(SPLIT, 8)
        DISPATCH_SPLIT(SPLIT, 16)
        DISPATCH_SPLIT(SPLIT, 32)
        DISPATCH_SPLIT(SPLIT, 48)
        DISPATCH_SPLIT(SPLIT, 64)
        default:
            return false;
    }
}

extern "C" void rmsnorm_launcher(const float* x,
                                 float* y,
                                 int B,
                                 int F,
                                 int N,
                                 float eps,
                                 int config,
                                 cudaStream_t stream) {
    if (B <= 0 || F <= 0 || N <= 0) return;
    const float inv_f = 1.0f / (float)F;

    if (config < 0) {
        // Auto-tuned choice.
        config = (F <= 64) ? 4 : 1;
    }

    switch (config) {
        case 7:  // diagnostic: no kernel at all
            return;
        case 8:  // diagnostic: single trivial kernel launch
            rms_noop<<<1, 32, 0, stream>>>(y);
            return;
        case 9: {  // diagnostic: pure copy ceiling
            const int total = B * F * N;
            if ((total & 3) == 0) {
                copy_vec4<256><<<div_up(total >> 2, 256), 256, 0, stream>>>(
                    reinterpret_cast<const float4*>(x), reinterpret_cast<float4*>(y), total >> 2);
            } else {
                copy_scalar<256><<<div_up(total, 256), 256, 0, stream>>>(x, y, total);
            }
            return;
        }
        case 10: {  // single pass, 128 threads
            if (launch_single_pass_tuned<128>(x, y, B, F, N, inv_f, eps, stream)) return;
            if (launch_single_pass(x, y, B, F, N, inv_f, eps, stream)) return;
            break;
        }
        case 11: {  // single pass, 512 threads
            if (launch_single_pass_tuned<512>(x, y, B, F, N, inv_f, eps, stream)) return;
            if (launch_single_pass(x, y, B, F, N, inv_f, eps, stream)) return;
            break;
        }
        case 12: {  // single pass, 64 threads
            if (launch_single_pass_tuned<64>(x, y, B, F, N, inv_f, eps, stream)) return;
            if (launch_single_pass(x, y, B, F, N, inv_f, eps, stream)) return;
            break;
        }
        case 13: {  // split-2 single pass
            if (launch_split<2>(x, y, B, F, N, inv_f, eps, stream)) return;
            if (launch_single_pass(x, y, B, F, N, inv_f, eps, stream)) return;
            break;
        }
        case 14: {  // split-4 single pass
            if (launch_split<4>(x, y, B, F, N, inv_f, eps, stream)) return;
            if (launch_single_pass(x, y, B, F, N, inv_f, eps, stream)) return;
            break;
        }
        case 15: {  // split-8 single pass
            if (launch_split<8>(x, y, B, F, N, inv_f, eps, stream)) return;
            if (launch_single_pass(x, y, B, F, N, inv_f, eps, stream)) return;
            break;
        }
        case 4: {  // single pass, register cached, 256 threads
            if (launch_single_pass(x, y, B, F, N, inv_f, eps, stream)) return;
            break;
        }
        case 2: {  // two pass, float4, 256 threads
            if ((N & 3) == 0) {
                dim3 grid(div_up(N >> 2, 256), B);
                rms_two_pass_vec4<256, 8><<<grid, 256, 0, stream>>>(x, y, N, F, inv_f, eps);
                return;
            }
            break;
        }
        case 3: {  // two pass, scalar, 128 threads
            dim3 grid(div_up(N, 128), B);
            rms_two_pass_scalar<128, 8><<<grid, 128, 0, stream>>>(x, y, N, F, inv_f, eps);
            return;
        }
        case 5: {  // two pass, float4, 128 threads
            if ((N & 3) == 0) {
                dim3 grid(div_up(N >> 2, 128), B);
                rms_two_pass_vec4<128, 8><<<grid, 128, 0, stream>>>(x, y, N, F, inv_f, eps);
                return;
            }
            break;
        }
        default:
            break;
    }

    // Fallback / config 1: generic two pass scalar.
    dim3 grid(div_up(N, 256), B);
    rms_two_pass_scalar<256, 8><<<grid, 256, 0, stream>>>(x, y, N, F, inv_f, eps);
}
