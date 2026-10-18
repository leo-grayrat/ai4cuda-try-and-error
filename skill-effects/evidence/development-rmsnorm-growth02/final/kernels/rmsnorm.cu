// ---------------------------------------------------------------------------
// Fused RMSNorm over dim = 1.
//
// The reference model computes, for an input of shape (B, F, *rest) laid out
// contiguously,
//        rms[b, m] = sqrt(mean_f x[b, f, m]^2 + eps)
//        y[b, f, m] = x[b, f, m] / rms[b, m]
// with  m = flattened trailing axes (M = numel / (B * F)).
//
// Everything is done by ONE kernel launch: the partial sum of squares, the
// reciprocal RMS and the scaling are fused, and the values that are needed a
// second time (the scaling pass) are kept by the thread that loaded them.
// No intermediate tensors are materialised, so the kernel moves the minimum
// possible amount of data: one read of the input plus one write of the output.
//
// Two implementation details matter for performance:
//   * every access is a 16-byte float4 access, fully coalesced along the
//     non-reduced axis M.  The four lanes of a float4 belong to four different
//     reduction rows, therefore four *independent* accumulators are kept (one
//     per lane) and four reciprocal RMS values are produced.
//   * the reduced axis F is split over the y dimension of the block (FS thread
//     rows); the FS per-row partial sums are combined through shared memory.
//     This raises the number of resident threads by a factor FS, which is what
//     makes such a small reduction fast.
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

#ifndef RMSNORM_CU
#define RMSNORM_CU

// ---------------------------------------------------------------- helpers ---
__device__ __forceinline__ float inv_rms(float sumsq, float invF, float eps) {
  return rsqrtf(fmaf(sumsq, invF, eps));
}

__device__ __forceinline__ float4 inv_rms4(float4 sumsq, float invF, float eps) {
  float4 r;
  r.x = inv_rms(sumsq.x, invF, eps);
  r.y = inv_rms(sumsq.y, invF, eps);
  r.z = inv_rms(sumsq.z, invF, eps);
  r.w = inv_rms(sumsq.w, invF, eps);
  return r;
}

__device__ __forceinline__ float4 scale4(float4 v, float4 s) {
  v.x *= s.x;
  v.y *= s.y;
  v.z *= s.z;
  v.w *= s.w;
  return v;
}

__device__ __forceinline__ void acc4(float4 &a, const float4 v) {
  a.x += v.x * v.x;
  a.y += v.y * v.y;
  a.z += v.z * v.z;
  a.w += v.w * v.w;
}

// Combine the FS per-lane partial sums produced by the FS thread rows.
template <int BM4, int FS>
__device__ __forceinline__ float4 reduce4(float4 acc, float4 *red, int tx,
                                          int ty) {
  red[ty * BM4 + tx] = acc;
  __syncthreads();
  float4 s = red[tx];
#pragma unroll
  for (int i = 1; i < FS; ++i) {
    float4 t = red[i * BM4 + tx];
    s.x += t.x;
    s.y += t.y;
    s.z += t.z;
    s.w += t.w;
  }
  return s;
}

// ------------------------------------ single-pass register-resident, float4 --
// Fast path: F is a compile-time constant and F % FS == 0, so the F/FS float4
// values handled by a thread stay in registers between the two passes.
template <int BM4, int FS, int F>
__global__ void k_vec4_reg(const float4 *__restrict__ in,
                           float4 *__restrict__ out, int M4, float invF,
                           float eps) {
  constexpr int NC = F / FS;  // values staged per thread
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int m4 = blockIdx.x * BM4 + tx;
  const int b = blockIdx.y;
  __shared__ float4 red[BM4 * FS];

  const long long base = (long long)b * (long long)F * (long long)M4;
  const float4 *p = in + base + m4;
  const bool valid = (m4 < M4);
  float4 vals[NC];
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int i = 0; i < NC; ++i) {
    const long long off = (long long)(ty + i * FS) * M4;
    vals[i] = valid ? __ldg(p + off) : make_float4(0.f, 0.f, 0.f, 0.f);
    acc4(acc, vals[i]);
  }
  const float4 inv4 = inv_rms4(reduce4<BM4, FS>(acc, red, tx, ty), invF, eps);

  if (valid) {
    float4 *q = out + base + m4;
#pragma unroll
    for (int i = 0; i < NC; ++i) {
      q[(long long)(ty + i * FS) * M4] = scale4(vals[i], inv4);
    }
  }
}

// ------------------------------------- single-pass shared-memory, float4 ----
// Generic vectorised path for any F: the (F x BM4) slab handled by a block is
// staged in shared memory so the input is still read only once.
template <int BM4, int FS>
__global__ void k_vec4_smem(const float4 *__restrict__ in,
                            float4 *__restrict__ out, int M4, int F, float invF,
                            float eps) {
  extern __shared__ float4 tile[];  // F * BM4 float4
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int m4 = blockIdx.x * BM4 + tx;
  const int b = blockIdx.y;
  __shared__ float4 red[BM4 * FS];

  const long long base = (long long)b * (long long)F * (long long)M4;
  const float4 *p = in + base + m4;
  const bool valid = (m4 < M4);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int f = ty; f < F; f += FS) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid) {
      v = __ldg(p + (long long)f * M4);
      acc4(acc, v);
    }
    tile[f * BM4 + tx] = v;
  }
  const float4 inv4 = inv_rms4(reduce4<BM4, FS>(acc, red, tx, ty), invF, eps);

  if (valid) {
    float4 *q = out + base + m4;
    for (int f = ty; f < F; f += FS) {
      q[(long long)f * M4] = scale4(tile[f * BM4 + tx], inv4);
    }
  }
}

// ------------------------------------------- generic two-pass, scalar loads --
// Used when the M stride is not a multiple of 4 (no float4 access possible) or
// when the tile of the shared-memory path would not fit.  Reads the input
// twice; the second read is served by cache.
template <int BM, int FS>
__global__ void k_scalar_two_pass(const float *__restrict__ in,
                                  float *__restrict__ out, int M, int F,
                                  float invF, float eps) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int m = blockIdx.x * BM + tx;
  const int b = blockIdx.y;
  __shared__ float red[BM * FS];

  const long long base = (long long)b * (long long)F * (long long)M;
  const float *p = in + base + m;
  const bool valid = (m < M);
  float acc = 0.f;
  if (valid) {
    for (int f = ty; f < F; f += FS) {
      float v = __ldg(p + (long long)f * M);
      acc += v * v;
    }
  }
  red[ty * BM + tx] = acc;
  __syncthreads();
  float s = red[tx];
#pragma unroll
  for (int i = 1; i < FS; ++i) s += red[i * BM + tx];
  const float inv = inv_rms(s, invF, eps);

  if (valid) {
    float *q = out + base + m;
    for (int f = ty; f < F; f += FS) {
      q[(long long)f * M] = __ldg(p + (long long)f * M) * inv;
    }
  }
}

// --------------------------------------------------------------- launcher ---
static inline dim3 mk_grid(int cols, int bm, int B) {
  return dim3((cols + bm - 1) / bm, B, 1);
}

extern "C" void rmsnorm_launch(const float *in, float *out, int B, int F, int M,
                               float invF, float eps, cudaStream_t stream) {
  const float4 *in4 = reinterpret_cast<const float4 *>(in);
  float4 *out4 = reinterpret_cast<float4 *>(out);
  const int M4 = M >> 2;

  if ((M & 3) == 0 && M4 >= 8) {
    if (F == 64) {  // register-resident fast path
      k_vec4_reg<32, 8, 64><<<mk_grid(M4, 32, B), dim3(32, 8, 1), 0, stream>>>(
          in4, out4, M4, invF, eps);
      return;
    }
    const size_t tile_bytes = (size_t)F * 32u * sizeof(float4);
    if (tile_bytes <= 48u * 1024u) {
      k_vec4_smem<32, 4>
          <<<mk_grid(M4, 32, B), dim3(32, 4, 1), tile_bytes, stream>>>(
              in4, out4, M4, F, invF, eps);
      return;
    }
  }
  k_scalar_two_pass<128, 4>
      <<<mk_grid(M, 128, B), dim3(128, 4, 1), 0, stream>>>(in, out, M, F, invF,
                                                           eps);
}

#endif  // RMSNORM_CU
