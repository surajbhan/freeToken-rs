// q4_0 dequant-GEMV, the decode-path workhorse (port of the mmvq idea from
// FreeToken's borrowed ggml kernels, simplified: scalar dequant in-loop,
// f32 activations). One thread block per output row; each thread walks a
// strided subset of the row's 18-byte blocks, then a shared-memory reduce.
#include <cuda_fp16.h>

// out[i] = silu(gu[i]) * gu[i + inter]  (gate_up laid out [gate | up])
extern "C" __global__ void silu_mul(const float* __restrict__ gu,
                                    float* __restrict__ out, int inter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inter) {
        float g = gu[i];
        out[i] = (g / (1.0f + expf(-g))) * gu[i + inter];
    }
}

// y[i] += a * x[i] — expert-weighted accumulation into the layer output
extern "C" __global__ void axpy(const float* __restrict__ x,
                                float* __restrict__ y, float a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

extern "C" __global__ void gemv_q4_0(
    const unsigned char* __restrict__ w, // [n_rows, k/32*18], rows consecutive
    const float* __restrict__ x,         // [k]
    float* __restrict__ y,               // [n_rows]
    int n_rows,
    int k)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int nblocks = k / 32;
    const unsigned char* wr = w + (size_t)row * nblocks * 18;

    float acc = 0.0f;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        const unsigned char* blk = wr + b * 18;
        float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const unsigned char* qs = blk + 2;
        const float* xb = x + b * 32;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int lo = qs[i] & 0xF;
            int hi = qs[i] >> 4;
            s += (lo - 8) * xb[i] + (hi - 8) * xb[i + 16];
        }
        acc += d * s;
    }

    __shared__ float sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = sh[0];
}

// ---- grouped (one-launch-per-layer) decode path ----

// y[e][row] = dot(W_e[row], x_e) for all routed experts in ONE launch.
// W_e = base + slots[e] * expert_bytes + bank_off (a bank inside the slot
// cache); x_e = x + e * x_stride (x_stride = 0 shares one activation).
// Grid: (n_rows, n_experts).
extern "C" __global__ void gemv_q4_0_grouped(
    const unsigned char* __restrict__ base,
    unsigned long long expert_bytes,
    unsigned long long bank_off,
    const int* __restrict__ slots,
    const float* __restrict__ x,
    int x_stride,
    float* __restrict__ y,
    int y_stride,
    int n_rows,
    int k)
{
    int row = blockIdx.x;
    int e = blockIdx.y;
    if (row >= n_rows) return;
    int nblocks = k / 32;
    const unsigned char* wr = base + (size_t)slots[e] * expert_bytes + bank_off
                            + (size_t)row * nblocks * 18;
    const float* xe = x + (size_t)e * x_stride;

    float acc = 0.0f;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        const unsigned char* blk = wr + b * 18;
        float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const unsigned char* qs = blk + 2;
        const float* xb = xe + b * 32;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int lo = qs[i] & 0xF;
            int hi = qs[i] >> 4;
            s += (lo - 8) * xb[i] + (hi - 8) * xb[i + 16];
        }
        acc += d * s;
    }

    __shared__ float sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)e * y_stride + row] = sh[0];
}

// v2 of the grouped GEMV: the activation is staged in shared memory once per
// block (the naive kernel re-read it from L2 for every row), rows are
// processed one-per-warp (4 rows/block), and the 18-byte q4_0 blocks are read
// as nine u16 loads (blocks are 2-byte aligned) instead of 18 byte loads.
// Grid: (ceil(n_rows/4), n_experts), block 128, smem = k floats.
extern "C" __global__ void gemv_q4_0_grouped_v2(
    const unsigned char* __restrict__ base,
    unsigned long long expert_bytes,
    unsigned long long bank_off,
    const int* __restrict__ slots,
    const float* __restrict__ x,
    int x_stride,
    float* __restrict__ y,
    int y_stride,
    int n_rows,
    int k)
{
    extern __shared__ float xs[];
    int e = blockIdx.y;
    const float* xe = x + (size_t)e * x_stride;
    for (int i = threadIdx.x; i < k; i += blockDim.x) xs[i] = xe[i];
    __syncthreads();

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= n_rows) return;
    int nblocks = k / 32;
    const unsigned char* wr = base + (size_t)slots[e] * expert_bytes + bank_off
                            + (size_t)row * nblocks * 18;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const unsigned short* p = reinterpret_cast<const unsigned short*>(wr + b * 18);
        float d = __half2float(__ushort_as_half(p[0]));
        const float* xb = xs + b * 32;
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int v = p[1 + j]; // bytes 2j and 2j+1 of the block's nibble bytes
            s += (v & 0xF) * xb[2 * j] + ((v >> 8) & 0xF) * xb[2 * j + 1]
               + ((v >> 4) & 0xF) * xb[2 * j + 16] + ((v >> 12) & 0xF) * xb[2 * j + 17]
               - 8.0f * (xb[2 * j] + xb[2 * j + 1] + xb[2 * j + 16] + xb[2 * j + 17]);
        }
        acc += d * s;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[(size_t)e * y_stride + row] = acc;
}

// out[e][j] = silu(gu[e][j]) * gu[e][inter + j] over all experts at once.
extern "C" __global__ void silu_mul_grouped(const float* __restrict__ gu,
                                            float* __restrict__ out,
                                            int inter, int n_experts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_experts * inter) {
        int e = i / inter, j = i % inter;
        float g = gu[(size_t)e * 2 * inter + j];
        out[i] = (g / (1.0f + expf(-g))) * gu[(size_t)e * 2 * inter + inter + j];
    }
}

// out[i] += a * sum_e y[e][i] — weighted accumulate of all expert outputs.
extern "C" __global__ void reduce_weighted(const float* __restrict__ y,
                                           float* __restrict__ out,
                                           float a, int hidden, int n_experts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        float s = 0.0f;
        for (int e = 0; e < n_experts; ++e) s += y[(size_t)e * hidden + i];
        out[i] += a * s;
    }
}

// ---- kernels for the full model forward (ft-model) ----

// GGUF q6_k GEMV: 256 elems / 210-byte block: ql[128] low-4, qh[64] hi-2,
// scales[16] i8 per 16 elems, then f16 d. value = d * sc * (q - 32).
// One warp per row, 4 rows per block (matches v2's shape; x staged in smem).
extern "C" __global__ void gemv_q6_k(
    const unsigned char* __restrict__ w, // [n_rows, k/256*210]
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_rows,
    int k)
{
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < k; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= n_rows) return;
    int nblocks = k / 256;
    const unsigned char* wr = w + (size_t)row * nblocks * 210;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const unsigned char* blk = wr + b * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char* sc = reinterpret_cast<const signed char*>(blk + 192);
        float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
        const float* xb = xs + b * 256;
        float s = 0.0f;
        // ggml q6_K layout: two 128-elem halves; within each half n (0..64)
        // pairs with n+64 sharing a ql byte, qh packs 4 x 2 bits.
        for (int half = 0; half < 2; ++half) {
            const unsigned char* qlh = ql + half * 64;
            const unsigned char* qhh = qh + half * 32;
            const float* xh = xb + half * 128;
            const signed char* sch = sc + half * 8;
            for (int i = 0; i < 32; ++i) {
                int q1 = (qlh[i] & 0xF) | (((qhh[i] >> 0) & 3) << 4);
                int q2 = (qlh[i + 32] & 0xF) | (((qhh[i] >> 2) & 3) << 4);
                int q3 = (qlh[i] >> 4) | (((qhh[i] >> 4) & 3) << 4);
                int q4 = (qlh[i + 32] >> 4) | (((qhh[i] >> 6) & 3) << 4);
                s += (float)sch[i / 16 + 0] * (q1 - 32) * xh[i]
                   + (float)sch[i / 16 + 2] * (q2 - 32) * xh[i + 32]
                   + (float)sch[i / 16 + 4] * (q3 - 32) * xh[i + 64]
                   + (float)sch[i / 16 + 6] * (q4 - 32) * xh[i + 96];
            }
        }
        acc += d * s;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[row] = acc;
}

// out[e][j] = gelu_tanh(gu[e][j]) * gu[e][inter + j] — gemma's expert/MLP
// activation (gelu_pytorch_tanh), grouped over experts.
extern "C" __global__ void gelu_mul_grouped(const float* __restrict__ gu,
                                            float* __restrict__ out,
                                            int inter, int n_experts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_experts * inter) {
        int e = i / inter, j = i % inter;
        float g = gu[(size_t)e * 2 * inter + j];
        float t = tanhf(0.7978845608f * (g + 0.044715f * g * g * g));
        out[i] = 0.5f * g * (1.0f + t) * gu[(size_t)e * 2 * inter + inter + j];
    }
}

// out[i] += sum_e wts[e] * y[e][i] — router-weighted expert combine.
extern "C" __global__ void reduce_expert_weighted(const float* __restrict__ y,
                                                  const float* __restrict__ wts,
                                                  float* __restrict__ out,
                                                  int hidden, int n_experts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        float s = 0.0f;
        for (int e = 0; e < n_experts; ++e) s += wts[e] * y[(size_t)e * hidden + i];
        out[i] += s;
    }
}
