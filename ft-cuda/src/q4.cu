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

// ---- v3: integer dp4a path (ggml mmvq approach) ----
// Activations are quantized once on-GPU to q8 blocks of 32:
//   { f32 d; f32 s; int8 qs[32] }  (40 bytes, s = d * sum(qs))
// The GEMV then does 8 dp4a per q4_0 block instead of 32 fp32 FMAs, and the
// -8 offset folds into s:  dot = d4 * (d8 * sum(q4*q8) - 8 * s8).

#define Q8_BLK 40

// one warp per 32-elem block: x[e*x_stride + b*32 + lane]
extern "C" __global__ void quantize_q8_grouped(
    const float* __restrict__ x,
    int x_stride,
    unsigned char* __restrict__ q8, // [n_experts, nblocks*40]
    int nblocks,
    int n_experts)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int e = warp / nblocks;
    int b = warp % nblocks;
    if (e >= n_experts) return;
    float v = x[(size_t)e * x_stride + b * 32 + lane];
    float a = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
    float d = a > 0.0f ? a / 127.0f : 0.0f;
    float inv = d > 0.0f ? 1.0f / d : 0.0f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    int s = q;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
    unsigned char* blk = q8 + ((size_t)e * nblocks + b) * Q8_BLK;
    if (lane == 0) {
        *reinterpret_cast<float*>(blk) = d;
        *reinterpret_cast<float*>(blk + 4) = d * (float)s;
    }
    blk[8 + lane] = (unsigned char)(signed char)q;
}

// grouped GEMV over q8-quantized activations. Same slot-cache addressing as
// v2; q8 activation staged in shared memory (nblocks*40 bytes).
extern "C" __global__ void gemv_q4_0_grouped_v3(
    const unsigned char* __restrict__ base,
    unsigned long long expert_bytes,
    unsigned long long bank_off,
    const int* __restrict__ slots,
    const unsigned char* __restrict__ q8, // [n_experts or 1, nblocks*40]
    int q8_stride_blocks,                  // 0 = shared across experts
    float* __restrict__ y,
    int y_stride,
    int n_rows,
    int k)
{
    extern __shared__ unsigned char q8s[];
    int e = blockIdx.y;
    int nblocks = k / 32;
    {
        const unsigned char* src = q8 + (size_t)e * q8_stride_blocks * Q8_BLK;
        for (int i = threadIdx.x; i < nblocks * Q8_BLK / 4; i += blockDim.x)
            reinterpret_cast<unsigned int*>(q8s)[i] =
                reinterpret_cast<const unsigned int*>(src)[i];
    }
    __syncthreads();

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= n_rows) return;
    const unsigned char* wr = base + (size_t)slots[e] * expert_bytes + bank_off
                            + (size_t)row * nblocks * 18;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const unsigned short* p = reinterpret_cast<const unsigned short*>(wr + b * 18);
        float d4 = __half2float(__ushort_as_half(p[0]));
        const unsigned char* qb = q8s + b * Q8_BLK;
        float d8 = *reinterpret_cast<const float*>(qb);
        float s8 = *reinterpret_cast<const float*>(qb + 4);
        const int* x8 = reinterpret_cast<const int*>(qb + 8);
        int isum = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned int g = (unsigned int)p[1 + 2 * i] | ((unsigned int)p[2 + 2 * i] << 16);
            int lo = g & 0x0F0F0F0Fu;          // q4 values 4i..4i+3
            int hi = (g >> 4) & 0x0F0F0F0Fu;   // q4 values 16+4i..19+4i
            isum = __dp4a(lo, x8[i], isum);
            isum = __dp4a(hi, x8[4 + i], isum);
        }
        acc += d4 * (d8 * (float)isum - 8.0f * s8);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[(size_t)e * y_stride + row] = acc;
}

// q6_k GEMV over q8 activations via dp4a (lm_head fast path). q6 values fit
// signed bytes; per-16 sub-scales applied on partial sums; the -32 offset
// uses dp4a(0x01010101, x) byte-sums. One warp per row, 4 rows/block,
// q8 staged in smem.
extern "C" __global__ void gemv_q6_k_q8(
    const unsigned char* __restrict__ w,
    const unsigned char* __restrict__ q8, // k/32 blocks of 40B
    float* __restrict__ y,
    int n_rows,
    int k)
{
    extern __shared__ unsigned char q8s[];
    int nq8 = k / 32;
    for (int i = threadIdx.x; i < nq8 * 40 / 4; i += blockDim.x)
        reinterpret_cast<unsigned int*>(q8s)[i] =
            reinterpret_cast<const unsigned int*>(q8)[i];
    __syncthreads();

    // 8 lanes per row (rows are short: k=2816 -> 11 q6k blocks), 16 rows/block
    int lane = threadIdx.x & 31;
    int sub = lane & 7;
    int row = blockIdx.x * 16 + (threadIdx.x >> 3);
    if (row >= n_rows) return;
    int nblocks = k / 256;
    const unsigned char* wr = w + (size_t)row * nblocks * 210;

    float acc = 0.0f;
    for (int b = sub; b < nblocks; b += 8) {
        const unsigned char* blk = wr + b * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char* sc = reinterpret_cast<const signed char*>(blk + 192);
        float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));

        float fsum = 0.0f;
        // 2 halves x 4 sub-rounds of 32 values; scales cover 16 consecutive.
        for (int half = 0; half < 2; ++half) {
            const unsigned char* qlh = ql + half * 64;
            const unsigned char* qhh = qh + half * 32;
            const signed char* sch = sc + half * 8;
            const unsigned char* xh = q8s + (b * 8 + half * 4) * 40;
            // groups g=0..3 map to value ranges [32g, 32g+32) of this half:
            //   g0: ql[i]&0xF  | qh bits0-1   -> values 0..31
            //   g1: ql[i+32]&F | qh bits2-3   -> values 32..63
            //   g2: ql[i]>>4   | qh bits4-5   -> values 64..95
            //   g3: ql[i+32]>>4| qh bits6-7   -> values 96..127
            for (int g = 0; g < 4; ++g) {
                const unsigned char* qlg = qlh + (g & 1) * 32;
                int shiftl = (g & 2) * 2;   // 0 or 4
                int shifth = g * 2;
                const unsigned char* xb = xh + g * 40; // matching q8 block of 32
                float d8 = *reinterpret_cast<const float*>(xb);
                const int* x8 = reinterpret_cast<const int*>(xb + 8);
                int isum0 = 0, isum1 = 0, xs0 = 0, xs1 = 0;
                // q6 blocks are 210B (2 mod 4): u32 loads may be unaligned,
                // so assemble from 2-byte-aligned u16 pairs.
                const unsigned short* pql = reinterpret_cast<const unsigned short*>(qlg);
                const unsigned short* pqh = reinterpret_cast<const unsigned short*>(qhh);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    unsigned int lq = (unsigned int)pql[2 * i] | ((unsigned int)pql[2 * i + 1] << 16);
                    unsigned int hq = (unsigned int)pqh[2 * i] | ((unsigned int)pqh[2 * i + 1] << 16);
                    unsigned int q6a = ((lq >> shiftl) & 0x0F0F0F0Fu)
                                     | (((hq >> shifth) & 0x03030303u) << 4);
                    isum0 = __dp4a((int)q6a, x8[i], isum0);
                    xs0 = __dp4a(0x01010101, x8[i], xs0);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    unsigned int lq = (unsigned int)pql[8 + 2 * i] | ((unsigned int)pql[9 + 2 * i] << 16);
                    unsigned int hq = (unsigned int)pqh[8 + 2 * i] | ((unsigned int)pqh[9 + 2 * i] << 16);
                    unsigned int q6a = ((lq >> shiftl) & 0x0F0F0F0Fu)
                                     | (((hq >> shifth) & 0x03030303u) << 4);
                    isum1 = __dp4a((int)q6a, x8[4 + i], isum1);
                    xs1 = __dp4a(0x01010101, x8[4 + i], xs1);
                }
                float s0 = (float)sch[g * 2];
                float s1 = (float)sch[g * 2 + 1];
                fsum += d8 * (s0 * (float)(isum0 - 32 * xs0) + s1 * (float)(isum1 - 32 * xs1));
            }
        }
        acc += d * fsum;
    }
#pragma unroll
    for (int off = 4; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (sub == 0) y[row] = acc;
}
