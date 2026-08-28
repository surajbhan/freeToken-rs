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

// like gemv_q4_0_grouped_v3 but with an explicit activation-row indirection:
// entry e uses q8 row x_idx[e]. Enables batched MoE (per (seq,expert) pairs)
// and any layout where the activation isn't 1:1 with the entry index.
extern "C" __global__ void gemv_q4_0_grouped_v3_idx(
    const unsigned char* __restrict__ base,
    unsigned long long expert_bytes,
    unsigned long long bank_off,
    const int* __restrict__ slots,
    const int* __restrict__ x_idx,
    const unsigned char* __restrict__ q8,
    int q8_stride_blocks,
    float* __restrict__ y,
    int y_stride,
    int n_rows,
    int k)
{
    extern __shared__ unsigned char q8s[];
    int e = blockIdx.y;
    int nblocks = k / 32;
    {
        const unsigned char* src = q8 + (size_t)x_idx[e] * q8_stride_blocks * 40;
        for (int i = threadIdx.x; i < nblocks * 40 / 4; i += blockDim.x)
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
        const unsigned char* qb = q8s + b * 40;
        float d8 = *reinterpret_cast<const float*>(qb);
        float s8 = *reinterpret_cast<const float*>(qb + 4);
        const int* x8 = reinterpret_cast<const int*>(qb + 8);
        int isum = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned int g = (unsigned int)p[1 + 2 * i] | ((unsigned int)p[2 + 2 * i] << 16);
            int lo = g & 0x0F0F0F0Fu;
            int hi = (g >> 4) & 0x0F0F0F0Fu;
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

// per-sequence weighted combine of (seq,expert)-pair outputs:
// out[s][i] += sum over pairs p with seq_of[p]==s of wts[p] * y[p][i].
// Grid: (ceil(hidden/256), n_seqs); each block-row s scans all pairs.
extern "C" __global__ void reduce_pairs_weighted(
    const float* __restrict__ y,      // [n_pairs, hidden]
    const float* __restrict__ wts,    // [n_pairs]
    const int* __restrict__ seq_of,   // [n_pairs]
    float* __restrict__ out,          // [n_seqs, hidden]
    int hidden,
    int n_pairs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (i >= hidden) return;
    float acc = 0.0f;
    for (int p = 0; p < n_pairs; ++p)
        if (seq_of[p] == s) acc += wts[p] * y[(size_t)p * hidden + i];
    out[(size_t)s * hidden + i] += acc;
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

// batched decode attention over the pooled KV cache: grid (n_heads, n_seqs);
// per-seq slot/pos/kv_start arrays. q/out are [n_seqs, n_heads*hd].
extern "C" __global__ void attn_decode_batch(
    const unsigned short* __restrict__ kpool, // [max_batch, max_seq, kv_dim] f16 bits
    const unsigned short* __restrict__ vpool,
    const float* __restrict__ q,
    float* __restrict__ out,
    const int* __restrict__ slot_arr,
    const int* __restrict__ pos_arr,      // pos (inclusive last index)
    int window,                            // 0 = full attention
    int kv_dim,
    int hd,
    int group,
    int n_heads,
    int max_seq)
{
    extern __shared__ float sc[];
    int h = blockIdx.x;
    int s = blockIdx.y;
    int g = h / group;
    int kv_end = pos_arr[s] + 1;
    int kv_start = (window > 0 && kv_end > window) ? kv_end - window : 0;
    int n = kv_end - kv_start;
    const float* qh = q + (size_t)s * n_heads * hd + (size_t)h * hd;
    const __half* kc = reinterpret_cast<const __half*>(kpool)
                     + (size_t)slot_arr[s] * max_seq * kv_dim;
    const __half* vc = reinterpret_cast<const __half*>(vpool)
                     + (size_t)slot_arr[s] * max_seq * kv_dim;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const __half* kt = kc + (size_t)(kv_start + i) * kv_dim + (size_t)g * hd;
        float sv = 0.0f;
        for (int d = 0; d < hd; ++d) sv += qh[d] * __half2float(kt[d]);
        sc[i] = sv;
    }
    __syncthreads();
    __shared__ float red[128];
    float m = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, sc[i]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s2]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e2 = __expf(sc[i] - m);
        sc[i] = e2;
        sum += e2;
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) red[threadIdx.x] += red[threadIdx.x + s2];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    __syncthreads();
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        const __half* v0 = vc + (size_t)kv_start * kv_dim + (size_t)g * hd + d;
        for (int i = 0; i < n; ++i)
            acc += sc[i] * __half2float(v0[(size_t)i * kv_dim]);
        out[(size_t)s * n_heads * hd + (size_t)h * hd + d] = acc * inv;
    }
}

// ---- decode attention (bs=1): one block per query head ----
// K/V cached as f16 [max_seq, kv_dim]; scores staged in dynamic smem
// (ctx * 4B). Two-pass softmax; GQA maps query head h -> kv head h/group.
// sm_scale = 1.0 (gemma4: q/k are per-head RMS-normed).
extern "C" __global__ void attn_decode(
    const __half* __restrict__ kc,   // [max_seq, kv_dim]
    const __half* __restrict__ vc,   // [max_seq, kv_dim]
    const float* __restrict__ q,     // [n_heads, hd]
    float* __restrict__ out,         // [n_heads, hd]
    int kv_start,
    int kv_end,                       // exclusive
    int kv_dim,
    int hd,
    int group)
{
    extern __shared__ float sc[];    // kv_end - kv_start scores
    int h = blockIdx.x;
    int g = h / group;
    const float* qh = q + (size_t)h * hd;
    int n = kv_end - kv_start;

    // pass 1: scores
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const __half* kt = kc + (size_t)(kv_start + i) * kv_dim + (size_t)g * hd;
        float s = 0.0f;
        for (int d = 0; d < hd; ++d) s += qh[d] * __half2float(kt[d]);
        sc[i] = s;
    }
    __syncthreads();

    // block max + exp-sum (thread-strided, then smem tree on 128 partials)
    __shared__ float red[128];
    float m = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, sc[i]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s2]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = __expf(sc[i] - m);
        sc[i] = e;
        sum += e;
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) red[threadIdx.x] += red[threadIdx.x + s2];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    __syncthreads();

    // pass 2: out[d] = sum_p w_p * V[p][d]
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        const __half* v0 = vc + (size_t)kv_start * kv_dim + (size_t)g * hd + d;
        for (int i = 0; i < n; ++i)
            acc += sc[i] * __half2float(v0[(size_t)i * kv_dim]);
        out[(size_t)h * hd + d] = acc * inv;
    }
}

// ---- GPU-residency kernels: norms, rope, combine (batched, one row/block) ----

// y[e] = rmsnorm(x[e]) * w   (w null-> no scale). x,y [n, dim].
extern "C" __global__ void rmsnorm_rows(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int dim,
    float eps,
    int has_w)
{
    extern __shared__ float sh[];
    int e = blockIdx.x;
    const float* xe = x + (size_t)e * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xe[i];
        ss += v * v;
    }
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) sh[threadIdx.x] += sh[threadIdx.x + s2];
        __syncthreads();
    }
    float r = rsqrtf(sh[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        y[(size_t)e * dim + i] = xe[i] * r * (has_w ? w[i] : 1.0f);
}

// x[e] += a[e]  (residual add), batched rows
extern "C" __global__ void add_rows(
    float* __restrict__ x, const float* __restrict__ a, int dim)
{
    int e = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) x[(size_t)e * dim + i] += a[(size_t)e * dim + i];
}

// per-head rmsnorm over a section of batched qkv rows: sequence s's heads
// live at x + s*seq_stride + head*hd. Grid: n_seqs * heads_per_seq.
extern "C" __global__ void rmsnorm_heads(
    float* __restrict__ x,
    const float* __restrict__ w,
    int hd,
    float eps,
    int has_w,
    int seq_stride,
    int heads_per_seq)
{
    extern __shared__ float sh[];
    int e = blockIdx.x;
    int seq = e / heads_per_seq;
    int head = e % heads_per_seq;
    float* xe = x + (size_t)seq * seq_stride + (size_t)head * hd;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = xe[i];
        ss += v * v;
    }
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) sh[threadIdx.x] += sh[threadIdx.x + s2];
        __syncthreads();
    }
    float r = rsqrtf(sh[0] / hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
        xe[i] *= r * (has_w ? w[i] : 1.0f);
}

// NeoX rope over a section of batched qkv rows (same addressing as
// rmsnorm_heads); pos per sequence from pos_arr.
extern "C" __global__ void rope_heads(
    float* __restrict__ x,
    const float* __restrict__ inv_freq, // [hd/2]
    const int* __restrict__ pos_arr,    // [n_seqs]
    int hd,
    int heads_per_seq,
    int seq_stride)
{
    int e = blockIdx.x;
    int seq = e / heads_per_seq;
    int pos = pos_arr[seq];
    float* xe = x + (size_t)seq * seq_stride + (size_t)(e % heads_per_seq) * hd;
    int half = hd / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float th = pos * inv_freq[i];
        float sn = __sinf(th), cs = __cosf(th);
        float a = xe[i], b = xe[i + half];
        xe[i] = a * cs - b * sn;
        xe[i + half] = a * sn + b * cs;
    }
}

// append k/v sections of batched qkv rows into the pooled f16 KV cache:
// seq s (grid.y) copies qkv[s*seq_stride + sec_off .. +kv_dim] (f32) into
// pool[slot_arr[s]*max_seq*kv_dim + pos_arr[s]*kv_dim ..] as f16.
extern "C" __global__ void kv_append(
    const float* __restrict__ qkv,
    unsigned short* __restrict__ pool,
    const int* __restrict__ slot_arr,
    const int* __restrict__ pos_arr,
    int sec_off,
    int seq_stride,
    int kv_dim,
    int max_seq)
{
    int s = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    float v = qkv[(size_t)s * seq_stride + sec_off + i];
    size_t dst = ((size_t)slot_arr[s] * max_seq + (size_t)pos_arr[s]) * kv_dim + i;
    pool[dst] = __half_as_ushort(__float2half(v));
}

// gemma dual-rmsnorm combine + residual + layer scalar, batched rows:
// x[e] = (x[e] + rms(rms(shared[e])*w1 + rms(routed[e])*w2)*w3) * scalar
extern "C" __global__ void dual_combine_rows(
    float* __restrict__ x,
    const float* __restrict__ shared_y,
    const float* __restrict__ routed_y,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    float scalar,
    int dim,
    float eps)
{
    extern __shared__ float sh[];
    float* buf = sh + blockDim.x; // dim floats scratch for combined
    int e = blockIdx.x;
    const float* s1 = shared_y + (size_t)e * dim;
    const float* s2 = routed_y + (size_t)e * dim;

    float ss1 = 0.0f, ss2 = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        ss1 += s1[i] * s1[i];
        ss2 += s2[i] * s2[i];
    }
    sh[threadIdx.x] = ss1;
    __syncthreads();
    for (int s3 = blockDim.x / 2; s3 > 0; s3 >>= 1) {
        if (threadIdx.x < s3) sh[threadIdx.x] += sh[threadIdx.x + s3];
        __syncthreads();
    }
    float r1 = rsqrtf(sh[0] / dim + eps);
    __syncthreads();
    sh[threadIdx.x] = ss2;
    __syncthreads();
    for (int s3 = blockDim.x / 2; s3 > 0; s3 >>= 1) {
        if (threadIdx.x < s3) sh[threadIdx.x] += sh[threadIdx.x + s3];
        __syncthreads();
    }
    float r2 = rsqrtf(sh[0] / dim + eps);
    __syncthreads();

    float ssc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float c = s1[i] * r1 * w1[i] + s2[i] * r2 * w2[i];
        buf[i] = c;
        ssc += c * c;
    }
    sh[threadIdx.x] = ssc;
    __syncthreads();
    for (int s3 = blockDim.x / 2; s3 > 0; s3 >>= 1) {
        if (threadIdx.x < s3) sh[threadIdx.x] += sh[threadIdx.x + s3];
        __syncthreads();
    }
    float r3 = rsqrtf(sh[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        size_t o = (size_t)e * dim + i;
        x[o] = (x[o] + buf[i] * r3 * w3[i]) * scalar;
    }
}

// small f32 GEMV for the router: y[s][r] = dot(W[r], x[s]), W [n_rows, dim]
// f32 (scales pre-folded). Grid: (n_rows, n_seqs), one warp-ish block per row.
extern "C" __global__ void gemv_f32_rows(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int dim,
    int n_rows)
{
    __shared__ float sh[128];
    int r = blockIdx.x;
    int s = blockIdx.y;
    const float* wr = w + (size_t)r * dim;
    const float* xs = x + (size_t)s * dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += wr[i] * xs[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) sh[threadIdx.x] += sh[threadIdx.x + s2];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)s * n_rows + r] = sh[0];
}

// strided row gather: dst[s*row_len + i] = src[s*src_stride + src_off + i]
extern "C" __global__ void gather_rows(
    const float* __restrict__ src, float* __restrict__ dst,
    int src_stride, int src_off, int row_len)
{
    int s = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row_len)
        dst[(size_t)s * row_len + i] = src[(size_t)s * src_stride + src_off + i];
}

// pointer-based variant of the v3 grouped GEMV: entry e's expert weights are
// at bases[e] + bank_off (VRAM slot or UVA-mapped pinned host bank).
extern "C" __global__ void gemv_q4_0_grouped_v3_ptr(
    const unsigned long long* __restrict__ bases,
    unsigned long long bank_off,
    const int* __restrict__ x_idx,
    const unsigned char* __restrict__ q8,
    int q8_stride_blocks,
    float* __restrict__ y,
    int y_stride,
    int n_rows,
    int k)
{
    extern __shared__ unsigned char q8s[];
    int e = blockIdx.y;
    int nblocks = k / 32;
    {
        const unsigned char* src = q8 + (size_t)x_idx[e] * q8_stride_blocks * 40;
        for (int i = threadIdx.x; i < nblocks * 40 / 4; i += blockDim.x)
            reinterpret_cast<unsigned int*>(q8s)[i] =
                reinterpret_cast<const unsigned int*>(src)[i];
    }
    __syncthreads();

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= n_rows) return;
    const unsigned char* wr = reinterpret_cast<const unsigned char*>(bases[e])
                            + bank_off + (size_t)row * nblocks * 18;

    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        const unsigned short* p = reinterpret_cast<const unsigned short*>(wr + b * 18);
        float d4 = __half2float(__ushort_as_half(p[0]));
        const unsigned char* qb = q8s + b * 40;
        float d8 = *reinterpret_cast<const float*>(qb);
        float s8 = *reinterpret_cast<const float*>(qb + 4);
        const int* x8 = reinterpret_cast<const int*>(qb + 8);
        int isum = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned int g = (unsigned int)p[1 + 2 * i] | ((unsigned int)p[2 + 2 * i] << 16);
            int lo = g & 0x0F0F0F0Fu;
            int hi = (g >> 4) & 0x0F0F0F0Fu;
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

// fused sampling: greedy argmax (softcap is monotonic -> skipped), or
// Gumbel-argmax for temperature sampling over softcapped logits.
// One block per sequence; logits [n_seqs, vocab].
extern "C" __global__ void sample_tokens(
    const float* __restrict__ logits,
    const float* __restrict__ temps,     // per seq
    unsigned long long* __restrict__ rng, // per seq xorshift state
    int* __restrict__ out,
    int vocab,
    float cap)
{
    __shared__ float best_v[256];
    __shared__ int best_i[256];
    int s = blockIdx.x;
    const float* lg = logits + (size_t)s * vocab;
    float temp = temps[s];
    unsigned long long st = rng[s] + (unsigned long long)(threadIdx.x + 1) * 0x9E3779B97F4A7C15ull;

    float bv = -1e30f;
    int bi = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = lg[i];
        if (temp > 0.0f) {
            v = tanhf(v / cap) * cap / temp;
            // per-candidate gumbel noise
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            float u = (float)(st >> 40) / 16777216.0f + 1e-9f;
            v += -__logf(-__logf(u));
        }
        if (v > bv) { bv = v; bi = i; }
    }
    best_v[threadIdx.x] = bv;
    best_i[threadIdx.x] = bi;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2 && best_v[threadIdx.x + s2] > best_v[threadIdx.x]) {
            best_v[threadIdx.x] = best_v[threadIdx.x + s2];
            best_i[threadIdx.x] = best_i[threadIdx.x + s2];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[s] = best_i[0];
        // advance the sequence's rng deterministically
        unsigned long long r = rng[s];
        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
        rng[s] = r;
    }
}

// ---- device-side routing: topk, LRU admission, promotion ----

// per-seq top-k of router logits + softmax + per-expert rescale.
// One block per sequence; E <= 1024. Writes ids [s*k..] and weights.
extern "C" __global__ void topk_router(
    const float* __restrict__ logits,   // [n_seqs, E]
    const float* __restrict__ expert_scale, // [E]
    int* __restrict__ ids,              // [n_seqs, k]
    float* __restrict__ wts,            // [n_seqs, k]
    int e_count,
    int k)
{
    __shared__ float sv[1024];
    __shared__ float red_v[128];
    __shared__ int red_i[128];
    __shared__ float sel_v[32];
    __shared__ int sel_i[32];
    int s = blockIdx.x;
    const float* lg = logits + (size_t)s * e_count;
    for (int i = threadIdx.x; i < e_count; i += blockDim.x) sv[i] = lg[i];
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float bv = -1e30f;
        int bi = 0;
        for (int i = threadIdx.x; i < e_count; i += blockDim.x) {
            if (sv[i] > bv) { bv = sv[i]; bi = i; }
        }
        red_v[threadIdx.x] = bv;
        red_i[threadIdx.x] = bi;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (threadIdx.x < s2 && red_v[threadIdx.x + s2] > red_v[threadIdx.x]) {
                red_v[threadIdx.x] = red_v[threadIdx.x + s2];
                red_i[threadIdx.x] = red_i[threadIdx.x + s2];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            sel_v[j] = red_v[0];
            sel_i[j] = red_i[0];
            sv[red_i[0]] = -1e30f;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = sel_v[0];
        float denom = 0.0f;
        for (int j = 0; j < k; ++j) {
            sel_v[j] = __expf(sel_v[j] - mx);
            denom += sel_v[j];
        }
        for (int j = 0; j < k; ++j) {
            ids[(size_t)s * k + j] = sel_i[j];
            wts[(size_t)s * k + j] = sel_v[j] / denom * expert_scale[sel_i[j]];
        }
    }
}

// device-side global LRU admission. Single block. For each of n_entries
// routed (layer,expert) keys: hit -> VRAM slot base; miss -> evict LRU slot,
// remap, emit the pinned host bank base (UVA read this step) and a promote
// record filling the slot for future steps.
extern "C" __global__ void lru_admit(
    const int* __restrict__ ids,        // [n_entries] expert ids
    int layer,
    int e_count,
    int n_slots,
    unsigned long long cache_base,
    unsigned long long banks_base,
    unsigned long long expert_bytes,
    int* __restrict__ map,              // [L*E] -> slot or -1
    int* __restrict__ slot_key,         // [n_slots] -> key or -1
    unsigned int* __restrict__ slot_last, // [n_slots]
    unsigned int* __restrict__ clock_ctr, // [1]
    unsigned long long* __restrict__ bases, // [n_entries] out
    int* __restrict__ promote_src_key,  // [n_entries] out (-1 = none)
    int* __restrict__ promote_dst_slot, // [n_entries] out
    int n_entries)
{
    __shared__ unsigned int red_v[128];
    __shared__ int red_i[128];
    // pass 1: touch every hit first so this step's hits can never be chosen
    // as eviction victims by this step's misses
    if (threadIdx.x == 0) {
        for (int t = 0; t < n_entries; ++t) {
            int key = layer * e_count + ids[t];
            int slot = map[key];
            if (slot >= 0) {
                unsigned int c = ++clock_ctr[0];
                slot_last[slot] = c;
                bases[t] = cache_base + (unsigned long long)slot * expert_bytes;
                promote_src_key[t] = -1;
            } else {
                promote_src_key[t] = -2; // marks "needs admission" for pass 2
            }
        }
    }
    __syncthreads();
    for (int t = 0; t < n_entries; ++t) {
        if (promote_src_key[t] != -2) continue;
        int key = layer * e_count + ids[t];
        int slot = map[key];
        if (slot >= 0) {
            // duplicate expert admitted earlier this pass (another sequence)
            if (threadIdx.x == 0) {
                bases[t] = cache_base + (unsigned long long)slot * expert_bytes;
                promote_src_key[t] = -1;
            }
        } else {
            // parallel LRU scan
            unsigned int bv = 0xFFFFFFFFu;
            int bi = 0;
            for (int i = threadIdx.x; i < n_slots; i += blockDim.x) {
                if (slot_last[i] < bv) { bv = slot_last[i]; bi = i; }
            }
            red_v[threadIdx.x] = bv;
            red_i[threadIdx.x] = bi;
            __syncthreads();
            for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
                if (threadIdx.x < s2 && red_v[threadIdx.x + s2] < red_v[threadIdx.x]) {
                    red_v[threadIdx.x] = red_v[threadIdx.x + s2];
                    red_i[threadIdx.x] = red_i[threadIdx.x + s2];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                int victim = red_i[0];
                int old = slot_key[victim];
                if (old >= 0) map[old] = -1;
                slot_key[victim] = key;
                map[key] = victim;
                unsigned int c = ++clock_ctr[0];
                slot_last[victim] = c;
                bases[t] = banks_base + (unsigned long long)key * expert_bytes;
                promote_src_key[t] = key;
                promote_dst_slot[t] = victim;
            }
        }
        __syncthreads();
    }
}

// promote missed experts into their assigned slots: entry t copies
// expert_bytes from the pinned host bank (UVA) into the VRAM slot.
// Grid: (chunks, n_entries); entries with promote_src_key < 0 no-op.
extern "C" __global__ void promote_experts(
    const int* __restrict__ promote_src_key,
    const int* __restrict__ promote_dst_slot,
    unsigned long long banks_base,
    unsigned long long cache_base,
    unsigned long long expert_bytes)
{
    int t = blockIdx.y;
    int key = promote_src_key[t];
    if (key < 0) return;
    const unsigned int* src = reinterpret_cast<const unsigned int*>(
        banks_base + (unsigned long long)key * expert_bytes);
    unsigned int* dst = reinterpret_cast<unsigned int*>(
        cache_base + (unsigned long long)promote_dst_slot[t] * expert_bytes);
    size_t n = expert_bytes / 4;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// ---- q4r: repacked q4_0 (scales-first, 16B-aligned nibble payloads) ----
// Row: [ nblocks x f16 scales, padded to 16B ][ nblocks x 16B qs ].
// A warp's lanes load consecutive uint4s -> fully coalesced 512B/step.

// helper shared by the idx/ptr variants
__device__ __forceinline__ float q4r_row_dot(
    const unsigned char* __restrict__ wr,
    const unsigned char* __restrict__ q8s,
    int nblocks,
    int qs_off,
    int lane)
{
    const __half* sc = reinterpret_cast<const __half*>(wr);
    const uint4* qs = reinterpret_cast<const uint4*>(wr + qs_off);
    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        float d4 = __half2float(sc[b]);
        uint4 g4 = qs[b];
        const unsigned char* qb = q8s + b * 40;
        float d8 = *reinterpret_cast<const float*>(qb);
        float s8 = *reinterpret_cast<const float*>(qb + 4);
        const int* x8 = reinterpret_cast<const int*>(qb + 8);
        int isum = 0;
        unsigned int gg[4] = {g4.x, g4.y, g4.z, g4.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int lo = gg[i] & 0x0F0F0F0Fu;
            int hi = (gg[i] >> 4) & 0x0F0F0F0Fu;
            isum = __dp4a(lo, x8[i], isum);
            isum = __dp4a(hi, x8[4 + i], isum);
        }
        acc += d4 * (d8 * (float)isum - 8.0f * s8);
    }
    return acc;
}

extern "C" __global__ void gemv_q4r_grouped_idx(
    const unsigned char* __restrict__ base,
    unsigned long long expert_bytes,
    unsigned long long bank_off,
    const int* __restrict__ slots,
    const int* __restrict__ x_idx,
    const unsigned char* __restrict__ q8,
    int q8_stride_blocks,
    float* __restrict__ y,
    int y_stride,
    int n_rows,
    int k,
    int row_bytes_r,
    int qs_off)
{
    extern __shared__ unsigned char q8s[];
    int e = blockIdx.y;
    int nblocks = k / 32;
    {
        const unsigned char* src = q8 + (size_t)x_idx[e] * q8_stride_blocks * 40;
        for (int i = threadIdx.x; i < nblocks * 40 / 4; i += blockDim.x)
            reinterpret_cast<unsigned int*>(q8s)[i] =
                reinterpret_cast<const unsigned int*>(src)[i];
    }
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= n_rows) return;
    const unsigned char* wr = base + (size_t)slots[e] * expert_bytes + bank_off
                            + (size_t)row * row_bytes_r;
    float acc = q4r_row_dot(wr, q8s, nblocks, qs_off, lane);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[(size_t)e * y_stride + row] = acc;
}

extern "C" __global__ void gemv_q4r_grouped_ptr(
    const unsigned long long* __restrict__ bases,
    unsigned long long bank_off,
    const int* __restrict__ x_idx,
    const unsigned char* __restrict__ q8,
    int q8_stride_blocks,
    float* __restrict__ y,
    int y_stride,
    int n_rows,
    int k,
    int row_bytes_r,
    int qs_off)
{
    extern __shared__ unsigned char q8s[];
    int e = blockIdx.y;
    int nblocks = k / 32;
    {
        const unsigned char* src = q8 + (size_t)x_idx[e] * q8_stride_blocks * 40;
        for (int i = threadIdx.x; i < nblocks * 40 / 4; i += blockDim.x)
            reinterpret_cast<unsigned int*>(q8s)[i] =
                reinterpret_cast<const unsigned int*>(src)[i];
    }
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= n_rows) return;
    const unsigned char* wr = reinterpret_cast<const unsigned char*>(bases[e])
                            + bank_off + (size_t)row * row_bytes_r;
    float acc = q4r_row_dot(wr, q8s, nblocks, qs_off, lane);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[(size_t)e * y_stride + row] = acc;
}
