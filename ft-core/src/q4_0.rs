//! GGUF q4_0 quantization: 32 weights per 18-byte block
//! (f16 scale `d` + 16 bytes of nibbles; byte i packs weight i in the low
//! nibble and weight i+16 in the high nibble). w = (q - 8) * d.
//! Matches ggml's block_q4_0, which FreeToken's borrowed kernels read.

use half::f16;
use rayon::prelude::*;

pub const QK: usize = 32;
pub const BLOCK_BYTES: usize = 18;

/// Bytes for one q4_0 row of `k` weights (`k` must be a multiple of 32).
pub fn row_bytes(k: usize) -> usize {
    assert_eq!(k % QK, 0);
    k / QK * BLOCK_BYTES
}

/// Quantize one f32 row into q4_0 blocks (ggml reference algorithm: d is the
/// signed max-amplitude value / -8, so the extreme value maps exactly to 0).
pub fn quantize_row(row: &[f32], out: &mut [u8]) {
    assert_eq!(out.len(), row_bytes(row.len()));
    for (xb, blk) in row.chunks_exact(QK).zip(out.chunks_exact_mut(BLOCK_BYTES)) {
        let mut amax = 0.0f32;
        let mut maxv = 0.0f32;
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                maxv = v;
            }
        }
        let d = maxv / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        blk[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        for i in 0..QK / 2 {
            let q0 = ((xb[i] * id + 8.5) as i32).clamp(0, 15) as u8;
            let q1 = ((xb[i + QK / 2] * id + 8.5) as i32).clamp(0, 15) as u8;
            blk[2 + i] = q0 | (q1 << 4);
        }
    }
}

/// Dequantize one q4_0 row into f32.
pub fn dequantize_row(row: &[u8], out: &mut [f32]) {
    assert_eq!(row.len(), row_bytes(out.len()));
    for (blk, xb) in row.chunks_exact(BLOCK_BYTES).zip(out.chunks_exact_mut(QK)) {
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for i in 0..QK / 2 {
            xb[i] = ((blk[2 + i] & 0xF) as i32 - 8) as f32 * d;
            xb[i + QK / 2] = ((blk[2 + i] >> 4) as i32 - 8) as f32 * d;
        }
    }
}

/// Activation vector quantized to q8_0 blocks (ggml's decode-path trick:
/// quantize x once per GEMV so the inner loop is integer SIMD).
pub struct Q8Vec {
    pub d: Vec<f32>,      // per-block scale
    pub qs: Vec<i8>,      // 32 per block
}

impl Q8Vec {
    pub fn quantize(x: &[f32]) -> Self {
        assert_eq!(x.len() % QK, 0);
        let nb = x.len() / QK;
        let mut d = Vec::with_capacity(nb);
        let mut qs = Vec::with_capacity(x.len());
        for xb in x.chunks_exact(QK) {
            let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            d.push(s);
            for &v in xb {
                qs.push((v * inv).round().clamp(-127.0, 127.0) as i8);
            }
        }
        Self { d, qs }
    }
}

/// y = W x for a q4_0 weight matrix W [n_rows, k] (rows consecutive),
/// f32 activations. Parallel over output rows — the CPU side of the hybrid
/// decode path (cpu_executor.py analog). Uses the AVX2 q4_0×q8_0 kernel when
/// available, else the scalar fallback.
pub fn gemv(w: &[u8], x: &[f32], y: &mut [f32]) {
    gemv_q8(w, &Q8Vec::quantize(x), y);
}

/// Same, with the activation vector already quantized (quantize once per
/// layer input, reuse across that step's CPU experts).
pub fn gemv_q8(w: &[u8], x8: &Q8Vec, y: &mut [f32]) {
    let k = x8.qs.len();
    let rb = row_bytes(k);
    assert_eq!(w.len(), rb * y.len());
    let use_avx2 = is_x86_feature_detected!("avx2");
    y.par_iter_mut().enumerate().for_each(|(r, yo)| {
        let row = &w[r * rb..(r + 1) * rb];
        *yo = if use_avx2 {
            unsafe { dot_q4_q8_avx2(row, x8) }
        } else {
            dot_q4_q8_scalar(row, x8)
        };
    });
}

fn dot_q4_q8_scalar(row: &[u8], x8: &Q8Vec) -> f32 {
    let mut acc = 0.0f32;
    for (b, blk) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let d4 = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let mut s = 0i32;
        for i in 0..QK / 2 {
            let lo = (blk[2 + i] & 0xF) as i32 - 8;
            let hi = (blk[2 + i] >> 4) as i32 - 8;
            s += lo * x8.qs[b * QK + i] as i32 + hi * x8.qs[b * QK + i + QK / 2] as i32;
        }
        acc += d4 * x8.d[b] * s as f32;
    }
    acc
}

/// ggml's vec_dot_q4_0_q8_0: nibbles unpacked to [0,16), the -8 offset folded
/// out via sum(q8) (w = (q-8)*d4 => dot = d4*d8*(sum(q*x8) - 8*sum(x8))),
/// signed maddubs for the i8 dot.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_q8_avx2(row: &[u8], x8: &Q8Vec) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let low_mask = _mm256_set1_epi8(0x0F);
    let ones16 = _mm256_set1_epi16(1);
    for (b, blk) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let d4 = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let scale = _mm256_set1_ps(d4 * x8.d[b]);

        // 16 packed bytes -> 32 nibbles laid out [w0..w15 | w16..w31]
        let packed = _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i);
        let q4 = _mm256_and_si256(
            _mm256_set_m128i(_mm_srli_epi16(packed, 4), packed),
            low_mask,
        );
        let q8 = _mm256_loadu_si256(x8.qs.as_ptr().add(b * QK) as *const __m256i);

        // sum(q4*x8): maddubs wants unsigned*signed — q4 is [0,15], fine
        let prod16 = _mm256_maddubs_epi16(q4, q8);
        let prod32 = _mm256_madd_epi16(prod16, ones16);
        // 8 * sum(x8): sad against zero handles signed poorly; use maddubs
        // with a constant 8 vector (unsigned 8 * signed q8)
        let eights = _mm256_set1_epi8(8);
        let off16 = _mm256_maddubs_epi16(eights, q8);
        let off32 = _mm256_madd_epi16(off16, ones16);

        let diff = _mm256_sub_epi32(prod32, off32);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(diff), scale, acc);
    }
    // horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_row(n: usize, seed: u32) -> Vec<f32> {
        // xorshift; deterministic, no rand dep
        let mut s = seed.wrapping_add(0x9E3779B9);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn roundtrip_error_is_small() {
        let row = pseudo_row(256, 7);
        let mut q = vec![0u8; row_bytes(256)];
        quantize_row(&row, &mut q);
        let mut back = vec![0f32; 256];
        dequantize_row(&q, &mut back);
        for (a, b) in row.iter().zip(&back) {
            assert!((a - b).abs() < 0.15, "{a} vs {b}");
        }
    }

    #[test]
    fn gemv_matches_f32_reference() {
        let k = 512;
        let n = 16;
        let x = pseudo_row(k, 1);
        let mut w = vec![0u8; row_bytes(k) * n];
        let mut wf = vec![0f32; k * n];
        for r in 0..n {
            let row = pseudo_row(k, 100 + r as u32);
            quantize_row(&row, &mut w[r * row_bytes(k)..(r + 1) * row_bytes(k)]);
            dequantize_row(
                &w[r * row_bytes(k)..(r + 1) * row_bytes(k)],
                &mut wf[r * k..(r + 1) * k],
            );
        }
        let mut y = vec![0f32; n];
        gemv(&w, &x, &mut y);
        for r in 0..n {
            let reference: f32 = (0..k).map(|i| wf[r * k + i] * x[i]).sum();
            // q8 activation quantization adds ~N(0, 0.03) noise on k=512
            // near-cancelling dots; bound at ~5 sigma. Layout/indexing bugs
            // produce errors orders of magnitude larger.
            assert!((y[r] - reference).abs() < 0.15, "row {r}: {} vs {reference}", y[r]);
        }
    }

    #[test]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let k = 4096;
        let row_f: Vec<f32> = pseudo_row(k, 42);
        let mut row = vec![0u8; row_bytes(k)];
        quantize_row(&row_f, &mut row);
        let x = pseudo_row(k, 7);
        let x8 = Q8Vec::quantize(&x);
        let a = dot_q4_q8_scalar(&row, &x8);
        let b = unsafe { dot_q4_q8_avx2(&row, &x8) };
        assert!((a - b).abs() < 1e-2 * (1.0 + a.abs()), "scalar {a} avx2 {b}");
    }
}

// ---- q4r: repacked q4_0 for coalesced GPU loads ----
// Row layout: [ scales: nblocks x f16, padded to 16B ][ qs: nblocks x 16B ].
// Block-internal byte order is unchanged, so dot math is identical.

pub fn q4r_scales_bytes(k: usize) -> usize {
    let n = k / QK * 2;
    (n + 15) / 16 * 16
}

pub fn q4r_row_bytes(k: usize) -> usize {
    q4r_scales_bytes(k) + k / QK * 16
}

/// repack one q4_0 row into q4r layout
pub fn repack_row_q4r(src: &[u8], dst: &mut [u8], k: usize) {
    let nb = k / QK;
    assert_eq!(src.len(), row_bytes(k));
    assert_eq!(dst.len(), q4r_row_bytes(k));
    let qs_off = q4r_scales_bytes(k);
    for b in 0..nb {
        dst[b * 2..b * 2 + 2].copy_from_slice(&src[b * 18..b * 18 + 2]);
        dst[qs_off + b * 16..qs_off + (b + 1) * 16]
            .copy_from_slice(&src[b * 18 + 2..(b + 1) * 18]);
    }
}

/// y = W x over a q4r matrix with pre-quantized activations (CPU hybrid path).
pub fn gemv_q8_r(w: &[u8], x8: &Q8Vec, y: &mut [f32]) {
    let k = x8.qs.len();
    let rb = q4r_row_bytes(k);
    assert_eq!(w.len(), rb * y.len());
    let use_avx2 = is_x86_feature_detected!("avx2");
    y.par_iter_mut().enumerate().for_each(|(r, yo)| {
        let row = &w[r * rb..(r + 1) * rb];
        *yo = if use_avx2 {
            unsafe { dot_q4r_q8_avx2(row, x8, k) }
        } else {
            dot_q4r_q8_scalar(row, x8, k)
        };
    });
}

fn dot_q4r_q8_scalar(row: &[u8], x8: &Q8Vec, k: usize) -> f32 {
    let nb = k / QK;
    let qs_off = q4r_scales_bytes(k);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let d4 = f16::from_le_bytes([row[b * 2], row[b * 2 + 1]]).to_f32();
        let qs = &row[qs_off + b * 16..qs_off + (b + 1) * 16];
        let mut s = 0i32;
        for i in 0..16 {
            let lo = (qs[i] & 0xF) as i32 - 8;
            let hi = (qs[i] >> 4) as i32 - 8;
            s += lo * x8.qs[b * QK + i] as i32 + hi * x8.qs[b * QK + i + 16] as i32;
        }
        acc += d4 * x8.d[b] * s as f32;
    }
    acc
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4r_q8_avx2(row: &[u8], x8: &Q8Vec, k: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = k / QK;
    let qs_off = q4r_scales_bytes(k);
    let mut acc = _mm256_setzero_ps();
    let low_mask = _mm256_set1_epi8(0x0F);
    let ones16 = _mm256_set1_epi16(1);
    let eights = _mm256_set1_epi8(8);
    for b in 0..nb {
        let d4 = f16::from_le_bytes([row[b * 2], row[b * 2 + 1]]).to_f32();
        let scale = _mm256_set1_ps(d4 * x8.d[b]);
        let packed = _mm_load_si128(row.as_ptr().add(qs_off + b * 16) as *const __m128i);
        let q4 = _mm256_and_si256(
            _mm256_set_m128i(_mm_srli_epi16(packed, 4), packed),
            low_mask,
        );
        let q8 = _mm256_loadu_si256(x8.qs.as_ptr().add(b * QK) as *const __m256i);
        let prod32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q4, q8), ones16);
        let off32 = _mm256_madd_epi16(_mm256_maddubs_epi16(eights, q8), ones16);
        let diff = _mm256_sub_epi32(prod32, off32);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(diff), scale, acc);
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

#[cfg(test)]
mod q4r_tests {
    use super::*;

    #[test]
    fn q4r_matches_q4() {
        let k = 2816usize;
        let n = 8usize;
        let mut s = 0xC0DEu32;
        let mut rnd = move || {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut w = vec![0u8; row_bytes(k) * n];
        let mut wr = vec![0u8; q4r_row_bytes(k) * n];
        for r in 0..n {
            let row: Vec<f32> = (0..k).map(|_| rnd()).collect();
            quantize_row(&row, &mut w[r * row_bytes(k)..(r + 1) * row_bytes(k)]);
            repack_row_q4r(
                &w[r * row_bytes(k)..(r + 1) * row_bytes(k)],
                &mut wr[r * q4r_row_bytes(k)..(r + 1) * q4r_row_bytes(k)],
                k,
            );
        }
        let x: Vec<f32> = (0..k).map(|_| rnd()).collect();
        let x8 = Q8Vec::quantize(&x);
        let mut y1 = vec![0f32; n];
        gemv_q8(&w, &x8, &mut y1);
        let mut y2 = vec![0f32; n];
        gemv_q8_r(&wr, &x8, &mut y2);
        for r in 0..n {
            assert!((y1[r] - y2[r]).abs() < 1e-3 * (1.0 + y1[r].abs()), "{r}: {} {}", y1[r], y2[r]);
        }
    }
}
