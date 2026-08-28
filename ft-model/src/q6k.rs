//! CPU q6_k dequantization (ggml block_q6_K): 256 elements per 210-byte block
//! — ql[128] low-4 bits, qh[64] high-2 bits, scales[16] i8 (per 16 elems),
//! f16 super-scale. Used for embedding-row lookup; the GPU kernel handles the
//! tied lm_head.

use half::f16;

pub const QK_K: usize = 256;
pub const BLOCK_BYTES: usize = 210;

pub fn row_bytes(k: usize) -> usize {
    assert_eq!(k % QK_K, 0);
    k / QK_K * BLOCK_BYTES
}

pub fn dequantize_row(row: &[u8], out: &mut [f32]) {
    assert_eq!(row.len(), row_bytes(out.len()));
    for (blk, y) in row.chunks_exact(BLOCK_BYTES).zip(out.chunks_exact_mut(QK_K)) {
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let d = f16::from_le_bytes([blk[208], blk[209]]).to_f32();
        for half_idx in 0..2 {
            let qlh = &ql[half_idx * 64..];
            let qhh = &qh[half_idx * 32..];
            let sch = &sc[half_idx * 8..];
            let yh = half_idx * 128;
            for i in 0..32 {
                let s0 = sch[i / 16] as i8 as f32;
                let s1 = sch[i / 16 + 2] as i8 as f32;
                let s2 = sch[i / 16 + 4] as i8 as f32;
                let s3 = sch[i / 16 + 6] as i8 as f32;
                let q1 = ((qlh[i] & 0xF) | ((qhh[i] & 3) << 4)) as i32 - 32;
                let q2 = ((qlh[i + 32] & 0xF) | (((qhh[i] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((qlh[i] >> 4) | (((qhh[i] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((qlh[i + 32] >> 4) | (((qhh[i] >> 6) & 3) << 4)) as i32 - 32;
                y[yh + i] = d * s0 * q1 as f32;
                y[yh + i + 32] = d * s1 * q2 as f32;
                y[yh + i + 64] = d * s2 * q3 as f32;
                y[yh + i + 96] = d * s3 * q4 as f32;
            }
        }
    }
}
