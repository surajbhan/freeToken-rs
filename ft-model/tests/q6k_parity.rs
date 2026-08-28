use cudarc::driver::CudaContext;
use ft_cuda::Q8_BLK;
use ft_model::q6k;
use half::f16;

#[test]
fn q6k_dp4a_matches_reference() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let g = ft_cuda::Q4Gemv::new(&ctx)?;

    let (n_rows, k) = (64usize, 2816usize);
    let rb = q6k::row_bytes(k);
    let mut s = 0xABCDu32;
    let mut rndb = move || {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        (s & 0xFF) as u8
    };
    let mut w = vec![0u8; n_rows * rb];
    for blk in w.chunks_exact_mut(q6k::BLOCK_BYTES) {
        for b in blk[..192].iter_mut() {
            *b = rndb();
        }
        // realistic sub-scales (tiny): random bytes here would make |w| ~40x
        // real weights and amplify q8 activation noise past any tolerance
        for b in blk[192..208].iter_mut() {
            *b = (rndb() % 9) as u8; // scales 0..8
        }
        blk[208..210].copy_from_slice(&f16::from_f32(0.01).to_le_bytes());
    }
    let mut s2 = 0x5555u32;
    let x: Vec<f32> = (0..k)
        .map(|_| {
            s2 ^= s2 << 13; s2 ^= s2 >> 17; s2 ^= s2 << 5;
            (s2 as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();

    // CPU reference over dequantized rows
    let mut y_ref = vec![0f32; n_rows];
    let mut row_f = vec![0f32; k];
    for r in 0..n_rows {
        q6k::dequantize_row(&w[r * rb..(r + 1) * rb], &mut row_f);
        y_ref[r] = row_f.iter().zip(&x).map(|(a, b)| a * b).sum();
    }

    let w_dev = stream.memcpy_stod(&w)?;
    let x_dev = stream.memcpy_stod(&x)?;

    // old f32 kernel
    let mut y1 = stream.alloc_zeros::<f32>(n_rows)?;
    g.gemv_q6k(&stream, &w_dev, &x_dev, &mut y1, n_rows, k)?;
    // new dp4a kernel
    let mut q8 = stream.alloc_zeros::<u8>(k / 32 * Q8_BLK)?;
    g.quantize_q8(&stream, &x_dev, 0, &mut q8, k, 1)?;
    let mut y2 = stream.alloc_zeros::<f32>(n_rows)?;
    g.gemv_q6k_q8(&stream, &w_dev, &q8, &mut y2, n_rows, k)?;

    let a = stream.memcpy_dtov(&y1)?;
    let b = stream.memcpy_dtov(&y2)?;
    for r in 0..n_rows {
        assert!((a[r] - y_ref[r]).abs() < 1e-2 * (1.0 + y_ref[r].abs()), "f32 kernel r={r}: {} vs {}", a[r], y_ref[r]);
        assert!((b[r] - y_ref[r]).abs() < 0.4 + 0.02 * y_ref[r].abs(), "dp4a kernel r={r}: {} vs {}", b[r], y_ref[r]);
    }
    Ok(())
}
