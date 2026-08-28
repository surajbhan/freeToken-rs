//! dp4a v3 path vs f32 v2 path on real quantized data.
use cudarc::driver::CudaContext;
use ft_core::q4_0;
use ft_cuda::Q8_BLK;

#[test]
fn v3_matches_v2() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let g = ft_cuda::Q4Gemv::new(&ctx)?;

    let (hidden, n_rows, n_exp) = (2816usize, 96usize, 3usize);
    let rb = q4_0::row_bytes(hidden);
    let eb = n_rows * rb;
    let mut s = 0xFACEu32;
    let mut rnd = move || {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        (s as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut cache = vec![0u8; n_exp * eb];
    for r in 0..n_exp * n_rows {
        let row: Vec<f32> = (0..hidden).map(|_| rnd()).collect();
        q4_0::quantize_row(&row, &mut cache[r * rb..(r + 1) * rb]);
    }
    let cache_dev = stream.memcpy_stod(&cache)?;
    let slots_dev = stream.memcpy_stod(&[2i32, 0, 1])?;
    let x: Vec<f32> = (0..hidden).map(|_| rnd()).collect();
    let x_dev = stream.memcpy_stod(&x)?;

    // v2 reference (f32 activations)
    let mut y2 = stream.alloc_zeros::<f32>(n_exp * n_rows)?;
    g.gemv_grouped(&stream, &cache_dev, eb, 0, &slots_dev, n_exp, &x_dev, 0, &mut y2, n_rows, n_rows, hidden)?;

    // v3: quantize shared x then dp4a
    let nblocks = hidden / 32;
    let mut q8 = stream.alloc_zeros::<u8>(nblocks * Q8_BLK)?;
    g.quantize_q8(&stream, &x_dev, 0, &mut q8, hidden, 1)?;
    let mut y3 = stream.alloc_zeros::<f32>(n_exp * n_rows)?;
    g.gemv_grouped_q8(&stream, &cache_dev, eb, 0, &slots_dev, n_exp, &q8, 0, &mut y3, n_rows, n_rows, hidden)?;

    let a = stream.memcpy_dtov(&y2)?;
    let b = stream.memcpy_dtov(&y3)?;
    let scale: f32 = (hidden as f32).sqrt() * 0.35; // rough |dot| scale
    for i in 0..a.len() {
        assert!(
            (a[i] - b[i]).abs() < 0.02 * scale,
            "i={i}: v2 {} v3 {}",
            a[i],
            b[i]
        );
    }
    Ok(())
}
