//! Grouped one-launch path vs per-expert launches on real quantized data.
use cudarc::driver::CudaContext;
use ft_core::q4_0;

#[test]
fn grouped_matches_single() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let g = ft_cuda::Q4Gemv::new(&ctx)?;

    let (hidden, inter, n_exp) = (256usize, 96usize, 5usize);
    let gu_rows = 2 * inter;
    let gu_bytes = q4_0::row_bytes(hidden) * gu_rows;
    let dn_bytes = q4_0::row_bytes(inter) * hidden;
    let eb = gu_bytes + dn_bytes;

    let mut s = 0xBEEFu32;
    let mut rnd = move || {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        (s as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    // slot cache with n_exp experts in shuffled slots
    let mut cache = vec![0u8; n_exp * eb];
    for e in 0..n_exp {
        for r in 0..gu_rows {
            let row: Vec<f32> = (0..hidden).map(|_| rnd()).collect();
            let o = e * eb + r * q4_0::row_bytes(hidden);
            q4_0::quantize_row(&row, &mut cache[o..o + q4_0::row_bytes(hidden)]);
        }
        for r in 0..hidden {
            let row: Vec<f32> = (0..inter).map(|_| rnd()).collect();
            let o = e * eb + gu_bytes + r * q4_0::row_bytes(inter);
            q4_0::quantize_row(&row, &mut cache[o..o + q4_0::row_bytes(inter)]);
        }
    }
    let cache_dev = stream.memcpy_stod(&cache)?;
    let slots: Vec<i32> = vec![3, 0, 4, 1, 2];
    let slots_dev = stream.memcpy_stod(&slots)?;
    let x: Vec<f32> = (0..hidden).map(|_| rnd()).collect();
    let x_dev = stream.memcpy_stod(&x)?;

    // grouped pipeline: gate_up -> silu -> down -> reduce
    let mut y_gu = stream.alloc_zeros::<f32>(n_exp * gu_rows)?;
    let mut act = stream.alloc_zeros::<f32>(n_exp * inter)?;
    let mut y_dn = stream.alloc_zeros::<f32>(n_exp * hidden)?;
    let mut out = stream.alloc_zeros::<f32>(hidden)?;
    g.gemv_grouped(&stream, &cache_dev, eb, 0, &slots_dev, n_exp, &x_dev, 0, &mut y_gu, gu_rows, gu_rows, hidden)?;
    g.silu_mul_grouped(&stream, &y_gu, &mut act, inter, n_exp)?;
    g.gemv_grouped(&stream, &cache_dev, eb, gu_bytes, &slots_dev, n_exp, &act, inter, &mut y_dn, hidden, hidden, inter)?;
    g.reduce_weighted(&stream, &y_dn, &mut out, 0.2, hidden, n_exp)?;
    let got = stream.memcpy_dtov(&out)?;

    // reference: per-expert single launches
    let mut y1 = stream.alloc_zeros::<f32>(gu_rows)?;
    let mut a1 = stream.alloc_zeros::<f32>(inter)?;
    let mut d1 = stream.alloc_zeros::<f32>(hidden)?;
    let mut ref_out = vec![0f32; hidden];
    for &slot in &slots {
        let s = slot as usize;
        let wgu = cache_dev.slice(s * eb..s * eb + gu_bytes);
        let wdn = cache_dev.slice(s * eb + gu_bytes..(s + 1) * eb);
        g.launch(&stream, &wgu, &x_dev, &mut y1, gu_rows, hidden)?;
        g.silu_mul(&stream, &y1, &mut a1, inter)?;
        g.launch(&stream, &wdn, &a1, &mut d1, hidden, inter)?;
        let v = stream.memcpy_dtov(&d1)?;
        for i in 0..hidden {
            ref_out[i] += 0.2 * v[i];
        }
    }
    for i in 0..hidden {
        assert!(
            (got[i] - ref_out[i]).abs() < 1e-3 * (1.0 + ref_out[i].abs()),
            "i={i}: grouped {} vs single {}",
            got[i],
            ref_out[i]
        );
    }
    Ok(())
}
