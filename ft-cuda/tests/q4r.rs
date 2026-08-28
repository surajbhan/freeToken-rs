use cudarc::driver::CudaContext;
use ft_core::q4_0;
use ft_cuda::Q8_BLK;

#[test]
fn q4r_gpu_matches_q4() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let g = ft_cuda::Q4Gemv::new(&ctx)?;
    for &(k, n_rows) in &[(2816usize, 64usize), (2112, 64), (704, 64), (4096, 32)] {
        let rb = q4_0::row_bytes(k);
        let rbr = q4_0::q4r_row_bytes(k);
        let qs_off = q4_0::q4r_scales_bytes(k);
        let mut s = 0xABCu32;
        let mut rnd = move || {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut w = vec![0u8; n_rows * rb];
        let mut wr = vec![0u8; n_rows * rbr];
        for r in 0..n_rows {
            let row: Vec<f32> = (0..k).map(|_| rnd()).collect();
            q4_0::quantize_row(&row, &mut w[r * rb..(r + 1) * rb]);
            q4_0::repack_row_q4r(&w[r * rb..(r + 1) * rb], &mut wr[r * rbr..(r + 1) * rbr], k);
        }
        let x: Vec<f32> = (0..k).map(|_| rnd()).collect();
        let w_dev = stream.memcpy_stod(&w)?;
        let wr_dev = stream.memcpy_stod(&wr)?;
        let x_dev = stream.memcpy_stod(&x)?;
        let mut q8 = stream.alloc_zeros::<u8>(k / 32 * Q8_BLK)?;
        g.quantize_q8(&stream, &x_dev, 0, &mut q8, k, 1)?;
        let zero = stream.memcpy_stod(&[0i32])?;
        let idx0 = stream.memcpy_stod(&[0i32])?;
        let mut y1 = stream.alloc_zeros::<f32>(n_rows)?;
        g.gemv_grouped_q8_idx(&stream, &w_dev, 0, 0, &zero, &idx0, 1, &q8, k / 32, &mut y1, n_rows, n_rows, k)?;
        let mut y2 = stream.alloc_zeros::<f32>(n_rows)?;
        g.gemv_q4r_idx(&stream, &wr_dev, 0, 0, &zero, &idx0, 1, &q8, k / 32, &mut y2, n_rows, n_rows, k, rbr, qs_off)?;
        let a = stream.memcpy_dtov(&y1)?;
        let b = stream.memcpy_dtov(&y2)?;
        for r in 0..n_rows {
            assert!(
                (a[r] - b[r]).abs() < 1e-3 * (1.0 + a[r].abs()),
                "k={k} r={r}: q4 {} q4r {}",
                a[r],
                b[r]
            );
        }
        println!("k={k} OK");
    }
    Ok(())
}
