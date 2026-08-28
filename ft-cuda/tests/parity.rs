//! GPU kernel vs CPU reference parity on real quantized data.
use cudarc::driver::CudaContext;
use ft_core::q4_0;

#[test]
fn gpu_gemv_matches_cpu() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let gemv = ft_cuda::Q4Gemv::new(&ctx)?;

    let (n_rows, k) = (256usize, 2048usize);
    let rb = q4_0::row_bytes(k);
    let mut s = 0x1234_5678u32;
    let mut rnd = move || {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        (s as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut w = vec![0u8; n_rows * rb];
    for r in 0..n_rows {
        let row: Vec<f32> = (0..k).map(|_| rnd()).collect();
        q4_0::quantize_row(&row, &mut w[r * rb..(r + 1) * rb]);
    }
    let x: Vec<f32> = (0..k).map(|_| rnd()).collect();

    // exact f32 reference over dequantized weights (the GPU kernel's own
    // math domain; q4_0::gemv quantizes activations to q8 and would differ)
    let mut y_cpu = vec![0f32; n_rows];
    let mut wrow = vec![0f32; k];
    for r in 0..n_rows {
        q4_0::dequantize_row(&w[r * rb..(r + 1) * rb], &mut wrow);
        y_cpu[r] = wrow.iter().zip(&x).map(|(a, b)| a * b).sum();
    }

    let w_dev = stream.memcpy_stod(&w)?;
    let x_dev = stream.memcpy_stod(&x)?;
    let mut y_dev = stream.alloc_zeros::<f32>(n_rows)?;
    gemv.launch(&stream, &w_dev.as_view(), &x_dev, &mut y_dev, n_rows, k)?;
    let y_gpu = stream.memcpy_dtov(&y_dev)?;

    for r in 0..n_rows {
        let (a, b) = (y_cpu[r], y_gpu[r]);
        assert!((a - b).abs() < 1e-2 * (1.0 + a.abs()), "row {r}: cpu {a} gpu {b}");
    }
    Ok(())
}
