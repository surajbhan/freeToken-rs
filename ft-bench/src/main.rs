//! ft-bench: Rust port of FreeToken's `ft bench bw` calibration
//! (moe/benchbw.py). Measures the two bandwidths the q* policy is built on:
//!   - pcie_bw: pinned-host -> GPU async copy rate (how fast misses stream in)
//!   - cpu_bw : how fast CPU threads can stream expert weights through a GEMV
//! then prints the recommended backend and hybrid fetch fraction.

use anyhow::Result;
use cudarc::driver::CudaContext;
use rayon::prelude::*;
use std::time::Instant;

const MB: usize = 1 << 20;

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    println!("gpu: {}", ctx.name()?);
    let stream = ctx.default_stream();

    let nbytes = 256 * MB;
    let iters = 30;

    // --- PCIe H2D, pageable (reference point; benchbw measures pinned) ---
    let host = vec![1u8; nbytes];
    let mut dev = stream.alloc_zeros::<u8>(nbytes)?;
    stream.memcpy_htod(&host, &mut dev)?; // warmup
    stream.synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        stream.memcpy_htod(&host, &mut dev)?;
    }
    stream.synchronize()?;
    let pageable_gbs = (nbytes * iters) as f64 / t.elapsed().as_secs_f64() / 1e9;

    // --- PCIe H2D, pinned (this is benchbw.py::measure_pcie_bw) ---
    let mut pinned = unsafe { ctx.alloc_pinned::<u8>(nbytes)? };
    pinned.as_mut_slice()?.fill(1);
    stream.memcpy_htod(&pinned, &mut dev)?;
    stream.synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        stream.memcpy_htod(&pinned, &mut dev)?;
    }
    stream.synchronize()?;
    let pcie_gbs = (nbytes * iters) as f64 / t.elapsed().as_secs_f64() / 1e9;

    // --- CPU expert-GEMV streaming bandwidth (benchbw.py cpu_bw analog) ---
    // Proxy for the q4_0 CPU GEMV: bandwidth-bound pass over per-expert weight
    // blocks, one rayon task per expert, accumulating a dot-product so the
    // reads can't be optimized away. Expert size matches a Qwen3-30B-A3B
    // q4_0 expert (~3 * 768 * 2048 * 0.5625 bytes ≈ 2.7 MB).
    let expert_bytes = 2_654_208usize;
    let num_experts = 96usize; // ~250 MB working set, larger than L3
    let weights: Vec<Vec<u8>> = (0..num_experts)
        .map(|i| vec![(i % 251) as u8; expert_bytes])
        .collect();
    let cpu_pass = || -> u64 {
        weights
            .par_iter()
            .map(|w| {
                let mut acc = 0u64;
                for chunk in w.chunks_exact(8) {
                    acc = acc.wrapping_add(u64::from_le_bytes(chunk.try_into().unwrap()));
                }
                acc
            })
            .sum()
    };
    std::hint::black_box(cpu_pass()); // warmup + page-in
    let cpu_iters = 20;
    let t = Instant::now();
    for _ in 0..cpu_iters {
        std::hint::black_box(cpu_pass());
    }
    let cpu_gbs =
        (expert_bytes * num_experts * cpu_iters) as f64 / t.elapsed().as_secs_f64() / 1e9;

    // --- effective CPU q4_0 GEMV bandwidth (what q* actually rides on):
    // real AVX2 dequant-dot over expert-sized q4_0 matrices, weights read
    // once per pass. This is FreeToken's cpu_bw analog measured through the
    // actual kernel, not raw streaming.
    let (hidden, inter) = (2048usize, 768usize);
    let gu_bytes = ft_core::q4_0::row_bytes(hidden) * 2 * inter;
    let dn_bytes = ft_core::q4_0::row_bytes(inter) * hidden;
    let n_exp = 64usize;
    let experts_w: Vec<Vec<u8>> = (0..n_exp)
        .map(|i| {
            let mut v = vec![0u8; gu_bytes + dn_bytes];
            let scale = half::f16::from_f32(0.01).to_le_bytes();
            for blk in v.chunks_exact_mut(ft_core::q4_0::BLOCK_BYTES) {
                blk[..2].copy_from_slice(&scale);
                blk[2..].fill(0x50 | (i as u8 & 0xF));
            }
            v
        })
        .collect();
    let xh = vec![0.1f32; hidden];
    let xi = vec![0.1f32; inter];
    let mut ygu = vec![0f32; 2 * inter];
    let mut ydn = vec![0f32; hidden];
    let gemv_pass = |ygu: &mut Vec<f32>, ydn: &mut Vec<f32>| {
        for w in &experts_w {
            ft_core::q4_0::gemv(&w[..gu_bytes], &xh, ygu);
            ft_core::q4_0::gemv(&w[gu_bytes..], &xi, ydn);
        }
    };
    gemv_pass(&mut ygu, &mut ydn); // warmup
    let gemv_iters = 10;
    let t = Instant::now();
    for _ in 0..gemv_iters {
        gemv_pass(&mut ygu, &mut ydn);
    }
    let cpu_gemv_gbs = ((gu_bytes + dn_bytes) * n_exp * gemv_iters) as f64
        / t.elapsed().as_secs_f64()
        / 1e9;

    println!("h2d pageable : {pageable_gbs:6.2} GB/s");
    println!("h2d pinned   : {pcie_gbs:6.2} GB/s   (pcie_bw)");
    println!("cpu streaming: {cpu_gbs:6.2} GB/s   ({} threads)", rayon::current_num_threads());
    println!("cpu q4_0 gemv: {cpu_gemv_gbs:6.2} GB/s   (cpu_bw)");
    let cpu_gbs = cpu_gemv_gbs;
    let rec = ft_core::recommend(cpu_gbs, pcie_gbs);
    let frac = ft_core::qstar_fraction(pcie_gbs, cpu_gbs);
    println!("recommend    : {rec}");
    println!("q* fraction  : {frac:.3}  (fetch {:.0}% of misses over PCIe, CPU computes the rest)", frac * 100.0);
    Ok(())
}
