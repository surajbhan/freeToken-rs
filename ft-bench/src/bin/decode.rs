//! Hybrid MoE decode microbenchmark — the Rust analog of FreeToken's decode
//! path (offload_cache.copy_missing + hybrid split + cpu_executor overlap),
//! with synthetic expert weights and a uniform-random router (worst case: real
//! routing has reuse locality, so real hit rates are higher).
//!
//! Geometry defaults match one Qwen3-30B-A3B MoE layer (128 experts, top-8,
//! hidden 2048, intermediate 768, q4_0). Copies are one async H2D per fetched
//! expert (2.65 MB each — large enough to run at full PCIe rate; the batched
//! cudaMemcpyBatchAsync of the original is a later optimization).
//!
//! Usage: decode [layers=8] [experts=128] [topk=8] [hidden=2048] [inter=768]
//!               [slots=512] [steps=100] [fraction=0.2] [mode=hybrid|offload|cpu]

use anyhow::Result;
use cudarc::driver::CudaContext;
use ft_core::{q4_0, qstar_fraction, split_misses, SlotCache};
use ft_gguf::Gguf;
use std::time::Instant;

fn arg(name: &str, default: f64) -> f64 {
    std::env::args()
        .find_map(|a| a.strip_prefix(&format!("{name}=")).map(|v| v.parse().unwrap()))
        .unwrap_or(default)
}
fn arg_s(name: &str, default: &str) -> String {
    std::env::args()
        .find_map(|a| a.strip_prefix(&format!("{name}=")).map(str::to_string))
        .unwrap_or_else(|| default.to_string())
}

fn main() -> Result<()> {
    let gguf_path = arg_s("gguf", "");
    let gguf = if gguf_path.is_empty() { None } else { Some(Gguf::open(&gguf_path)?) };

    let (mut layers, mut experts, mut hidden, mut inter, mut topk) = (
        arg("layers", 8.0) as usize,
        arg("experts", 128.0) as usize,
        arg("hidden", 2048.0) as usize,
        arg("inter", 768.0) as usize,
        arg("topk", 8.0) as usize,
    );
    let mut moe_blocks: Vec<usize> = Vec::new(); // gguf block ids that hold experts
    let mut total_moe_layers = 0usize;
    if let Some(g) = &gguf {
        let arch = g.meta["general.architecture"].as_str().unwrap().to_string();
        let m = |k: &str| g.meta[&format!("{arch}.{k}")].as_u64().unwrap() as usize;
        experts = m("expert_count");
        topk = m("expert_used_count");
        // dense layers (no routed experts) are skipped: probe the tensor table.
        // qwen3moe ships split gate/up expert tensors; gemma4 ships them fused.
        moe_blocks = (0..m("block_count"))
            .filter(|l| {
                g.tensors.contains_key(&format!("blk.{l}.ffn_gate_exps.weight"))
                    || g.tensors.contains_key(&format!("blk.{l}.ffn_gate_up_exps.weight"))
            })
            .collect();
        // geometry from the tensors themselves (metadata key names vary by arch)
        let b0 = moe_blocks.first().copied().unwrap_or(0);
        if let Ok(t) = g.tensor(&format!("blk.{b0}.ffn_gate_up_exps.weight")) {
            hidden = t.dims[0] as usize;
            inter = t.dims[1] as usize / 2;
        } else if let Ok(t) = g.tensor(&format!("blk.{b0}.ffn_gate_exps.weight")) {
            hidden = t.dims[0] as usize;
            inter = t.dims[1] as usize;
        }
        total_moe_layers = moe_blocks.len();
        moe_blocks.truncate(arg("max_layers", 1e9) as usize);
        layers = moe_blocks.len();
        println!(
            "gguf: {arch} {} blocks, {total_moe_layers} MoE (using {layers}) {experts}E top{topk} hidden={hidden} inter={inter}",
            m("block_count")
        );
    } else {
        total_moe_layers = 48;
    }
    let slots = arg("slots", 512.0) as usize;
    let steps = arg("steps", 100.0) as usize;
    let warmup = arg("warmup", 50.0) as usize;
    let mode = arg_s("mode", "hybrid");
    let fractions: Vec<f64> = match mode.as_str() {
        "offload" => vec![1.0],
        "cpu" => vec![0.0],
        _ => arg_s(
            "fractions",
            &arg("fraction", qstar_fraction(arg("pcie", 6.4), arg("cpubw", 31.0))).to_string(),
        )
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect(),
    };
    let fraction = fractions[0]; // banner shows the first; the sweep prints per-run

    let gu_rows = 2 * inter; // gate+up fused, k=hidden
    let gu_bytes = q4_0::row_bytes(hidden) * gu_rows;
    let down_bytes = q4_0::row_bytes(inter) * hidden; // down: hidden rows, k=inter
    let eb = gu_bytes + down_bytes;
    println!(
        "mode={mode} fraction={fraction:.3} | {layers}L x {experts}E top{topk} \
         hidden={hidden} inter={inter} | expert={:.2}MB host_banks={:.2}GB cache={slots} slots={:.2}GB",
        eb as f64 / 1e6,
        (layers * experts * eb) as f64 / 1e9,
        (slots * eb) as f64 / 1e9
    );

    let ctx = CudaContext::new(0)?;
    let compute = ctx.default_stream();
    let copy_stream = ctx.new_stream()?;
    let gemv = ft_cuda::Q4Gemv::new(&ctx)?;

    // --- pinned host banks, one valid-looking q4_0 template per expert ---
    let mut template = vec![0u8; eb];
    let scale = half::f16::from_f32(0.01).to_le_bytes();
    for blk in template.chunks_exact_mut(q4_0::BLOCK_BYTES) {
        blk[..2].copy_from_slice(&scale);
        blk[2..].fill(0x5A);
    }
    anyhow::ensure!(layers > 0, "no MoE layers found in this model");
    let t_load = Instant::now();
    let mut host = ft_cuda::HostBanks::new(&ctx, layers * experts * eb)?;
    if let Some(g) = &gguf {
        let hs = host.as_mut_slice();
        let half_gu = gu_bytes / 2; // gate rows then up rows
        for (l, &blk_id) in moe_blocks.iter().enumerate() {
            let fused = format!("blk.{blk_id}.ffn_gate_up_exps.weight");
            let down = g.tensor_data(&format!("blk.{blk_id}.ffn_down_exps.weight"))?;
            assert_eq!(
                g.tensor(&format!("blk.{blk_id}.ffn_down_exps.weight"))?.ggml_type,
                2,
                "need a pure Q4_0 gguf (down bank)"
            );
            if let Ok(gu) = g.tensor_data(&fused) {
                assert_eq!(g.tensor(&fused)?.ggml_type, 2, "need a Q4_0 gguf");
                for e in 0..experts {
                    let dst = &mut hs[(l * experts + e) * eb..(l * experts + e + 1) * eb];
                    dst[..gu_bytes].copy_from_slice(&gu[e * gu_bytes..(e + 1) * gu_bytes]);
                    dst[gu_bytes..].copy_from_slice(&down[e * down_bytes..(e + 1) * down_bytes]);
                }
            } else {
                let gate = g.tensor_data(&format!("blk.{blk_id}.ffn_gate_exps.weight"))?;
                let up = g.tensor_data(&format!("blk.{blk_id}.ffn_up_exps.weight"))?;
                assert_eq!(g.tensor(&format!("blk.{blk_id}.ffn_gate_exps.weight"))?.ggml_type, 2, "need a Q4_0 gguf");
                for e in 0..experts {
                    let dst = &mut hs[(l * experts + e) * eb..(l * experts + e + 1) * eb];
                    dst[..half_gu].copy_from_slice(&gate[e * half_gu..(e + 1) * half_gu]);
                    dst[half_gu..gu_bytes].copy_from_slice(&up[e * half_gu..(e + 1) * half_gu]);
                    dst[gu_bytes..].copy_from_slice(&down[e * down_bytes..(e + 1) * down_bytes]);
                }
            }
        }
        println!("loaded real expert banks in {:.1}s", t_load.elapsed().as_secs_f64());
    } else {
        let hs = host.as_mut_slice();
        for e in 0..layers * experts {
            hs[e * eb..(e + 1) * eb].copy_from_slice(&template);
        }
    }

    // --- GPU slot cache + activation/output buffers ---
    let mut cache_buf = compute.alloc_zeros::<u8>(slots * eb)?;
    let x_hidden = compute.memcpy_stod(&vec![0.1f32; hidden])?;
    let x_inter = compute.memcpy_stod(&vec![0.1f32; inter])?;
    let grouped = arg("grouped", 1.0) != 0.0;
    let mut y_gu: Vec<_> = (0..topk)
        .map(|_| compute.alloc_zeros::<f32>(gu_rows))
        .collect::<Result<_, _>>()?;
    let mut y_down: Vec<_> = (0..topk)
        .map(|_| compute.alloc_zeros::<f32>(hidden))
        .collect::<Result<_, _>>()?;
    let mut act: Vec<_> = (0..topk)
        .map(|_| compute.alloc_zeros::<f32>(inter))
        .collect::<Result<_, _>>()?;
    // grouped path: contiguous [topk, *] buffers + device slot list
    let mut g_y_gu = compute.alloc_zeros::<f32>(topk * gu_rows)?;
    let mut g_act = compute.alloc_zeros::<f32>(topk * inter)?;
    let mut g_y_down = compute.alloc_zeros::<f32>(topk * hidden)?;
    let mut g_slots = compute.alloc_zeros::<i32>(topk)?;
    let mut g_x_q8 = compute.alloc_zeros::<u8>(hidden / 32 * ft_cuda::Q8_BLK)?;
    let mut g_act_q8 = compute.alloc_zeros::<u8>(topk * inter / 32 * ft_cuda::Q8_BLK)?;
    let mut out_h = compute.alloc_zeros::<f32>(hidden)?;
    let mut cpu_part_dev = compute.alloc_zeros::<f32>(hidden)?;
    let mut y_cpu_gu = vec![0f32; gu_rows];
    let mut y_cpu_down = vec![0f32; hidden];
    let mut act_cpu = vec![0f32; inter];
    let mut cpu_part = vec![0f32; hidden];
    let x_cpu_hidden = vec![0.1f32; hidden];

    // locality: probability that a routed expert repeats the previous token's
    // pick for this layer slot (real LLM routing reuses experts heavily;
    // locality=0 is the uniform worst case).
    let locality = arg("locality", 0.0);
    let mut prev: Vec<Vec<u32>> = vec![Vec::new(); layers];
    let mut rng = 0xC0FFEEu32;
    let mut rnd = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng
    };
    let mut route = move |layer: usize, experts: usize, topk: usize| -> Vec<u32> {
        let mut picked: Vec<u32> = Vec::with_capacity(topk);
        for i in 0..topk {
            let keep = !prev[layer].is_empty()
                && (rnd() as f64 / u32::MAX as f64) < locality
                && !picked.contains(&prev[layer][i]);
            let e = if keep {
                prev[layer][i]
            } else {
                loop {
                    let c = (rnd() as usize % experts) as u32;
                    if !picked.contains(&c) {
                        break c;
                    }
                }
            };
            picked.push(e);
        }
        prev[layer] = picked.clone();
        picked
    };

    for &fraction in &fractions {
    let mut cache = SlotCache::new(slots as u32);
    let mut fetched_total = 0u64;
    let mut cpu_total = 0u64;
    let mut t0 = Instant::now();
    for step in 0..warmup + steps {
        if step == warmup {
            // steady state reached: reset counters and clock
            cache.hits_total = 0;
            cache.misses_total = 0;
            fetched_total = 0;
            cpu_total = 0;
            t0 = Instant::now();
        }
        for layer in 0..layers {
            let routed = route(layer, experts, topk);
            let lk = cache.lookup(layer as u32, &routed);

            let miss_ids: Vec<u32> = lk.misses.iter().map(|&(e, _)| e).collect();
            let (fetch_ids, cpu_ids) = split_misses(&miss_ids, fraction);
            let slot_of = |e: u32| lk.misses.iter().find(|&&(me, _)| me == e).unwrap().1;

            // 1) enqueue H2D for fetched misses on the copy stream
            for &e in &fetch_ids {
                let slot = slot_of(e) as usize;
                let off = (layer * experts + e as usize) * eb;
                let hs = host.as_slice();
                let mut dst = cache_buf.slice_mut(slot * eb..(slot + 1) * eb);
                copy_stream.memcpy_htod(&hs[off..off + eb], &mut dst)?;
            }
            let copied = copy_stream.record_event(None)?;
            compute.wait(&copied)?;

            // 2) GPU experts, full FFN chain: gate_up -> silu*up -> down ->
            //    weighted accumulate (async on compute stream)
            let ew = 1.0f32 / topk as f32; // uniform routing weights
            let gpu_experts: Vec<u32> = lk
                .hits
                .iter()
                .map(|&(_, s)| s)
                .chain(fetch_ids.iter().map(|&e| slot_of(e)))
                .collect();
            let n = gpu_experts.len();
            if grouped && n > 0 {
                // FreeToken-style fused layer: 4 launches + one tiny H2D
                // regardless of expert count.
                let slots_i32: Vec<i32> = gpu_experts.iter().map(|&s| s as i32).collect();
                compute.memcpy_htod(&slots_i32, &mut g_slots.slice_mut(0..n))?;
                gemv.quantize_q8(&compute, &x_hidden, 0, &mut g_x_q8, hidden, 1)?;
                gemv.gemv_grouped_q8(&compute, &cache_buf, eb, 0, &g_slots, n, &g_x_q8, 0, &mut g_y_gu, gu_rows, gu_rows, hidden)?;
                gemv.silu_mul_grouped(&compute, &g_y_gu, &mut g_act, inter, n)?;
                gemv.quantize_q8(&compute, &g_act, inter, &mut g_act_q8, inter, n)?;
                gemv.gemv_grouped_q8(&compute, &cache_buf, eb, gu_bytes, &g_slots, n, &g_act_q8, inter / 32, &mut g_y_down, hidden, hidden, inter)?;
                gemv.reduce_weighted(&compute, &g_y_down, &mut out_h, ew, hidden, n)?;
            } else {
                for (i, &slot) in gpu_experts.iter().enumerate() {
                    let s = slot as usize;
                    let wgu = cache_buf.slice(s * eb..s * eb + gu_bytes);
                    let wdn = cache_buf.slice(s * eb + gu_bytes..(s + 1) * eb);
                    gemv.launch(&compute, &wgu, &x_hidden, &mut y_gu[i], gu_rows, hidden)?;
                    gemv.silu_mul(&compute, &y_gu[i], &mut act[i], inter)?;
                    gemv.launch(&compute, &wdn, &act[i], &mut y_down[i], hidden, inter)?;
                    gemv.axpy(&compute, &y_down[i], &mut out_h, ew, hidden)?;
                }
            }

            // 3) CPU computes the remaining misses while the GPU works
            //    (activation quantized once per layer, reused across experts)
            if !cpu_ids.is_empty() {
                let x8 = q4_0::Q8Vec::quantize(&x_cpu_hidden);
                cpu_part.fill(0.0);
                for &e in &cpu_ids {
                    let off = (layer * experts + e as usize) * eb;
                    let hs = host.as_slice();
                    q4_0::gemv_q8(&hs[off..off + gu_bytes], &x8, &mut y_cpu_gu);
                    for i in 0..inter {
                        let g = y_cpu_gu[i];
                        act_cpu[i] = g / (1.0 + (-g).exp()) * y_cpu_gu[i + inter];
                    }
                    q4_0::gemv(&hs[off + gu_bytes..off + eb], &act_cpu, &mut y_cpu_down);
                    for i in 0..hidden {
                        cpu_part[i] += ew * y_cpu_down[i];
                    }
                }
                // merge the CPU partial into the layer output (8KB H2D + axpy)
                compute.memcpy_htod(&cpu_part, &mut cpu_part_dev)?;
                gemv.axpy(&compute, &cpu_part_dev, &mut out_h, 1.0, hidden)?;
            }

            compute.synchronize()?; // layer boundary (attention would sit here)
            fetched_total += fetch_ids.len() as u64;
            cpu_total += cpu_ids.len() as u64;
        }
    }
    let dt = t0.elapsed().as_secs_f64();

    let ms_step = dt * 1000.0 / steps as f64;
    let per_layer = ms_step / layers as f64;
    println!(
        "fraction={fraction:.2} steps={steps}: {ms_step:.2} ms/step ({per_layer:.3} ms/MoE-layer) | \
         hit_rate={:.1}% fetched/step={:.2} cpu/step={:.2} | {}MoE-L: {:.1} ms/tok -> {:.1} tok/s (MoE FFN only)",
        cache.hit_rate() * 100.0,
        fetched_total as f64 / (steps * layers) as f64,
        cpu_total as f64 / (steps * layers) as f64,
        total_moe_layers,
        per_layer * total_moe_layers as f64,
        1000.0 / (per_layer * total_moe_layers as f64)
    );
    } // fraction sweep
    Ok(())
}
