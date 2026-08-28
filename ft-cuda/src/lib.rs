//! ft-cuda: GPU side of the offload engine — the q4_0 dequant-GEMV kernel
//! (PTX embedded at build time) plus a thin launch wrapper.

use anyhow::{bail, Result};
use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Pinned host memory that stays CPU-cached: ordinary pages registered with
/// `cuMemHostRegister` (what torch's `pin_memory()` does). cudarc's own
/// `alloc_pinned` uses CU_MEMHOSTALLOC_WRITECOMBINED, whose uncached CPU
/// reads are ~100x slow — fatal for the hybrid path, where the CPU executor
/// GEMVs directly over the same banks the GPU fetches from.
pub struct HostBanks {
    buf: Box<[u8]>,
    _ctx: Arc<CudaContext>,
}

impl HostBanks {
    pub fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        ctx.bind_to_thread()?;
        let mut buf = vec![0u8; len].into_boxed_slice();
        let rc = unsafe {
            sys::cuMemHostRegister_v2(buf.as_mut_ptr() as *mut core::ffi::c_void, len, 0)
        };
        if rc != sys::CUresult::CUDA_SUCCESS {
            bail!("cuMemHostRegister failed: {rc:?}");
        }
        Ok(Self { buf, _ctx: ctx.clone() })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    /// Device-visible address of the registered host memory (UVA): kernels
    /// can read expert banks directly over PCIe without an explicit fetch.
    pub fn device_ptr(&self) -> Result<u64> {
        let mut dptr: sys::CUdeviceptr = 0;
        let rc = unsafe {
            sys::cuMemHostGetDevicePointer_v2(
                &mut dptr,
                self.buf.as_ptr() as *mut core::ffi::c_void,
                0,
            )
        };
        if rc != sys::CUresult::CUDA_SUCCESS {
            bail!("cuMemHostGetDevicePointer failed: {rc:?}");
        }
        Ok(dptr as u64)
    }
}

impl Drop for HostBanks {
    fn drop(&mut self) {
        unsafe {
            sys::cuMemHostUnregister(self.buf.as_mut_ptr() as *mut core::ffi::c_void);
        }
    }
}

pub struct Q4Gemv {
    func: CudaFunction,
    silu_mul: CudaFunction,
    axpy: CudaFunction,
    grouped: CudaFunction,
    silu_grouped: CudaFunction,
    reduce: CudaFunction,
    q6k: CudaFunction,
    gelu_grouped: CudaFunction,
    reduce_ew: CudaFunction,
    quant_q8: CudaFunction,
    grouped_v3: CudaFunction,
    q6k_q8: CudaFunction,
    attn: CudaFunction,
    grouped_v3_idx: CudaFunction,
    reduce_pairs: CudaFunction,
    rmsnorm_rows: CudaFunction,
    add_rows: CudaFunction,
    rmsnorm_heads: CudaFunction,
    rope_heads: CudaFunction,
    kv_append: CudaFunction,
    attn_batch: CudaFunction,
    dual_combine: CudaFunction,
    gemv_f32: CudaFunction,
    gather: CudaFunction,
    grouped_v3_ptr: CudaFunction,
    sample: CudaFunction,
    topk: CudaFunction,
    admit: CudaFunction,
    promote: CudaFunction,
    q4r_idx: CudaFunction,
    q4r_ptr: CudaFunction,
}

/// bytes per 32-element q8 activation block (f32 d, f32 s, 32x i8)
pub const Q8_BLK: usize = 40;

impl Q4Gemv {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/q4.ptx"));
        let module = ctx.load_module(Ptx::from_src(ptx))?;
        let func = module.load_function("gemv_q4_0")?;
        let silu_mul = module.load_function("silu_mul")?;
        let axpy = module.load_function("axpy")?;
        let grouped = module.load_function("gemv_q4_0_grouped_v2")?;
        let silu_grouped = module.load_function("silu_mul_grouped")?;
        let reduce = module.load_function("reduce_weighted")?;
        let q6k = module.load_function("gemv_q6_k")?;
        let gelu_grouped = module.load_function("gelu_mul_grouped")?;
        let reduce_ew = module.load_function("reduce_expert_weighted")?;
        let quant_q8 = module.load_function("quantize_q8_grouped")?;
        let grouped_v3 = module.load_function("gemv_q4_0_grouped_v3")?;
        let q6k_q8 = module.load_function("gemv_q6_k_q8")?;
        let attn = module.load_function("attn_decode")?;
        let grouped_v3_idx = module.load_function("gemv_q4_0_grouped_v3_idx")?;
        let reduce_pairs = module.load_function("reduce_pairs_weighted")?;
        let rmsnorm_rows = module.load_function("rmsnorm_rows")?;
        let add_rows = module.load_function("add_rows")?;
        let rmsnorm_heads = module.load_function("rmsnorm_heads")?;
        let rope_heads = module.load_function("rope_heads")?;
        let kv_append = module.load_function("kv_append")?;
        let attn_batch = module.load_function("attn_decode_batch")?;
        let dual_combine = module.load_function("dual_combine_rows")?;
        let gemv_f32 = module.load_function("gemv_f32_rows")?;
        let gather = module.load_function("gather_rows")?;
        let grouped_v3_ptr = module.load_function("gemv_q4_0_grouped_v3_ptr")?;
        let sample = module.load_function("sample_tokens")?;
        let topk = module.load_function("topk_router")?;
        let admit = module.load_function("lru_admit")?;
        let promote = module.load_function("promote_experts")?;
        let q4r_idx = module.load_function("gemv_q4r_grouped_idx")?;
        let q4r_ptr = module.load_function("gemv_q4r_grouped_ptr")?;
        Ok(Self { func, silu_mul, axpy, grouped, silu_grouped, reduce, q6k, gelu_grouped, reduce_ew, quant_q8, grouped_v3, q6k_q8, attn, grouped_v3_idx, reduce_pairs, rmsnorm_rows, add_rows, rmsnorm_heads, rope_heads, kv_append, attn_batch, dual_combine, gemv_f32, gather, grouped_v3_ptr, sample, topk, admit, promote, q4r_idx, q4r_ptr })
    }

    /// One launch computing y[e] = W_e x_e for every routed expert of a layer.
    /// `bank_off` selects the bank inside each expert's slot (0 = gate_up,
    /// gu_bytes = down); `x_stride` = 0 shares one activation across experts.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_grouped(
        &self,
        stream: &Arc<CudaStream>,
        cache: &CudaSlice<u8>,
        expert_bytes: usize,
        bank_off: usize,
        slots: &CudaSlice<i32>,
        n_experts: usize,
        x: &CudaSlice<f32>,
        x_stride: usize,
        y: &mut CudaSlice<f32>,
        y_stride: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        assert_eq!(k % 32, 0);
        assert!(slots.len() >= n_experts);
        assert!(y.len() >= n_experts * y_stride);
        let (eb, off) = (expert_bytes as u64, bank_off as u64);
        let (xs, ys) = (x_stride as i32, y_stride as i32);
        let (nr, ki) = (n_rows as i32, k as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(4) as u32, n_experts as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (k * 4) as u32,
        };
        let mut lb = stream.launch_builder(&self.grouped);
        lb.arg(cache).arg(&eb).arg(&off).arg(slots).arg(x).arg(&xs).arg(y).arg(&ys).arg(&nr).arg(&ki);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// Decode attention, one block per query head. K/V are f16 stored as u16
    /// buffers [max_seq, kv_dim]; smem = (kv_end-kv_start)*4 bytes of scores.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        stream: &Arc<CudaStream>,
        kc: &CudaSlice<u16>,
        vc: &CudaSlice<u16>,
        q: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        kv_start: usize,
        kv_end: usize,
        kv_dim: usize,
        hd: usize,
        group: usize,
        n_heads: usize,
    ) -> Result<()> {
        let n = kv_end - kv_start;
        assert!(n > 0);
        let smem = n * 4;
        assert!(smem <= 48 * 1024, "context too long for smem scores");
        let (a, b, c, d, e) = (
            kv_start as i32,
            kv_end as i32,
            kv_dim as i32,
            hd as i32,
            group as i32,
        );
        let cfg = LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        let mut lb = stream.launch_builder(&self.attn);
        lb.arg(kc).arg(vc).arg(q).arg(out).arg(&a).arg(&b).arg(&c).arg(&d).arg(&e);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }


    /// y[e] = rmsnorm(x[e]) * w over n batched rows. w=None -> unscaled.
    pub fn rmsnorm_rows_dev(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        w: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        dim: usize,
        eps: f32,
        n: usize,
    ) -> Result<()> {
        let (d, hw) = (dim as i32, w.is_some() as i32);
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * 4,
        };
        let wref = w.unwrap_or(x);
        let mut lb = stream.launch_builder(&self.rmsnorm_rows);
        lb.arg(x).arg(wref).arg(y).arg(&d).arg(&eps).arg(&hw);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// x[e] += a[e] over n rows of `dim`.
    pub fn add_rows_dev(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        dim: usize,
        n: usize,
    ) -> Result<()> {
        let d = dim as i32;
        let cfg = LaunchConfig {
            grid_dim: (dim.div_ceil(256) as u32, n as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.add_rows);
        lb.arg(x).arg(a).arg(&d);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// per-head rmsnorm on a section of batched qkv rows.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_heads_dev(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaViewMut<'_, f32>,
        w: &CudaSlice<f32>,
        has_w: bool,
        hd: usize,
        eps: f32,
        seq_stride: usize,
        heads_per_seq: usize,
        n_seqs: usize,
    ) -> Result<()> {
        let (h, hw, ss, hp) = (hd as i32, has_w as i32, seq_stride as i32, heads_per_seq as i32);
        let cfg = LaunchConfig {
            grid_dim: ((n_seqs * heads_per_seq) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * 4,
        };
        let mut lb = stream.launch_builder(&self.rmsnorm_heads);
        lb.arg(x).arg(w).arg(&h).arg(&eps).arg(&hw).arg(&ss).arg(&hp);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// NeoX rope on a section of batched qkv rows.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_heads_dev(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaViewMut<'_, f32>,
        inv_freq: &CudaSlice<f32>,
        pos_arr: &CudaSlice<i32>,
        hd: usize,
        heads_per_seq: usize,
        seq_stride: usize,
        n_seqs: usize,
    ) -> Result<()> {
        let (h, hp, ss) = (hd as i32, heads_per_seq as i32, seq_stride as i32);
        let cfg = LaunchConfig {
            grid_dim: ((n_seqs * heads_per_seq) as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.rope_heads);
        lb.arg(x).arg(inv_freq).arg(pos_arr).arg(&h).arg(&hp).arg(&ss);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// append k or v sections into the pooled f16 KV cache.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_dev(
        &self,
        stream: &Arc<CudaStream>,
        qkv: &CudaSlice<f32>,
        pool: &mut CudaSlice<u16>,
        slot_arr: &CudaSlice<i32>,
        pos_arr: &CudaSlice<i32>,
        sec_off: usize,
        seq_stride: usize,
        kv_dim: usize,
        max_seq: usize,
        n_seqs: usize,
    ) -> Result<()> {
        let (so, ss, kd, ms) = (sec_off as i32, seq_stride as i32, kv_dim as i32, max_seq as i32);
        let cfg = LaunchConfig {
            grid_dim: (kv_dim.div_ceil(256) as u32, n_seqs as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.kv_append);
        lb.arg(qkv).arg(pool).arg(slot_arr).arg(pos_arr).arg(&so).arg(&ss).arg(&kd).arg(&ms);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// batched decode attention over the pooled KV cache.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_batch_dev(
        &self,
        stream: &Arc<CudaStream>,
        kpool: &CudaSlice<u16>,
        vpool: &CudaSlice<u16>,
        q: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        slot_arr: &CudaSlice<i32>,
        pos_arr: &CudaSlice<i32>,
        window: usize,
        kv_dim: usize,
        hd: usize,
        group: usize,
        n_heads: usize,
        max_seq: usize,
        n_seqs: usize,
        max_ctx: usize,
    ) -> Result<()> {
        let smem = max_ctx * 4;
        assert!(smem <= 48 * 1024);
        let (w, kd, h, g, nh, ms) = (
            window as i32,
            kv_dim as i32,
            hd as i32,
            group as i32,
            n_heads as i32,
            max_seq as i32,
        );
        let cfg = LaunchConfig {
            grid_dim: (n_heads as u32, n_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        let mut lb = stream.launch_builder(&self.attn_batch);
        lb.arg(kpool).arg(vpool).arg(q).arg(out).arg(slot_arr).arg(pos_arr)
            .arg(&w).arg(&kd).arg(&h).arg(&g).arg(&nh).arg(&ms);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// device top-k router: ids + softmaxed rescaled weights per sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn topk_router_dev(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaSlice<f32>,
        expert_scale: &CudaSlice<f32>,
        ids: &mut CudaSlice<i32>,
        wts: &mut CudaSlice<f32>,
        e_count: usize,
        k: usize,
        n_seqs: usize,
    ) -> Result<()> {
        assert!(e_count <= 1024 && k <= 32);
        let (e, kk) = (e_count as i32, k as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_seqs as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.topk);
        lb.arg(logits).arg(expert_scale).arg(ids).arg(wts).arg(&e).arg(&kk);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// device-side LRU admission over the expert slot cache.
    #[allow(clippy::too_many_arguments)]
    pub fn lru_admit_dev(
        &self,
        stream: &Arc<CudaStream>,
        ids: &CudaSlice<i32>,
        layer: usize,
        e_count: usize,
        n_slots: usize,
        cache_base: u64,
        banks_base: u64,
        expert_bytes: usize,
        map: &mut CudaSlice<i32>,
        slot_key: &mut CudaSlice<i32>,
        slot_last: &mut CudaSlice<u32>,
        clock: &mut CudaSlice<u32>,
        bases: &mut CudaSlice<u64>,
        promote_src: &mut CudaSlice<i32>,
        promote_dst: &mut CudaSlice<i32>,
        n_entries: usize,
    ) -> Result<()> {
        let (l, e, ns, eb, ne) = (
            layer as i32,
            e_count as i32,
            n_slots as i32,
            expert_bytes as u64,
            n_entries as i32,
        );
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.admit);
        lb.arg(ids).arg(&l).arg(&e).arg(&ns).arg(&cache_base).arg(&banks_base).arg(&eb)
            .arg(map).arg(slot_key).arg(slot_last).arg(clock)
            .arg(bases).arg(promote_src).arg(promote_dst).arg(&ne);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// device-side promotion of missed experts into their slots (UVA copy).
    pub fn promote_experts_dev(
        &self,
        stream: &Arc<CudaStream>,
        promote_src: &CudaSlice<i32>,
        promote_dst: &CudaSlice<i32>,
        banks_base: u64,
        cache_base: u64,
        expert_bytes: usize,
        n_entries: usize,
    ) -> Result<()> {
        let eb = expert_bytes as u64;
        let cfg = LaunchConfig {
            grid_dim: (96, n_entries as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.promote);
        lb.arg(promote_src).arg(promote_dst).arg(&banks_base).arg(&cache_base).arg(&eb);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// gemma dual-rmsnorm combine + residual + scalar, batched rows.
    #[allow(clippy::too_many_arguments)]
    pub fn dual_combine_dev(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaSlice<f32>,
        shared_y: &CudaSlice<f32>,
        routed_y: &CudaSlice<f32>,
        w1: &CudaSlice<f32>,
        w2: &CudaSlice<f32>,
        w3: &CudaSlice<f32>,
        scalar: f32,
        dim: usize,
        eps: f32,
        n: usize,
    ) -> Result<()> {
        let d = dim as i32;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: ((128 + dim) * 4) as u32,
        };
        let mut lb = stream.launch_builder(&self.dual_combine);
        lb.arg(x).arg(shared_y).arg(routed_y).arg(w1).arg(w2).arg(w3).arg(&scalar).arg(&d).arg(&eps);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// v3 grouped GEMV over per-entry base POINTERS (u64): each entry's
    /// weights live either in the VRAM slot cache or directly in pinned host
    /// banks (UVA) — misses stream over PCIe inside the kernel, no explicit
    /// fetch or sync required.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_grouped_q8_ptr(
        &self,
        stream: &Arc<CudaStream>,
        bases: &CudaSlice<u64>,
        bank_off: usize,
        x_idx: &CudaSlice<i32>,
        n_entries: usize,
        q8: &CudaSlice<u8>,
        q8_stride_blocks: usize,
        y: &mut CudaSlice<f32>,
        y_stride: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        assert_eq!(k % 32, 0);
        let nblocks = k / 32;
        assert!(bases.len() >= n_entries && x_idx.len() >= n_entries);
        assert!(y.len() >= n_entries * y_stride);
        let off = bank_off as u64;
        let (qs, ys) = (q8_stride_blocks as i32, y_stride as i32);
        let (nr, ki) = (n_rows as i32, k as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(4) as u32, n_entries as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (nblocks * Q8_BLK) as u32,
        };
        let mut lb = stream.launch_builder(&self.grouped_v3_ptr);
        lb.arg(bases).arg(&off).arg(x_idx).arg(q8).arg(&qs).arg(y).arg(&ys).arg(&nr).arg(&ki);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// fused greedy/Gumbel sampling over batched logits.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_tokens_dev(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaSlice<f32>,
        temps: &CudaSlice<f32>,
        rng: &mut CudaSlice<u64>,
        out: &mut CudaSlice<i32>,
        vocab: usize,
        cap: f32,
        n_seqs: usize,
    ) -> Result<()> {
        let v = vocab as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_seqs as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.sample);
        lb.arg(logits).arg(temps).arg(rng).arg(out).arg(&v).arg(&cap);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// strided row gather (e.g. q sections of qkv rows -> contiguous).
    pub fn gather_rows_dev(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        src_stride: usize,
        src_off: usize,
        row_len: usize,
        n_seqs: usize,
    ) -> Result<()> {
        let (ss, so, rl) = (src_stride as i32, src_off as i32, row_len as i32);
        let cfg = LaunchConfig {
            grid_dim: (row_len.div_ceil(256) as u32, n_seqs as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.gather);
        lb.arg(src).arg(dst).arg(&ss).arg(&so).arg(&rl);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// f32 router GEMV: y[s][r] = dot(W[r], x[s]).
    pub fn gemv_f32_rows_dev(
        &self,
        stream: &Arc<CudaStream>,
        w: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        dim: usize,
        n_rows: usize,
        n_seqs: usize,
    ) -> Result<()> {
        let (d, nr) = (dim as i32, n_rows as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, n_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * 4,
        };
        let mut lb = stream.launch_builder(&self.gemv_f32);
        lb.arg(w).arg(x).arg(y).arg(&d).arg(&nr);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }


    /// q4r (repacked, coalesced) grouped GEMV with activation indirection.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4r_idx(
        &self,
        stream: &Arc<CudaStream>,
        cache: &CudaSlice<u8>,
        expert_bytes: usize,
        bank_off: usize,
        slots: &CudaSlice<i32>,
        x_idx: &CudaSlice<i32>,
        n_entries: usize,
        q8: &CudaSlice<u8>,
        q8_stride_blocks: usize,
        y: &mut CudaSlice<f32>,
        y_stride: usize,
        n_rows: usize,
        k: usize,
        row_bytes_r: usize,
        qs_off: usize,
    ) -> Result<()> {
        assert_eq!(k % 32, 0);
        let nblocks = k / 32;
        let (eb, off) = (expert_bytes as u64, bank_off as u64);
        let (qs, ys, nr, ki) = (q8_stride_blocks as i32, y_stride as i32, n_rows as i32, k as i32);
        let (rb, qo) = (row_bytes_r as i32, qs_off as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(8) as u32, n_entries as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (nblocks * Q8_BLK) as u32,
        };
        let mut lb = stream.launch_builder(&self.q4r_idx);
        lb.arg(cache).arg(&eb).arg(&off).arg(slots).arg(x_idx).arg(q8).arg(&qs)
            .arg(y).arg(&ys).arg(&nr).arg(&ki).arg(&rb).arg(&qo);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// q4r grouped GEMV over per-entry base pointers (UVA-capable).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4r_ptr(
        &self,
        stream: &Arc<CudaStream>,
        bases: &CudaSlice<u64>,
        bank_off: usize,
        x_idx: &CudaSlice<i32>,
        n_entries: usize,
        q8: &CudaSlice<u8>,
        q8_stride_blocks: usize,
        y: &mut CudaSlice<f32>,
        y_stride: usize,
        n_rows: usize,
        k: usize,
        row_bytes_r: usize,
        qs_off: usize,
    ) -> Result<()> {
        assert_eq!(k % 32, 0);
        let nblocks = k / 32;
        let off = bank_off as u64;
        let (qs, ys, nr, ki) = (q8_stride_blocks as i32, y_stride as i32, n_rows as i32, k as i32);
        let (rb, qo) = (row_bytes_r as i32, qs_off as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(8) as u32, n_entries as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (nblocks * Q8_BLK) as u32,
        };
        let mut lb = stream.launch_builder(&self.q4r_ptr);
        lb.arg(bases).arg(&off).arg(x_idx).arg(q8).arg(&qs)
            .arg(y).arg(&ys).arg(&nr).arg(&ki).arg(&rb).arg(&qo);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// v3 grouped GEMV with activation indirection: entry e reads q8 row
    /// x_idx[e]. For batched MoE (seq,expert) pairs and batched dense ops.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_grouped_q8_idx(
        &self,
        stream: &Arc<CudaStream>,
        cache: &CudaSlice<u8>,
        expert_bytes: usize,
        bank_off: usize,
        slots: &CudaSlice<i32>,
        x_idx: &CudaSlice<i32>,
        n_entries: usize,
        q8: &CudaSlice<u8>,
        q8_stride_blocks: usize,
        y: &mut CudaSlice<f32>,
        y_stride: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        assert_eq!(k % 32, 0);
        let nblocks = k / 32;
        assert!(slots.len() >= n_entries && x_idx.len() >= n_entries);
        assert!(y.len() >= n_entries * y_stride);
        let (eb, off) = (expert_bytes as u64, bank_off as u64);
        let (qs, ys) = (q8_stride_blocks as i32, y_stride as i32);
        let (nr, ki) = (n_rows as i32, k as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(4) as u32, n_entries as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (nblocks * Q8_BLK) as u32,
        };
        let mut lb = stream.launch_builder(&self.grouped_v3_idx);
        lb.arg(cache).arg(&eb).arg(&off).arg(slots).arg(x_idx).arg(q8).arg(&qs).arg(y).arg(&ys).arg(&nr).arg(&ki);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// out[s] += sum of pair outputs with seq_of[p]==s, weighted by wts[p].
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_pairs_weighted(
        &self,
        stream: &Arc<CudaStream>,
        y: &CudaSlice<f32>,
        wts: &CudaSlice<f32>,
        seq_of: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        hidden: usize,
        n_pairs: usize,
        n_seqs: usize,
    ) -> Result<()> {
        let (h, np) = (hidden as i32, n_pairs as i32);
        let cfg = LaunchConfig {
            grid_dim: (hidden.div_ceil(256) as u32, n_seqs as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.reduce_pairs);
        lb.arg(y).arg(wts).arg(seq_of).arg(out).arg(&h).arg(&np);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// attn_decode over view q/out (sliced per layer geometry).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_view(
        &self,
        stream: &Arc<CudaStream>,
        kc: &CudaSlice<u16>,
        vc: &CudaSlice<u16>,
        q: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, f32>,
        kv_start: usize,
        kv_end: usize,
        kv_dim: usize,
        hd: usize,
        group: usize,
        n_heads: usize,
    ) -> Result<()> {
        let n = kv_end - kv_start;
        assert!(n > 0);
        let smem = n * 4;
        assert!(smem <= 48 * 1024, "context too long for smem scores");
        let (a, b, c, d, e) = (
            kv_start as i32,
            kv_end as i32,
            kv_dim as i32,
            hd as i32,
            group as i32,
        );
        let cfg = LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: smem as u32,
        };
        let mut lb = stream.launch_builder(&self.attn);
        lb.arg(kc).arg(vc).arg(q).arg(out).arg(&a).arg(&b).arg(&c).arg(&d).arg(&e);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// quantize_q8 over a device view; x_stride separates per-entry rows
    /// (0 shares one vector across entries).
    pub fn quantize_q8_view(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaView<'_, f32>,
        x_stride: usize,
        q8: &mut CudaSlice<u8>,
        k: usize,
        n_experts: usize,
    ) -> Result<()> {
        assert_eq!(k % 32, 0);
        let nblocks = k / 32;
        assert!(q8.len() >= n_experts * nblocks * Q8_BLK);
        let total_warps = n_experts * nblocks;
        let (xs, nb, ne) = (x_stride as i32, nblocks as i32, n_experts as i32);
        let cfg = LaunchConfig {
            grid_dim: ((total_warps * 32).div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.quant_q8);
        lb.arg(x).arg(&xs).arg(q8).arg(&nb).arg(&ne);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// Quantize activations to q8 blocks on-GPU: x [n_experts, x_stride] ->
    /// q8 [n_experts, (k/32)*40 bytes]. x_stride = 0 shares one vector.
    pub fn quantize_q8(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        x_stride: usize,
        q8: &mut CudaSlice<u8>,
        k: usize,
        n_experts: usize,
    ) -> Result<()> {
        assert_eq!(k % 32, 0);
        let nblocks = k / 32;
        assert!(q8.len() >= n_experts * nblocks * Q8_BLK);
        let total_warps = n_experts * nblocks;
        let (xs, nb, ne) = (x_stride as i32, nblocks as i32, n_experts as i32);
        let cfg = LaunchConfig {
            grid_dim: ((total_warps * 32).div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.quant_q8);
        lb.arg(x).arg(&xs).arg(q8).arg(&nb).arg(&ne);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// v3 grouped GEMV over q8-quantized activations (dp4a integer path).
    /// q8_stride_blocks = 0 shares one activation across experts.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_grouped_q8(
        &self,
        stream: &Arc<CudaStream>,
        cache: &CudaSlice<u8>,
        expert_bytes: usize,
        bank_off: usize,
        slots: &CudaSlice<i32>,
        n_experts: usize,
        q8: &CudaSlice<u8>,
        q8_stride_blocks: usize,
        y: &mut CudaSlice<f32>,
        y_stride: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        assert_eq!(k % 32, 0);
        let nblocks = k / 32;
        assert!(slots.len() >= n_experts);
        assert!(y.len() >= n_experts * y_stride);
        let (eb, off) = (expert_bytes as u64, bank_off as u64);
        let (qs, ys) = (q8_stride_blocks as i32, y_stride as i32);
        let (nr, ki) = (n_rows as i32, k as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(4) as u32, n_experts as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (nblocks * Q8_BLK) as u32,
        };
        let mut lb = stream.launch_builder(&self.grouped_v3);
        lb.arg(cache).arg(&eb).arg(&off).arg(slots).arg(q8).arg(&qs).arg(y).arg(&ys).arg(&nr).arg(&ki);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// q6_k GEMV over q8-quantized activations (dp4a lm_head fast path).
    pub fn gemv_q6k_q8(
        &self,
        stream: &Arc<CudaStream>,
        w: &CudaSlice<u8>,
        q8: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        assert_eq!(k % 256, 0);
        assert_eq!(w.len(), n_rows * (k / 256) * 210);
        assert!(q8.len() >= k / 32 * Q8_BLK);
        let (nr, ki) = (n_rows as i32, k as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(16) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (k / 32 * Q8_BLK) as u32,
        };
        let mut lb = stream.launch_builder(&self.q6k_q8);
        lb.arg(w).arg(q8).arg(y).arg(&nr).arg(&ki);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// y = W x for a q6_k matrix (n_rows, k) — the tied lm_head. Smem = k floats.
    pub fn gemv_q6k(
        &self,
        stream: &Arc<CudaStream>,
        w: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        assert_eq!(k % 256, 0);
        assert_eq!(w.len(), n_rows * (k / 256) * 210);
        let (nr, ki) = (n_rows as i32, k as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (k * 4) as u32,
        };
        let mut lb = stream.launch_builder(&self.q6k);
        lb.arg(w).arg(x).arg(y).arg(&nr).arg(&ki);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// out[e] = gelu_tanh(gate) * up for all experts in one launch (gemma).
    pub fn gelu_mul_grouped(
        &self,
        stream: &Arc<CudaStream>,
        gu: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        inter: usize,
        n_experts: usize,
    ) -> Result<()> {
        let (ni, ne) = (inter as i32, n_experts as i32);
        let total = inter * n_experts;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.gelu_grouped);
        lb.arg(gu).arg(out).arg(&ni).arg(&ne);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// out += sum_e wts[e] * y[e] — router-weighted combine.
    pub fn reduce_expert_weighted(
        &self,
        stream: &Arc<CudaStream>,
        y: &CudaSlice<f32>,
        wts: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        hidden: usize,
        n_experts: usize,
    ) -> Result<()> {
        let (h, ne) = (hidden as i32, n_experts as i32);
        let cfg = LaunchConfig {
            grid_dim: (hidden.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.reduce_ew);
        lb.arg(y).arg(wts).arg(out).arg(&h).arg(&ne);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// out[e] = silu(gate) * up for all experts in one launch.
    pub fn silu_mul_grouped(
        &self,
        stream: &Arc<CudaStream>,
        gu: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        inter: usize,
        n_experts: usize,
    ) -> Result<()> {
        let (ni, ne) = (inter as i32, n_experts as i32);
        let total = inter * n_experts;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.silu_grouped);
        lb.arg(gu).arg(out).arg(&ni).arg(&ne);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// out += a * sum_e y[e] in one launch.
    pub fn reduce_weighted(
        &self,
        stream: &Arc<CudaStream>,
        y: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        a: f32,
        hidden: usize,
        n_experts: usize,
    ) -> Result<()> {
        let (h, ne) = (hidden as i32, n_experts as i32);
        let cfg = LaunchConfig {
            grid_dim: (hidden.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.reduce);
        lb.arg(y).arg(out).arg(&a).arg(&h).arg(&ne);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// out[0..inter] = silu(gu[0..inter]) * gu[inter..2*inter]
    pub fn silu_mul(
        &self,
        stream: &Arc<CudaStream>,
        gu: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        inter: usize,
    ) -> Result<()> {
        let n = inter as i32;
        let cfg = LaunchConfig {
            grid_dim: (inter.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.silu_mul);
        lb.arg(gu).arg(out).arg(&n);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// y += a * x
    pub fn axpy(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        a: f32,
        n: usize,
    ) -> Result<()> {
        let ni = n as i32;
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.axpy);
        lb.arg(x).arg(y).arg(&a).arg(&ni);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// y[0..n_rows] = W x, where `w` is a q4_0 view of n_rows * (k/32*18)
    /// bytes. Async on `stream`.
    pub fn launch(
        &self,
        stream: &Arc<CudaStream>,
        w: &CudaView<'_, u8>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        n_rows: usize,
        k: usize,
    ) -> Result<()> {
        assert_eq!(k % 32, 0);
        assert_eq!(w.len(), n_rows * (k / 32) * 18);
        let n = n_rows as i32;
        let ki = k as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut lb = stream.launch_builder(&self.func);
        lb.arg(w).arg(x).arg(y).arg(&n).arg(&ki);
        unsafe { lb.launch(cfg)? };
        Ok(())
    }
}
