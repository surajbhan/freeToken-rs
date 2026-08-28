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
        Ok(Self { func, silu_mul, axpy, grouped, silu_grouped, reduce, q6k, gelu_grouped, reduce_ew, quant_q8, grouped_v3, q6k_q8, attn })
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

    /// quantize_q8 over a device view (single shared vector).
    pub fn quantize_q8_view(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaView<'_, f32>,
        q8: &mut CudaSlice<u8>,
        k: usize,
        n_experts: usize,
    ) -> Result<()> {
        assert_eq!(k % 32, 0);
        let nblocks = k / 32;
        assert!(q8.len() >= n_experts * nblocks * Q8_BLK);
        let total_warps = n_experts * nblocks;
        let (xs, nb, ne) = (0i32, nblocks as i32, n_experts as i32);
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
