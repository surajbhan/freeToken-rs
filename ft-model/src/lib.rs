//! ft-model: Gemma-4 MoE forward pass on top of the freeToken-rs engine.
//!
//! Architecture ported from FreeToken's reference implementation
//! (python/freetoken/models/gemma4/): SWA/full attention with k_eq_v full
//! layers, per-head q/k/v norms, dual (shared MLP || routed MoE) feed-forward
//! combined through a triple RMSNorm, per-layer output scalars, softcapped
//! logits over a tied q6_k lm_head.
//!
//! Compute split (correctness-first): dense q4_0 GEMVs + expert FFN + lm_head
//! on GPU; norms, rope, attention, routing, sampling on CPU (bs=1 decode).

pub mod q6k;
pub mod tokenizer;

use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use ft_core::{q4_0, split_misses, SlotCache};
use ft_cuda::{HostBanks, Q4Gemv};
use ft_gguf::Gguf;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

pub const HIDDEN: usize = 2816;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub n_heads: usize,
    pub kv_heads: Vec<usize>,
    pub swa: Vec<bool>,
    pub head_dim_swa: usize,
    pub head_dim_full: usize,
    pub window: usize,
    pub rope_base_full: f32,
    pub rope_base_swa: f32,
    pub eps: f32,
    pub ffn: usize,
    pub n_experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub softcap: f32,
    pub vocab: usize,
    pub max_seq: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let m = |k: &str| -> Result<u64> {
            g.meta
                .get(&format!("gemma4.{k}"))
                .and_then(|v| v.as_u64())
                .with_context(|| format!("gemma4.{k}"))
        };
        let mf = |k: &str| -> Result<f32> {
            match g.meta.get(&format!("gemma4.{k}")) {
                Some(ft_gguf::Value::F32(v)) => Ok(*v),
                _ => bail!("gemma4.{k} not f32"),
            }
        };
        let arr_u = |k: &str| -> Result<Vec<usize>> {
            match g.meta.get(&format!("gemma4.{k}")) {
                Some(ft_gguf::Value::Arr(v)) => {
                    Ok(v.iter().map(|x| x.as_u64().unwrap() as usize).collect())
                }
                _ => bail!("gemma4.{k} not array"),
            }
        };
        let arr_b = |k: &str| -> Result<Vec<bool>> {
            match g.meta.get(&format!("gemma4.{k}")) {
                Some(ft_gguf::Value::Arr(v)) => Ok(v
                    .iter()
                    .map(|x| matches!(x, ft_gguf::Value::Bool(true)))
                    .collect()),
                _ => bail!("gemma4.{k} not array"),
            }
        };
        let vocab = match g.meta.get("tokenizer.ggml.tokens") {
            Some(ft_gguf::Value::Arr(v)) => v.len(),
            _ => bail!("no tokens"),
        };
        Ok(Self {
            n_layers: m("block_count")? as usize,
            n_heads: m("attention.head_count")? as usize,
            kv_heads: arr_u("attention.head_count_kv")?,
            swa: arr_b("attention.sliding_window_pattern")?,
            head_dim_swa: m("attention.key_length_swa")? as usize,
            head_dim_full: m("attention.key_length")? as usize,
            window: m("attention.sliding_window")? as usize,
            rope_base_full: mf("rope.freq_base")?,
            rope_base_swa: mf("rope.freq_base_swa")?,
            eps: mf("attention.layer_norm_rms_epsilon")?,
            ffn: m("feed_forward_length")? as usize,
            n_experts: m("expert_count")? as usize,
            topk: m("expert_used_count")? as usize,
            moe_inter: m("expert_feed_forward_length")? as usize,
            softcap: mf("final_logit_softcapping")?,
            vocab,
            max_seq: 2048,
        })
    }

    pub fn head_dim(&self, layer: usize) -> usize {
        if self.swa[layer] { self.head_dim_swa } else { self.head_dim_full }
    }
    pub fn rope_base(&self, layer: usize) -> f32 {
        if self.swa[layer] { self.rope_base_swa } else { self.rope_base_full }
    }
}

/// Repack a q4_0 tensor [n_rows, k] into the coalesced q4r layout.
fn repack_q4r(src: &[u8], n_rows: usize, k: usize) -> Vec<u8> {
    let rb = q4_0::row_bytes(k);
    let rbr = q4_0::q4r_row_bytes(k);
    assert_eq!(src.len(), n_rows * rb);
    let mut out = vec![0u8; n_rows * rbr];
    out.par_chunks_exact_mut(rbr)
        .enumerate()
        .for_each(|(r, dst)| q4_0::repack_row_q4r(&src[r * rb..(r + 1) * rb], dst, k));
    out
}

fn q4r_geom(k: usize) -> (usize, usize) {
    (q4_0::q4r_row_bytes(k), q4_0::q4r_scales_bytes(k))
}

/// Dequantize any small f32/f16/q4_0 gguf tensor to Vec<f32>.
fn to_f32(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name)?;
    let data = g.tensor_data(name)?;
    let n: u64 = t.dims.iter().product();
    match t.ggml_type {
        0 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        1 => Ok(data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect()),
        2 => {
            let mut out = vec![0f32; n as usize];
            for (i, row) in out.chunks_exact_mut(32).enumerate() {
                q4_0::dequantize_row(&data[i * 18..(i + 1) * 18], row);
            }
            Ok(out)
        }
        ty => bail!("{name}: unsupported small-tensor type {ty}"),
    }
}

pub struct LayerWeights {
    // GPU q4_0 packed
    pub qkv: CudaSlice<u8>,
    pub qkv_rows: usize,
    pub o: CudaSlice<u8>,
    pub gate_up: CudaSlice<u8>,
    pub down: CudaSlice<u8>,
    // CPU f32
    pub attn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub pre_ffw2: Vec<f32>,
    pub post_ffw: Vec<f32>,
    pub post_ffw1: Vec<f32>,
    pub post_ffw2: Vec<f32>,
    pub layer_scalar: f32,
    pub router_w: Vec<f32>,         // [n_experts, HIDDEN]
    pub router_scale: Vec<f32>,     // [HIDDEN]
    pub per_expert_scale: Vec<f32>, // [n_experts]
    // device-resident copies for the GPU decode path
    pub attn_norm_d: CudaSlice<f32>,
    pub q_norm_d: CudaSlice<f32>,
    pub k_norm_d: CudaSlice<f32>,
    pub post_attn_d: CudaSlice<f32>,
    pub ffn_norm_d: CudaSlice<f32>,
    pub pre_ffw2_d: CudaSlice<f32>,
    pub post_ffw_d: CudaSlice<f32>,
    pub post_ffw1_d: CudaSlice<f32>,
    pub post_ffw2_d: CudaSlice<f32>,
    /// router weights with per-dim scale and H^-0.5 folded in
    pub router_wf_d: CudaSlice<f32>,
    pub expert_scale_d: CudaSlice<f32>,
}

pub struct MoeEngine {
    pub banks: HostBanks,
    pub cache_buf: CudaSlice<u8>,
    pub cache: SlotCache,
    pub eb: usize,
    pub gu_bytes: usize,
    pub fraction: f64,
    slots_dev: CudaSlice<i32>,
    wts_dev: CudaSlice<f32>,
    // host staging kept alive across the deferred sync (uploads are async)
    slots_host: Vec<i32>,
    wts_host: Vec<f32>,
    x_dev: CudaSlice<f32>,
    x_q8: CudaSlice<u8>,
    act_q8: CudaSlice<u8>,
    y_gu: CudaSlice<f32>,
    act: CudaSlice<f32>,
    y_down: CudaSlice<f32>,
    out_dev: CudaSlice<f32>,
}

/// GPU q4_0 GEMV of a standalone dense tensor: out = W x (one "expert"),
/// via the dp4a v3 path (activations quantized to q8 on-GPU).
#[allow(clippy::too_many_arguments)]
fn dense_gemv(
    stream: &Arc<CudaStream>,
    gemv: &Q4Gemv,
    xbufs: &mut HashMap<usize, CudaSlice<f32>>,
    q8bufs: &mut HashMap<usize, CudaSlice<u8>>,
    ybufs: &mut HashMap<usize, CudaSlice<f32>>,
    zero_slot: &CudaSlice<i32>,
    w: &CudaSlice<u8>,
    x_host: &[f32],
    n_rows: usize,
    k: usize,
    out: &mut [f32],
) -> Result<()> {
    assert_eq!(x_host.len(), k);
    assert_eq!(out.len(), n_rows);
    if !xbufs.contains_key(&k) {
        xbufs.insert(k, stream.alloc_zeros::<f32>(k)?);
    }
    if !q8bufs.contains_key(&k) {
        q8bufs.insert(k, stream.alloc_zeros::<u8>(k / 32 * ft_cuda::Q8_BLK)?);
    }
    if !ybufs.contains_key(&n_rows) {
        ybufs.insert(n_rows, stream.alloc_zeros::<f32>(n_rows)?);
    }
    let xd = xbufs.get_mut(&k).unwrap();
    stream.memcpy_htod(x_host, xd)?;
    let xd = xbufs.get(&k).unwrap();
    let q8 = q8bufs.get_mut(&k).unwrap();
    gemv.quantize_q8(stream, xd, 0, q8, k, 1)?;
    let q8 = q8bufs.get(&k).unwrap();
    let yd = ybufs.get_mut(&n_rows).unwrap();
    gemv.gemv_grouped_q8(stream, w, 0, 0, zero_slot, 1, q8, 0, yd, n_rows, n_rows, k)?;
    let yd = ybufs.get(&n_rows).unwrap();
    stream.memcpy_dtoh(yd, out)?;
    stream.synchronize()?;
    Ok(())
}

#[derive(Default, Clone)]
pub struct Profile {
    pub embed_us: u64,
    pub attn_norms_us: u64,   // CPU norms + rope + qk-norms
    pub qkv_gemv_us: u64,     // upload + gemv + download + sync
    pub attn_cpu_us: u64,     // scores/softmax/weighted-V on CPU
    pub o_gemv_us: u64,
    pub shared_mlp_us: u64,   // gate_up + gelu + down (incl. transfers)
    pub router_us: u64,
    pub moe_us: u64,
    pub combine_us: u64,
    pub lm_head_us: u64,
    pub tokens: u64,
}

impl Profile {
    pub fn report(&self) -> String {
        let t = self.tokens.max(1);
        let per = |v: u64| v as f64 / t as f64 / 1000.0;
        let total = self.embed_us
            + self.attn_norms_us
            + self.qkv_gemv_us
            + self.attn_cpu_us
            + self.o_gemv_us
            + self.shared_mlp_us
            + self.router_us
            + self.moe_us
            + self.combine_us
            + self.lm_head_us;
        format!(
            "per-token ms: embed {:.2} | attn-norms/rope {:.2} | qkv-gemv {:.2} | attn-cpu {:.2} | o-gemv {:.2} | shared-mlp {:.2} | router {:.2} | moe {:.2} | combine {:.2} | lm_head {:.2} | SUM {:.2}",
            per(self.embed_us),
            per(self.attn_norms_us),
            per(self.qkv_gemv_us),
            per(self.attn_cpu_us),
            per(self.o_gemv_us),
            per(self.shared_mlp_us),
            per(self.router_us),
            per(self.moe_us),
            per(self.combine_us),
            per(self.lm_head_us),
            per(total)
        )
    }
}

pub struct Model {
    pub prof: Profile,
    pub cfg: Config,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub embed_cpu: Vec<u8>,
    /// tied lm_head, requantized q6_k -> q4_0 at load for the dp4a path
    pub lm_head_q4: CudaSlice<u8>,
    pub moe: MoeEngine,
    pub gemv: Q4Gemv,
    pub stream: Arc<CudaStream>,
    pub copy_stream: Arc<CudaStream>,
    zero_slot: CudaSlice<i32>,
    xbufs: HashMap<usize, CudaSlice<f32>>,
    q8bufs: HashMap<usize, CudaSlice<u8>>,
    ybufs: HashMap<usize, CudaSlice<f32>>,
    act_dev: CudaSlice<f32>,
    logits_dev: CudaSlice<f32>,
    /// pooled per-layer f16 KV caches: [max_batch * max_seq * kv_dim]
    k_pool: Vec<CudaSlice<u16>>,
    v_pool: Vec<CudaSlice<u16>>,
    attn_out_dev: CudaSlice<f32>,
    x_res: CudaSlice<f32>,
    inv_freq_swa_d: CudaSlice<f32>,
    inv_freq_full_d: CudaSlice<f32>,
    pos_dev: CudaSlice<i32>,
    slot_dev: CudaSlice<i32>,
    start_dev: CudaSlice<i32>,
    router_logits_dev: CudaSlice<f32>,
    sc_a: CudaSlice<f32>,
    sc_o: CudaSlice<f32>,
    attn_partials: CudaSlice<f32>,
    sc_pf: CudaSlice<f32>,
    sc_ri: CudaSlice<f32>,
    final_norm_d: CudaSlice<f32>,
    temps_dev: CudaSlice<f32>,
    rng_dev: CudaSlice<u64>,
    tok_out_dev: CudaSlice<i32>,
    want_sample: bool,
    /// fully device-side routing/admission (graph-compatible path)
    pub gpu_routing: bool,
    pub sync_prof: bool,
    n_slots: usize,
    lru_map: CudaSlice<i32>,
    lru_slot_key: CudaSlice<i32>,
    lru_slot_last: CudaSlice<u32>,
    lru_clock: CudaSlice<u32>,
    moe_ids_dev: CudaSlice<i32>,
    promote_src_dev: CudaSlice<i32>,
    promote_dst_dev: CudaSlice<i32>,
    /// instantiated CUDA graph per batch size (raw CUgraphExec as usize)
    graph_exec: Vec<usize>,
    captured_events: Vec<cudarc::driver::CudaEvent>,
    inv_freq_swa: Vec<f32>,
    inv_freq_full: Vec<f32>,
    pub max_batch: usize,
    /// per batch-slot sequence position
    pub seq_pos: Vec<usize>,
    // batched MoE pair staging
    pair_slots: CudaSlice<i32>,
    pair_xidx: CudaSlice<i32>,
    pair_wts: CudaSlice<f32>,
    pair_seq: CudaSlice<i32>,
    pair_bases: CudaSlice<u64>,
    pair_bases_h: Vec<u64>,
    banks_dptr: u64,
    cache_dptr: u64,
    pending_fetch: Option<cudarc::driver::CudaEvent>,
    pair_slots_h: Vec<i32>,
    pair_xidx_h: Vec<i32>,
    pair_wts_h: Vec<f32>,
    pair_seq_h: Vec<i32>,
    pair_y_gu: CudaSlice<f32>,
    pair_act: CudaSlice<f32>,
    pair_act_q8: CudaSlice<u8>,
    pair_y_down: CudaSlice<f32>,
    routed_out_dev: CudaSlice<f32>,
    dense_idx: CudaSlice<i32>,
    zeros_b: CudaSlice<i32>,
    q8_preff: CudaSlice<u8>,
    q8_routed: CudaSlice<u8>,
}

fn rmsnorm_into(x: &[f32], w: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ss + eps).sqrt();
    match w {
        Some(w) => {
            for i in 0..x.len() {
                out[i] = x[i] * r * w[i];
            }
        }
        None => {
            for i in 0..x.len() {
                out[i] = x[i] * r;
            }
        }
    }
}

fn rmsnorm_inplace(x: &mut [f32], w: Option<&[f32]>, eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ss + eps).sqrt();
    match w {
        Some(w) => {
            for (v, wi) in x.iter_mut().zip(w) {
                *v *= r * wi;
            }
        }
        None => {
            for v in x.iter_mut() {
                *v *= r;
            }
        }
    }
}

/// NeoX half-rotation rope over each head in `v`, using a precomputed
/// inverse-frequency table (powf per element was ~2 ms/token).
fn rope_neox(v: &mut [f32], head_dim: usize, pos: usize, inv_freq: &[f32]) {
    let half = head_dim / 2;
    for h in v.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let (s, c) = (pos as f32 * inv_freq[i]).sin_cos();
            let (a, b) = (h[i], h[i + half]);
            h[i] = a * c - b * s;
            h[i + half] = a * s + b * c;
        }
    }
}

fn build_inv_freq(head_dim: usize, base: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
        .collect()
}

fn gelu_tanh(g: f32) -> f32 {
    0.5 * g * (1.0 + (0.7978845608f32 * (g + 0.044715 * g * g * g)).tanh())
}

impl Model {
    pub fn load(
        g: &Gguf,
        ctx: &Arc<CudaContext>,
        cache_slots: usize,
        fraction: f64,
        max_batch: usize,
    ) -> Result<Self> {
        let cfg = Config::from_gguf(g)?;
        // We order the compute/copy streams explicitly with events; cudarc's
        // automatic per-buffer event tracking would inject waits on
        // pre-capture events and break CUDA-graph capture.
        unsafe { ctx.disable_event_tracking() };
        // non-default stream: the legacy default stream cannot be graph-captured
        let stream = ctx.new_stream()?;
        let copy_stream = ctx.new_stream()?;
        let gemv = Q4Gemv::new(ctx)?;

        let rb_h = q4_0::row_bytes(HIDDEN);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let t = |sfx: &str| format!("blk.{l}.{sfx}");
            let hd = cfg.head_dim(l);
            let kvh = cfg.kv_heads[l];
            let q_rows = cfg.n_heads * hd;
            let kv_rows = kvh * hd;

            // fused qkv (full layers ship no attn_v: k reused as v)
            let qd = g.tensor_data(&t("attn_q.weight"))?;
            let kd = g.tensor_data(&t("attn_k.weight"))?;
            let vd = g.tensor_data(&t("attn_v.weight")).unwrap_or(kd);
            anyhow::ensure!(qd.len() == q_rows * rb_h, "attn_q shape");
            anyhow::ensure!(kd.len() == kv_rows * rb_h, "attn_k shape");
            let mut qkv_host = Vec::with_capacity(qd.len() + kd.len() + vd.len());
            qkv_host.extend_from_slice(qd);
            qkv_host.extend_from_slice(kd);
            qkv_host.extend_from_slice(vd);
            let qkv_rows = q_rows + 2 * kv_rows;
            let qkv = stream.memcpy_stod(&repack_q4r(&qkv_host, qkv_rows, HIDDEN))?;

            let o = stream.memcpy_stod(&repack_q4r(
                g.tensor_data(&t("attn_output.weight"))?,
                HIDDEN,
                q_rows,
            ))?;

            let gd = g.tensor_data(&t("ffn_gate.weight"))?;
            let ud = g.tensor_data(&t("ffn_up.weight"))?;
            let mut gu_host = Vec::with_capacity(gd.len() + ud.len());
            gu_host.extend_from_slice(gd);
            gu_host.extend_from_slice(ud);
            let gate_up = stream.memcpy_stod(&repack_q4r(&gu_host, 2 * cfg.ffn, HIDDEN))?;
            let down = stream.memcpy_stod(&repack_q4r(
                g.tensor_data(&t("ffn_down.weight"))?,
                HIDDEN,
                cfg.ffn,
            ))?;

            let attn_norm = to_f32(g, &t("attn_norm.weight"))?;
            let q_norm = to_f32(g, &t("attn_q_norm.weight"))?;
            let k_norm = to_f32(g, &t("attn_k_norm.weight"))?;
            let post_attn_norm = to_f32(g, &t("post_attention_norm.weight"))?;
            let ffn_norm = to_f32(g, &t("ffn_norm.weight"))?;
            let pre_ffw2 = to_f32(g, &t("pre_ffw_norm_2.weight"))?;
            let post_ffw = to_f32(g, &t("post_ffw_norm.weight"))?;
            let post_ffw1 = to_f32(g, &t("post_ffw_norm_1.weight"))?;
            let post_ffw2 = to_f32(g, &t("post_ffw_norm_2.weight"))?;
            let router_w = to_f32(g, &t("ffn_gate_inp.weight"))?;
            let router_scale = to_f32(g, &t("ffn_gate_inp.scale"))?;
            let root = (HIDDEN as f32).powf(-0.5);
            let mut router_wf = router_w.clone();
            for row in router_wf.chunks_exact_mut(HIDDEN) {
                for (j, v) in row.iter_mut().enumerate() {
                    *v *= router_scale[j] * root;
                }
            }
            layers.push(LayerWeights {
                qkv,
                qkv_rows,
                o,
                gate_up,
                down,
                attn_norm_d: stream.memcpy_stod(&attn_norm)?,
                q_norm_d: stream.memcpy_stod(&q_norm)?,
                k_norm_d: stream.memcpy_stod(&k_norm)?,
                post_attn_d: stream.memcpy_stod(&post_attn_norm)?,
                ffn_norm_d: stream.memcpy_stod(&ffn_norm)?,
                pre_ffw2_d: stream.memcpy_stod(&pre_ffw2)?,
                post_ffw_d: stream.memcpy_stod(&post_ffw)?,
                post_ffw1_d: stream.memcpy_stod(&post_ffw1)?,
                post_ffw2_d: stream.memcpy_stod(&post_ffw2)?,
                router_wf_d: stream.memcpy_stod(&router_wf)?,
                expert_scale_d: stream.memcpy_stod(&to_f32(g, &t("ffn_down_exps.scale"))?)?,
                attn_norm,
                q_norm,
                k_norm,
                post_attn_norm,
                ffn_norm,
                pre_ffw2,
                post_ffw,
                post_ffw1,
                post_ffw2,
                layer_scalar: to_f32(g, &t("layer_output_scale.weight"))?[0],
                router_w,
                router_scale,
                per_expert_scale: to_f32(g, &t("ffn_down_exps.scale"))?,
            });
        }

        // embedding (q6_k, row per token) doubles as the tied lm_head on GPU
        let et = g.tensor("token_embd.weight")?;
        anyhow::ensure!(et.ggml_type == 14, "embedding must be q6_k");
        let embed_cpu = g.tensor_data("token_embd.weight")?.to_vec();
        // Requantize the tied lm_head q6_k -> q4_0: the dp4a q4_0 kernel runs
        // ~3x faster and the small extra quantization error only touches
        // logits (embedding lookups keep the exact q6_k table on the CPU).
        let erb6 = q6k::row_bytes(HIDDEN);
        let rb4 = q4_0::row_bytes(HIDDEN);
        let mut lm_q4 = vec![0u8; cfg.vocab * rb4];
        lm_q4
            .par_chunks_exact_mut(rb4)
            .enumerate()
            .for_each(|(r, dst)| {
                let mut row = vec![0f32; HIDDEN];
                q6k::dequantize_row(&embed_cpu[r * erb6..(r + 1) * erb6], &mut row);
                q4_0::quantize_row(&row, dst);
            });
        let lm_head_q4 = stream.memcpy_stod(&repack_q4r(&lm_q4, cfg.vocab, HIDDEN))?;
        drop(lm_q4);
        let final_norm = to_f32(g, "output_norm.weight")?;

        // MoE expert banks (repacked q4r layout)
        let gu_rows = 2 * cfg.moe_inter;
        let gu_bytes_src = rb_h * gu_rows;
        let down_bytes_src = q4_0::row_bytes(cfg.moe_inter) * HIDDEN;
        let gu_bytes = q4_0::q4r_row_bytes(HIDDEN) * gu_rows;
        let down_bytes = q4_0::q4r_row_bytes(cfg.moe_inter) * HIDDEN;
        let eb = gu_bytes + down_bytes;
        let mut banks = HostBanks::new(ctx, cfg.n_layers * cfg.n_experts * eb)?;
        {
            let hs = banks.as_mut_slice();
            for l in 0..cfg.n_layers {
                let gu = g.tensor_data(&format!("blk.{l}.ffn_gate_up_exps.weight"))?;
                let dn = g.tensor_data(&format!("blk.{l}.ffn_down_exps.weight"))?;
                hs[l * cfg.n_experts * eb..(l + 1) * cfg.n_experts * eb]
                    .par_chunks_exact_mut(eb)
                    .enumerate()
                    .for_each(|(e, dst)| {
                        let gsrc = &gu[e * gu_bytes_src..(e + 1) * gu_bytes_src];
                        let dsrc = &dn[e * down_bytes_src..(e + 1) * down_bytes_src];
                        let rb = q4_0::row_bytes(HIDDEN);
                        let rbr = q4_0::q4r_row_bytes(HIDDEN);
                        for r in 0..gu_rows {
                            q4_0::repack_row_q4r(
                                &gsrc[r * rb..(r + 1) * rb],
                                &mut dst[r * rbr..(r + 1) * rbr],
                                HIDDEN,
                            );
                        }
                        let rb2 = q4_0::row_bytes(cfg.moe_inter);
                        let rbr2 = q4_0::q4r_row_bytes(cfg.moe_inter);
                        for r in 0..HIDDEN {
                            q4_0::repack_row_q4r(
                                &dsrc[r * rb2..(r + 1) * rb2],
                                &mut dst[gu_bytes + r * rbr2..gu_bytes + (r + 1) * rbr2],
                                cfg.moe_inter,
                            );
                        }
                    });
            }
        }
        let moe = MoeEngine {
            cache_buf: stream.alloc_zeros::<u8>(cache_slots * eb)?,
            cache: SlotCache::new(cache_slots as u32),
            banks,
            eb,
            gu_bytes,
            fraction,
            slots_dev: stream.alloc_zeros::<i32>(cfg.topk)?,
            wts_dev: stream.alloc_zeros::<f32>(cfg.topk)?,
            slots_host: Vec::with_capacity(cfg.topk),
            wts_host: Vec::with_capacity(cfg.topk),
            x_dev: stream.alloc_zeros::<f32>(HIDDEN)?,
            x_q8: stream.alloc_zeros::<u8>(HIDDEN / 32 * ft_cuda::Q8_BLK)?,
            act_q8: stream
                .alloc_zeros::<u8>(cfg.topk * cfg.moe_inter / 32 * ft_cuda::Q8_BLK)?,
            y_gu: stream.alloc_zeros::<f32>(cfg.topk * gu_rows)?,
            act: stream.alloc_zeros::<f32>(cfg.topk * cfg.moe_inter)?,
            y_down: stream.alloc_zeros::<f32>(cfg.topk * HIDDEN)?,
            out_dev: stream.alloc_zeros::<f32>(HIDDEN)?,
        };

        let kv_dims: Vec<usize> = (0..cfg.n_layers)
            .map(|l| cfg.kv_heads[l] * cfg.head_dim(l))
            .collect();
        let mut m = Self {
            prof: Profile::default(),
            zero_slot: stream.alloc_zeros::<i32>(1)?,
            xbufs: HashMap::new(),
            q8bufs: HashMap::new(),
            ybufs: HashMap::new(),
            act_dev: stream.alloc_zeros::<f32>(max_batch * cfg.ffn)?,
            logits_dev: stream.alloc_zeros::<f32>(max_batch * cfg.vocab)?,
            k_pool: kv_dims
                .iter()
                .map(|d| stream.alloc_zeros::<u16>(max_batch * cfg.max_seq * d))
                .collect::<Result<_, _>>()?,
            v_pool: kv_dims
                .iter()
                .map(|d| stream.alloc_zeros::<u16>(max_batch * cfg.max_seq * d))
                .collect::<Result<_, _>>()?,
            attn_out_dev: stream
                .alloc_zeros::<f32>(max_batch * cfg.n_heads * cfg.head_dim_full)?,
            x_res: stream.alloc_zeros::<f32>(max_batch * HIDDEN)?,
            inv_freq_swa_d: stream
                .memcpy_stod(&build_inv_freq(cfg.head_dim_swa, cfg.rope_base_swa))?,
            inv_freq_full_d: stream
                .memcpy_stod(&build_inv_freq(cfg.head_dim_full, cfg.rope_base_full))?,
            pos_dev: stream.alloc_zeros::<i32>(max_batch)?,
            slot_dev: stream.alloc_zeros::<i32>(max_batch)?,
            start_dev: stream.alloc_zeros::<i32>(max_batch)?,
            router_logits_dev: stream.alloc_zeros::<f32>(max_batch * cfg.n_experts)?,
            sc_a: stream.alloc_zeros::<f32>(max_batch * HIDDEN)?,
            sc_o: stream.alloc_zeros::<f32>(max_batch * cfg.n_heads * cfg.head_dim_full)?,
            attn_partials: stream.alloc_zeros::<f32>(
                max_batch * cfg.n_heads * cfg.max_seq.div_ceil(128) * (cfg.head_dim_full + 2),
            )?,
            sc_pf: stream.alloc_zeros::<f32>(max_batch * HIDDEN)?,
            sc_ri: stream.alloc_zeros::<f32>(max_batch * HIDDEN)?,
            final_norm_d: stream.memcpy_stod(&final_norm)?,
            temps_dev: stream.alloc_zeros::<f32>(max_batch)?,
            rng_dev: {
                let seeds: Vec<u64> = (0..max_batch as u64)
                    .map(|i| 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i + 1))
                    .collect();
                stream.memcpy_stod(&seeds)?
            },
            tok_out_dev: stream.alloc_zeros::<i32>(max_batch)?,
            want_sample: false,
            gpu_routing: false,
            sync_prof: std::env::var("FT_SYNCPROF").is_ok(),
            n_slots: cache_slots,
            lru_map: {
                let v = vec![-1i32; cfg.n_layers * cfg.n_experts];
                stream.memcpy_stod(&v)?
            },
            lru_slot_key: {
                let v = vec![-1i32; cache_slots];
                stream.memcpy_stod(&v)?
            },
            lru_slot_last: stream.alloc_zeros::<u32>(cache_slots)?,
            lru_clock: stream.alloc_zeros::<u32>(1)?,
            moe_ids_dev: stream.alloc_zeros::<i32>(max_batch * cfg.topk)?,
            promote_src_dev: stream.alloc_zeros::<i32>(max_batch * cfg.topk)?,
            promote_dst_dev: stream.alloc_zeros::<i32>(max_batch * cfg.topk)?,
            graph_exec: vec![0; max_batch + 1],
            captured_events: Vec::new(),
            inv_freq_swa: build_inv_freq(cfg.head_dim_swa, cfg.rope_base_swa),
            inv_freq_full: build_inv_freq(cfg.head_dim_full, cfg.rope_base_full),
            max_batch,
            seq_pos: vec![0; max_batch],
            pair_slots: stream.alloc_zeros::<i32>(max_batch * cfg.topk)?,
            pair_xidx: stream.alloc_zeros::<i32>(max_batch * cfg.topk)?,
            pair_wts: stream.alloc_zeros::<f32>(max_batch * cfg.topk)?,
            pair_seq: stream.alloc_zeros::<i32>(max_batch * cfg.topk)?,
            pair_bases: stream.alloc_zeros::<u64>(max_batch * cfg.topk)?,
            pair_bases_h: Vec::new(),
            banks_dptr: 0,
            cache_dptr: 0,
            pending_fetch: None,
            pair_slots_h: Vec::new(),
            pair_xidx_h: Vec::new(),
            pair_wts_h: Vec::new(),
            pair_seq_h: Vec::new(),
            pair_y_gu: stream.alloc_zeros::<f32>(max_batch * cfg.topk * 2 * cfg.moe_inter)?,
            pair_act: stream.alloc_zeros::<f32>(max_batch * cfg.topk * cfg.moe_inter)?,
            pair_act_q8: stream
                .alloc_zeros::<u8>(max_batch * cfg.topk * cfg.moe_inter / 32 * ft_cuda::Q8_BLK)?,
            pair_y_down: stream.alloc_zeros::<f32>(max_batch * cfg.topk * HIDDEN)?,
            routed_out_dev: stream.alloc_zeros::<f32>(max_batch * HIDDEN)?,
            dense_idx: {
                let idx: Vec<i32> = (0..(max_batch * cfg.topk) as i32).collect();
                stream.memcpy_stod(&idx)?
            },
            zeros_b: stream.alloc_zeros::<i32>(max_batch * cfg.topk)?,
            q8_preff: stream.alloc_zeros::<u8>(max_batch * HIDDEN / 32 * ft_cuda::Q8_BLK)?,
            q8_routed: stream.alloc_zeros::<u8>(max_batch * HIDDEN / 32 * ft_cuda::Q8_BLK)?,
            cfg,
            layers,
            final_norm,
            embed_cpu,
            lm_head_q4,
            moe,
            gemv,
            stream,
            copy_stream,
        };
        m.init_uva()?;
        Ok(m)
    }

    /// resolve device-visible base addresses for the slot cache and the
    /// pinned host banks (UVA) — enables pointer-based expert GEMVs.
    fn init_uva(&mut self) -> Result<()> {
        use cudarc::driver::DevicePtr;
        self.banks_dptr = self.moe.banks.device_ptr()?;
        let (p, _g) = self.moe.cache_buf.device_ptr(&self.stream);
        self.cache_dptr = p as u64;
        Ok(())
    }

    /// debug: read back the dense_idx device buffer
    pub fn debug_dense_idx(&self) -> Result<Vec<i32>> {
        let v = self.stream.memcpy_dtov(&self.dense_idx)?;
        Ok(v)
    }

    /// Reset a batch slot's sequence (fresh request).
    pub fn reset_slot(&mut self, slot: usize) {
        self.seq_pos[slot] = 0;
    }

    /// Single-sequence step on batch slot 0.
    pub fn forward_token(&mut self, token: u32) -> Result<Vec<f32>> {
        let mut out = self.forward_batch(&[(0, token)])?;
        Ok(out.pop().unwrap())
    }

    /// Pure-GPU decode stack for `nb` sequences (gpu_routing only): every op
    /// from the residual stream through sampling, no CPU work, fixed launch
    /// shapes — the CUDA-graph body. Inputs (x_res, pos/slot/temps, pair
    /// maps) must already be uploaded.
    fn enqueue_stack(&mut self, nb: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        anyhow::ensure!(self.gpu_routing, "enqueue_stack requires gpu_routing");
        let skip_moe = std::env::var("FT_SKIP_MOE").is_ok();
        let skip_attn = std::env::var("FT_SKIP_ATTN").is_ok();
        let skip_dense = std::env::var("FT_SKIP_DENSE").is_ok();
        let skip_lmhead = std::env::var("FT_SKIP_LMHEAD").is_ok();
        for l in 0..cfg.n_layers {
            let hd = cfg.head_dim(l);
            let kvh = cfg.kv_heads[l];
            let q_rows = cfg.n_heads * hd;
            let kv_dim = kvh * hd;
            let qkv_rows = self.layers[l].qkv_rows;
            let group = cfg.n_heads / kvh;

            self.gemv.rmsnorm_rows_dev(
                &self.stream, &self.x_res, Some(&self.layers[l].attn_norm_d),
                &mut self.sc_a, HIDDEN, cfg.eps, nb,
            )?;
            {
                let q8 = self.q8bufs.get_mut(&HIDDEN).unwrap();
                self.gemv.quantize_q8(&self.stream, &self.sc_a, HIDDEN, q8, HIDDEN, nb)?;
            }
            {
                let q8 = self.q8bufs.get(&HIDDEN).unwrap();
                let yd = self.ybufs.get_mut(&qkv_rows).unwrap();
                if !skip_dense { {
                    let (rbr, qo) = q4r_geom(HIDDEN);
                    self.gemv.gemv_q4r_idx(
                        &self.stream, &self.layers[l].qkv, 0, 0, &self.zeros_b, &self.dense_idx,
                        nb, q8, HIDDEN / 32, yd, qkv_rows, qkv_rows, HIDDEN, rbr, qo,
                    )?;
                } }
            }
            {
                let yd = self.ybufs.get_mut(&qkv_rows).unwrap();
                let n_all = nb * qkv_rows;
                {
                    let mut qv = yd.slice_mut(0..n_all);
                    self.gemv.rmsnorm_heads_dev(
                        &self.stream, &mut qv, &self.layers[l].q_norm_d, true, hd, cfg.eps,
                        qkv_rows, cfg.n_heads, nb,
                    )?;
                }
                {
                    let mut kv2 = yd.slice_mut(q_rows..n_all);
                    self.gemv.rmsnorm_heads_dev(
                        &self.stream, &mut kv2, &self.layers[l].k_norm_d, true, hd, cfg.eps,
                        qkv_rows, kvh, nb,
                    )?;
                }
                {
                    let mut vv = yd.slice_mut(q_rows + kv_dim..n_all);
                    self.gemv.rmsnorm_heads_dev(
                        &self.stream, &mut vv, &self.layers[l].k_norm_d, false, hd, cfg.eps,
                        qkv_rows, kvh, nb,
                    )?;
                }
                let inv_freq = if cfg.swa[l] { &self.inv_freq_swa_d } else { &self.inv_freq_full_d };
                {
                    let mut qv = yd.slice_mut(0..n_all);
                    self.gemv.rope_heads_dev(
                        &self.stream, &mut qv, inv_freq, &self.pos_dev, hd, cfg.n_heads,
                        qkv_rows, nb,
                    )?;
                }
                {
                    let mut kv2 = yd.slice_mut(q_rows..n_all);
                    self.gemv.rope_heads_dev(
                        &self.stream, &mut kv2, inv_freq, &self.pos_dev, hd, kvh, qkv_rows, nb,
                    )?;
                }
            }
            {
                let yd = self.ybufs.get(&qkv_rows).unwrap();
                self.gemv.kv_append_dev(
                    &self.stream, yd, &mut self.k_pool[l], &self.slot_dev, &self.pos_dev,
                    q_rows, qkv_rows, kv_dim, cfg.max_seq, nb,
                )?;
                self.gemv.kv_append_dev(
                    &self.stream, yd, &mut self.v_pool[l], &self.slot_dev, &self.pos_dev,
                    q_rows + kv_dim, qkv_rows, kv_dim, cfg.max_seq, nb,
                )?;
                self.gemv
                    .gather_rows_dev(&self.stream, yd, &mut self.attn_out_dev, qkv_rows, 0, q_rows, nb)?;
            }
            if !skip_attn {
                self.gemv.attn_decode_batch_dev(
                    &self.stream, &self.k_pool[l], &self.v_pool[l], &self.attn_out_dev,
                    &mut self.attn_partials, &mut self.sc_o, &self.slot_dev, &self.pos_dev,
                    if cfg.swa[l] { cfg.window } else { 0 },
                    kv_dim, hd, group, cfg.n_heads, cfg.max_seq, nb,
                    cfg.max_seq,
                )?;
            }
            {
                let q8 = self.q8bufs.get_mut(&q_rows).unwrap();
                self.gemv
                    .quantize_q8(&self.stream, &self.sc_o, q_rows, q8, q_rows, nb)?;
            }
            {
                let q8 = self.q8bufs.get(&q_rows).unwrap();
                let yd = self.ybufs.get_mut(&HIDDEN).unwrap();
                if !skip_dense { {
                    let (rbr, qo) = q4r_geom(q_rows);
                    self.gemv.gemv_q4r_idx(
                        &self.stream, &self.layers[l].o, 0, 0, &self.zeros_b, &self.dense_idx,
                        nb, q8, q_rows / 32, yd, HIDDEN, HIDDEN, q_rows, rbr, qo,
                    )?;
                } }
            }
            {
                let yd = self.ybufs.get(&HIDDEN).unwrap();
                self.gemv.rmsnorm_rows_dev(
                    &self.stream, yd, Some(&self.layers[l].post_attn_d), &mut self.sc_a,
                    HIDDEN, cfg.eps, nb,
                )?;
            }
            self.gemv
                .add_rows_dev(&self.stream, &mut self.x_res, &self.sc_a, HIDDEN, nb)?;

            self.gemv.rmsnorm_rows_dev(
                &self.stream, &self.x_res, Some(&self.layers[l].ffn_norm_d), &mut self.sc_pf,
                HIDDEN, cfg.eps, nb,
            )?;
            self.gemv.rmsnorm_rows_dev(
                &self.stream, &self.x_res, Some(&self.layers[l].pre_ffw2_d), &mut self.sc_ri,
                HIDDEN, cfg.eps, nb,
            )?;
            let gu_rows = 2 * cfg.ffn;
            self.gemv
                .quantize_q8(&self.stream, &self.sc_pf, HIDDEN, &mut self.q8_preff, HIDDEN, nb)?;
            {
                let yd = self.ybufs.get_mut(&gu_rows).unwrap();
                if !skip_dense { {
                    let (rbr, qo) = q4r_geom(HIDDEN);
                    self.gemv.gemv_q4r_idx(
                        &self.stream, &self.layers[l].gate_up, 0, 0, &self.zeros_b, &self.dense_idx,
                        nb, &self.q8_preff, HIDDEN / 32, yd, gu_rows, gu_rows, HIDDEN, rbr, qo,
                    )?;
                } }
            }
            {
                let yd = self.ybufs.get(&gu_rows).unwrap();
                self.gemv
                    .gelu_mul_grouped(&self.stream, yd, &mut self.act_dev, cfg.ffn, nb)?;
            }
            {
                let q8a = self.q8bufs.get_mut(&cfg.ffn).unwrap();
                self.gemv
                    .quantize_q8(&self.stream, &self.act_dev, cfg.ffn, q8a, cfg.ffn, nb)?;
            }
            {
                let q8a = self.q8bufs.get(&cfg.ffn).unwrap();
                let yd = self.ybufs.get_mut(&HIDDEN).unwrap();
                if !skip_dense { {
                    let (rbr, qo) = q4r_geom(cfg.ffn);
                    self.gemv.gemv_q4r_idx(
                        &self.stream, &self.layers[l].down, 0, 0, &self.zeros_b, &self.dense_idx,
                        nb, q8a, cfg.ffn / 32, yd, HIDDEN, HIDDEN, cfg.ffn, rbr, qo,
                    )?;
                } }
            }
            self.gemv
                .rmsnorm_rows_dev(&self.stream, &self.x_res, None, &mut self.sc_a, HIDDEN, cfg.eps, nb)?;
            self.gemv.gemv_f32_rows_dev(
                &self.stream, &self.layers[l].router_wf_d, &self.sc_a,
                &mut self.router_logits_dev, HIDDEN, cfg.n_experts, nb,
            )?;
            self.gemv
                .quantize_q8(&self.stream, &self.sc_ri, HIDDEN, &mut self.q8_routed, HIDDEN, nb)?;
            self.gemv.topk_router_dev(
                &self.stream, &self.router_logits_dev, &self.layers[l].expert_scale_d,
                &mut self.moe_ids_dev, &mut self.pair_wts, cfg.n_experts, cfg.topk, nb,
            )?;
            self.gemv.lru_admit_dev(
                &self.stream, &self.moe_ids_dev, l, cfg.n_experts, self.n_slots,
                self.cache_dptr, self.banks_dptr, self.moe.eb,
                &mut self.lru_map, &mut self.lru_slot_key, &mut self.lru_slot_last,
                &mut self.lru_clock, &mut self.pair_bases,
                &mut self.promote_src_dev, &mut self.promote_dst_dev, nb * cfg.topk,
            )?;
            let ev_a = self.stream.record_event(None)?;
            self.copy_stream.wait(&ev_a)?;
            self.captured_events.push(ev_a);
            self.gemv.promote_experts_dev(
                &self.copy_stream, &self.promote_src_dev, &self.promote_dst_dev,
                self.banks_dptr, self.cache_dptr, self.moe.eb, nb * cfg.topk,
            )?;
            // two-phase: pair gemvs read misses from their freshly-promoted
            // slots, so the fetch must land before they launch
            let ev_f = self.copy_stream.record_event(None)?;
            self.stream.wait(&ev_f)?;
            self.captured_events.push(ev_f);

            let n_pairs = nb * cfg.topk;
            {
                let mut r = self.routed_out_dev.slice_mut(0..nb * HIDDEN);
                self.stream.memset_zeros(&mut r)?;
            }
            if !skip_moe {
            let gu_rows_e = 2 * cfg.moe_inter;
            {
                let (rbr, qo) = q4r_geom(HIDDEN);
                self.gemv.gemv_q4r_ptr(
                    &self.stream, &self.pair_bases, 0, &self.pair_xidx, n_pairs,
                    &self.q8_routed, HIDDEN / 32, &mut self.pair_y_gu, gu_rows_e,
                    gu_rows_e, HIDDEN, rbr, qo,
                )?;
            }
            self.gemv.gelu_mul_grouped(
                &self.stream, &self.pair_y_gu, &mut self.pair_act, cfg.moe_inter, n_pairs,
            )?;
            self.gemv.quantize_q8(
                &self.stream, &self.pair_act, cfg.moe_inter, &mut self.pair_act_q8,
                cfg.moe_inter, n_pairs,
            )?;
            {
                let (rbr, qo) = q4r_geom(cfg.moe_inter);
                self.gemv.gemv_q4r_ptr(
                    &self.stream, &self.pair_bases, self.moe.gu_bytes, &self.dense_idx,
                    n_pairs, &self.pair_act_q8, cfg.moe_inter / 32, &mut self.pair_y_down,
                    HIDDEN, HIDDEN, cfg.moe_inter, rbr, qo,
                )?;
            }
            self.gemv.reduce_pairs_weighted(
                &self.stream, &self.pair_y_down, &self.pair_wts, &self.pair_seq,
                &mut self.routed_out_dev, HIDDEN, n_pairs, nb,
            )?;
            }
            {
                let sh2 = self.ybufs.get(&HIDDEN).unwrap();
                self.gemv.dual_combine_dev(
                    &self.stream, &mut self.x_res, sh2, &self.routed_out_dev,
                    &self.layers[l].post_ffw1_d, &self.layers[l].post_ffw2_d,
                    &self.layers[l].post_ffw_d, self.layers[l].layer_scalar,
                    HIDDEN, cfg.eps, nb,
                )?;
            }
        }
        // rejoin the promote fork (capture must not end with a dangling
        // branch); only the final layer's promote can still be in flight here
        if let Some(ev) = self.pending_fetch.take() {
            self.stream.wait(&ev)?;
            self.captured_events.push(ev);
        }
        // final norm + lm_head + sample
        self.gemv.rmsnorm_rows_dev(
            &self.stream, &self.x_res, Some(&self.final_norm_d), &mut self.sc_a,
            HIDDEN, cfg.eps, nb,
        )?;
        self.gemv
            .quantize_q8(&self.stream, &self.sc_a, HIDDEN, &mut self.q8_preff, HIDDEN, nb)?;
        if !skip_lmhead { {
            let (rbr, qo) = q4r_geom(HIDDEN);
            self.gemv.gemv_q4r_idx(
                &self.stream, &self.lm_head_q4, 0, 0, &self.zeros_b, &self.dense_idx,
                nb, &self.q8_preff, HIDDEN / 32, &mut self.logits_dev, cfg.vocab,
                cfg.vocab, HIDDEN, rbr, qo,
            )?;
        } }
        self.gemv.sample_tokens_dev(
            &self.stream, &self.logits_dev, &self.temps_dev, &mut self.rng_dev,
            &mut self.tok_out_dev, cfg.vocab, cfg.softcap, nb,
        )?;
        Ok(())
    }

    /// allocate every lazy scratch buffer the decode stack touches (must be
    /// done before graph capture — allocation is not capturable).
    fn ensure_buffers(&mut self) -> Result<()> {
        let cfg = self.cfg.clone();
        let mut q8_keys = vec![HIDDEN, cfg.ffn];
        let mut y_keys = vec![HIDDEN, 2 * cfg.ffn];
        for l in 0..cfg.n_layers {
            let q_rows = cfg.n_heads * cfg.head_dim(l);
            q8_keys.push(q_rows);
            y_keys.push(self.layers[l].qkv_rows);
        }
        for k in q8_keys {
            if !self.q8bufs.contains_key(&k) {
                self.q8bufs.insert(
                    k,
                    self.stream
                        .alloc_zeros::<u8>(self.max_batch * k / 32 * ft_cuda::Q8_BLK)?,
                );
            }
        }
        for r in y_keys {
            if !self.ybufs.contains_key(&r) {
                self.ybufs
                    .insert(r, self.stream.alloc_zeros::<f32>(self.max_batch * r)?);
            }
        }
        Ok(())
    }

    fn upload_inputs(&mut self, reqs: &[(usize, u32)], temps: &[f32]) -> Result<()> {
        let nb = reqs.len();
        let cfg_topk = self.cfg.topk;
        let scale = (HIDDEN as f32).sqrt();
        let erb = q6k::row_bytes(HIDDEN);
        let mut h_all = vec![0f32; nb * HIDDEN];
        for (i, &(slot, token)) in reqs.iter().enumerate() {
            anyhow::ensure!(self.seq_pos[slot] < self.cfg.max_seq, "context overflow");
            let row = &mut h_all[i * HIDDEN..(i + 1) * HIDDEN];
            q6k::dequantize_row(
                &self.embed_cpu[token as usize * erb..(token as usize + 1) * erb],
                row,
            );
            for v in row.iter_mut() {
                *v *= scale;
            }
        }
        {
            let mut xr = self.x_res.slice_mut(0..nb * HIDDEN);
            self.stream.memcpy_htod(&h_all[..nb * HIDDEN], &mut xr)?;
        }
        let pos_h: Vec<i32> = reqs.iter().map(|&(sl, _)| self.seq_pos[sl] as i32).collect();
        let slot_h: Vec<i32> = reqs.iter().map(|&(sl, _)| sl as i32).collect();
        {
            let mut pd = self.pos_dev.slice_mut(0..nb);
            self.stream.memcpy_htod(&pos_h, &mut pd)?;
            let mut sd = self.slot_dev.slice_mut(0..nb);
            self.stream.memcpy_htod(&slot_h, &mut sd)?;
            let mut td = self.temps_dev.slice_mut(0..nb);
            self.stream.memcpy_htod(temps, &mut td)?;
        }
        let xi: Vec<i32> = (0..nb * cfg_topk).map(|p2| (p2 / cfg_topk) as i32).collect();
        {
            let mut xd = self.pair_xidx.slice_mut(0..nb * cfg_topk);
            self.stream.memcpy_htod(&xi, &mut xd)?;
            let mut sq = self.pair_seq.slice_mut(0..nb * cfg_topk);
            self.stream.memcpy_htod(&xi, &mut sq)?;
        }
        Ok(())
    }

    /// Graph-replayed decode step: builds a CUDA graph of the whole token on
    /// first use per batch size, then replays it — one graph launch instead
    /// of ~750 kernel launches.
    pub fn forward_sample_graphed(
        &mut self,
        reqs: &[(usize, u32)],
        temps: &[f32],
    ) -> Result<Vec<u32>> {
        use cudarc::driver::sys as cus;
        let nb = reqs.len();
        anyhow::ensure!(self.gpu_routing, "graphed path requires gpu_routing");
        self.ensure_buffers()?;
        self.upload_inputs(reqs, temps)?;
        // drain any pre-capture cross-stream state: a pending event recorded
        // outside the capture would poison it (STREAM_CAPTURE_ISOLATION)
        if let Some(ev) = self.pending_fetch.take() {
            self.stream.wait(&ev)?;
        }
        self.stream.synchronize()?;
        self.copy_stream.synchronize()?;
        if self.graph_exec[nb] == 0 {
            // build: capture the pure-GPU stack
            let raw = cudarc::driver::sys::CUstream::from(self.stream.cu_stream());
            unsafe {
                let rc = cus::cuStreamBeginCapture_v2(
                    raw,
                    cus::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                );
                anyhow::ensure!(rc == cus::CUresult::CUDA_SUCCESS, "begin capture: {rc:?}");
            }
            self.enqueue_stack(nb)?;
            let mut graph: cus::CUgraph = std::ptr::null_mut();
            unsafe {
                let rc = cus::cuStreamEndCapture(raw, &mut graph);
                anyhow::ensure!(rc == cus::CUresult::CUDA_SUCCESS, "end capture: {rc:?}");
                let mut exec: cus::CUgraphExec = std::ptr::null_mut();
                let rc = cus::cuGraphInstantiateWithFlags(&mut exec, graph, 0);
                anyhow::ensure!(rc == cus::CUresult::CUDA_SUCCESS, "instantiate: {rc:?}");
                cus::cuGraphDestroy(graph);
                self.graph_exec[nb] = exec as usize;
            }
        }
        // a slot promoted during the previous replay must be settled before
        // any admit in this replay can hand it out as a hit
        if let Some(ev) = self.pending_fetch.take() {
            self.stream.wait(&ev)?;
            self.captured_events.push(ev);
        }
        unsafe {
            let raw = cudarc::driver::sys::CUstream::from(self.stream.cu_stream());
            let rc = cus::cuGraphLaunch(self.graph_exec[nb] as cus::CUgraphExec, raw);
            anyhow::ensure!(rc == cus::CUresult::CUDA_SUCCESS, "graph launch: {rc:?}");
        }
        let mut toks = vec![0i32; nb];
        self.stream
            .memcpy_dtoh(&self.tok_out_dev.slice(0..nb), &mut toks)?;
        self.stream.synchronize()?;
        for &(slot, _) in reqs {
            self.seq_pos[slot] += 1;
        }
        self.prof.tokens += nb as u64;
        Ok(toks.into_iter().map(|t| t as u32).collect())
    }

    /// Like forward_batch but samples on-device and downloads only the next
    /// token per sequence (4 bytes) instead of full logits. temps: per-entry
    /// sampling temperature (<=0 = greedy).
    pub fn forward_sample(
        &mut self,
        reqs: &[(usize, u32)],
        temps: &[f32],
    ) -> Result<Vec<u32>> {
        let nb = reqs.len();
        assert_eq!(temps.len(), nb);
        self.want_sample = true;
        {
            let mut td = self.temps_dev.slice_mut(0..nb);
            self.stream.memcpy_htod(temps, &mut td)?;
        }
        let _ = self.forward_batch(reqs)?;
        self.want_sample = false;
        let mut toks = vec![0i32; nb];
        self.stream
            .memcpy_dtoh(&self.tok_out_dev.slice(0..nb), &mut toks)?;
        self.stream.synchronize()?;
        Ok(toks.into_iter().map(|t| t as u32).collect())
    }

    /// One decode step for a set of (batch_slot, token) entries — the
    /// continuous-batching workhorse. All dense ops, the shared MLP, routed
    /// experts and the lm_head run batched on GPU; norms/rope/router/topk on
    /// CPU per sequence. Returns softcapped logits per entry.
    pub fn forward_batch(&mut self, reqs: &[(usize, u32)]) -> Result<Vec<Vec<f32>>> {
        let cfg = self.cfg.clone();
        let nb = reqs.len();
        anyhow::ensure!(nb >= 1 && nb <= self.max_batch, "bad batch size");
        let sync_prof = self.sync_prof;
        let mut t = Instant::now();
        macro_rules! lap {
            ($field:ident) => {{
                if sync_prof {
                    self.stream.synchronize()?;
                }
                let now = Instant::now();
                self.prof.$field += now.duration_since(t).as_micros() as u64;
                t = now;
            }};
        }
        let scale = (HIDDEN as f32).sqrt();
        let erb = q6k::row_bytes(HIDDEN);
        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(nb);
        for &(slot, token) in reqs {
            anyhow::ensure!(slot < self.max_batch, "bad slot");
            anyhow::ensure!(self.seq_pos[slot] < cfg.max_seq, "context overflow");
            let mut x = vec![0f32; HIDDEN];
            q6k::dequantize_row(
                &self.embed_cpu[token as usize * erb..(token as usize + 1) * erb],
                &mut x,
            );
            for v in x.iter_mut() {
                *v *= scale;
            }
            xs.push(x);
        }
        lap!(embed_us);

        // lazy batch-capacity buffers keyed by width
        macro_rules! q8buf {
            ($k:expr) => {{
                if !self.q8bufs.contains_key(&$k) {
                    self.q8bufs.insert(
                        $k,
                        self.stream
                            .alloc_zeros::<u8>(self.max_batch * $k / 32 * ft_cuda::Q8_BLK)?,
                    );
                }
            }};
        }
        macro_rules! ybuf {
            ($r:expr) => {{
                if !self.ybufs.contains_key(&$r) {
                    self.ybufs
                        .insert($r, self.stream.alloc_zeros::<f32>(self.max_batch * $r)?);
                }
            }};
        }

        // upload the embedded residual stream + per-seq pos/slot arrays
        let mut h_all = vec![0f32; nb * HIDDEN];
        for (i, x) in xs.iter().enumerate() {
            h_all[i * HIDDEN..(i + 1) * HIDDEN].copy_from_slice(x);
        }
        {
            let mut xr = self.x_res.slice_mut(0..nb * HIDDEN);
            self.stream.memcpy_htod(&h_all[..nb * HIDDEN], &mut xr)?;
        }
        let pos_h: Vec<i32> = reqs.iter().map(|&(sl, _)| self.seq_pos[sl] as i32).collect();
        let slot_h: Vec<i32> = reqs.iter().map(|&(sl, _)| sl as i32).collect();
        {
            let mut pd = self.pos_dev.slice_mut(0..nb);
            self.stream.memcpy_htod(&pos_h, &mut pd)?;
            let mut sd = self.slot_dev.slice_mut(0..nb);
            self.stream.memcpy_htod(&slot_h, &mut sd)?;
        }

        q8buf!(HIDDEN);
        if self.gpu_routing {
            // constant pair->sequence mapping for fixed nb
            let xi: Vec<i32> = (0..nb * cfg.topk).map(|p| (p / cfg.topk) as i32).collect();
            let mut xd = self.pair_xidx.slice_mut(0..nb * cfg.topk);
            self.stream.memcpy_htod(&xi, &mut xd)?;
            let mut sq = self.pair_seq.slice_mut(0..nb * cfg.topk);
            self.stream.memcpy_htod(&xi, &mut sq)?;
        }
        let mut routed_in_all = vec![0f32; nb * HIDDEN];
        let mut router_logits = vec![0f32; nb * cfg.n_experts];

        for l in 0..cfg.n_layers {
            let hd = cfg.head_dim(l);
            let kvh = cfg.kv_heads[l];
            let q_rows = cfg.n_heads * hd;
            let kv_dim = kvh * hd;
            let qkv_rows = self.layers[l].qkv_rows;
            let group = cfg.n_heads / kvh;
            let max_pos = reqs.iter().map(|&(sl, _)| self.seq_pos[sl]).max().unwrap();
            // --- attention block, device-side ---
            t = Instant::now();
            ybuf!(qkv_rows);
            self.gemv.rmsnorm_rows_dev(
                &self.stream, &self.x_res, Some(&self.layers[l].attn_norm_d),
                &mut self.sc_a, HIDDEN, cfg.eps, nb,
            )?;
            {
                let q8 = self.q8bufs.get_mut(&HIDDEN).unwrap();
                self.gemv.quantize_q8(&self.stream, &self.sc_a, HIDDEN, q8, HIDDEN, nb)?;
            }
            {
                let q8 = self.q8bufs.get(&HIDDEN).unwrap();
                let yd = self.ybufs.get_mut(&qkv_rows).unwrap();
                {
                    let (rbr, qo) = q4r_geom(HIDDEN);
                    self.gemv.gemv_q4r_idx(
                        &self.stream, &self.layers[l].qkv, 0, 0, &self.zeros_b, &self.dense_idx,
                        nb, q8, HIDDEN / 32, yd, qkv_rows, qkv_rows, HIDDEN, rbr, qo,
                    )?;
                }
            }
            {
                let yd = self.ybufs.get_mut(&qkv_rows).unwrap();
                let n_all = nb * qkv_rows;
                {
                    let mut qv = yd.slice_mut(0..n_all);
                    self.gemv.rmsnorm_heads_dev(
                        &self.stream, &mut qv, &self.layers[l].q_norm_d, true, hd, cfg.eps,
                        qkv_rows, cfg.n_heads, nb,
                    )?;
                }
                {
                    let mut kv2 = yd.slice_mut(q_rows..n_all);
                    self.gemv.rmsnorm_heads_dev(
                        &self.stream, &mut kv2, &self.layers[l].k_norm_d, true, hd, cfg.eps,
                        qkv_rows, kvh, nb,
                    )?;
                }
                {
                    let mut vv = yd.slice_mut(q_rows + kv_dim..n_all);
                    self.gemv.rmsnorm_heads_dev(
                        &self.stream, &mut vv, &self.layers[l].k_norm_d, false, hd, cfg.eps,
                        qkv_rows, kvh, nb,
                    )?;
                }
                let inv_freq = if cfg.swa[l] { &self.inv_freq_swa_d } else { &self.inv_freq_full_d };
                {
                    let mut qv = yd.slice_mut(0..n_all);
                    self.gemv.rope_heads_dev(
                        &self.stream, &mut qv, inv_freq, &self.pos_dev, hd, cfg.n_heads,
                        qkv_rows, nb,
                    )?;
                }
                {
                    let mut kv2 = yd.slice_mut(q_rows..n_all);
                    self.gemv.rope_heads_dev(
                        &self.stream, &mut kv2, inv_freq, &self.pos_dev, hd, kvh, qkv_rows, nb,
                    )?;
                }
            }
            {
                let yd = self.ybufs.get(&qkv_rows).unwrap();
                self.gemv.kv_append_dev(
                    &self.stream, yd, &mut self.k_pool[l], &self.slot_dev, &self.pos_dev,
                    q_rows, qkv_rows, kv_dim, cfg.max_seq, nb,
                )?;
                self.gemv.kv_append_dev(
                    &self.stream, yd, &mut self.v_pool[l], &self.slot_dev, &self.pos_dev,
                    q_rows + kv_dim, qkv_rows, kv_dim, cfg.max_seq, nb,
                )?;
                // contiguous q for the attention kernel
                self.gemv
                    .gather_rows_dev(&self.stream, yd, &mut self.attn_out_dev, qkv_rows, 0, q_rows, nb)?;
            }
            self.gemv.attn_decode_batch_dev(
                &self.stream, &self.k_pool[l], &self.v_pool[l], &self.attn_out_dev,
                &mut self.attn_partials, &mut self.sc_o, &self.slot_dev, &self.pos_dev,
                if cfg.swa[l] { cfg.window } else { 0 },
                kv_dim, hd, group, cfg.n_heads, cfg.max_seq, nb, max_pos + 1,
            )?;
            lap!(attn_cpu_us);

            // --- o_proj + post-attn norm + residual, device-side ---
            q8buf!(q_rows);
            ybuf!(HIDDEN);
            {
                let q8 = self.q8bufs.get_mut(&q_rows).unwrap();
                self.gemv
                    .quantize_q8(&self.stream, &self.sc_o, q_rows, q8, q_rows, nb)?;
            }
            {
                let q8 = self.q8bufs.get(&q_rows).unwrap();
                let yd = self.ybufs.get_mut(&HIDDEN).unwrap();
                {
                    let (rbr, qo) = q4r_geom(q_rows);
                    self.gemv.gemv_q4r_idx(
                        &self.stream, &self.layers[l].o, 0, 0, &self.zeros_b, &self.dense_idx,
                        nb, q8, q_rows / 32, yd, HIDDEN, HIDDEN, q_rows, rbr, qo,
                    )?;
                }
            }
            {
                let yd = self.ybufs.get(&HIDDEN).unwrap();
                self.gemv.rmsnorm_rows_dev(
                    &self.stream, yd, Some(&self.layers[l].post_attn_d), &mut self.sc_a,
                    HIDDEN, cfg.eps, nb,
                )?;
            }
            self.gemv
                .add_rows_dev(&self.stream, &mut self.x_res, &self.sc_a, HIDDEN, nb)?;
            lap!(o_gemv_us);

            // --- pre-ff norms, shared MLP, router — all async ---
            self.gemv.rmsnorm_rows_dev(
                &self.stream, &self.x_res, Some(&self.layers[l].ffn_norm_d), &mut self.sc_pf,
                HIDDEN, cfg.eps, nb,
            )?;
            self.gemv.rmsnorm_rows_dev(
                &self.stream, &self.x_res, Some(&self.layers[l].pre_ffw2_d), &mut self.sc_ri,
                HIDDEN, cfg.eps, nb,
            )?;
            let gu_rows = 2 * cfg.ffn;
            q8buf!(cfg.ffn);
            ybuf!(gu_rows);
            self.gemv
                .quantize_q8(&self.stream, &self.sc_pf, HIDDEN, &mut self.q8_preff, HIDDEN, nb)?;
            {
                let yd = self.ybufs.get_mut(&gu_rows).unwrap();
                {
                    let (rbr, qo) = q4r_geom(HIDDEN);
                    self.gemv.gemv_q4r_idx(
                        &self.stream, &self.layers[l].gate_up, 0, 0, &self.zeros_b, &self.dense_idx,
                        nb, &self.q8_preff, HIDDEN / 32, yd, gu_rows, gu_rows, HIDDEN, rbr, qo,
                    )?;
                }
            }
            {
                let yd = self.ybufs.get(&gu_rows).unwrap();
                self.gemv
                    .gelu_mul_grouped(&self.stream, yd, &mut self.act_dev, cfg.ffn, nb)?;
            }
            {
                let q8a = self.q8bufs.get_mut(&cfg.ffn).unwrap();
                self.gemv
                    .quantize_q8(&self.stream, &self.act_dev, cfg.ffn, q8a, cfg.ffn, nb)?;
            }
            {
                let q8a = self.q8bufs.get(&cfg.ffn).unwrap();
                let yd = self.ybufs.get_mut(&HIDDEN).unwrap();
                {
                    let (rbr, qo) = q4r_geom(cfg.ffn);
                    self.gemv.gemv_q4r_idx(
                        &self.stream, &self.layers[l].down, 0, 0, &self.zeros_b, &self.dense_idx,
                        nb, q8a, cfg.ffn / 32, yd, HIDDEN, HIDDEN, cfg.ffn, rbr, qo,
                    )?;
                }
            }
            // router: unscaled rms + folded-weight f32 gemv
            self.gemv
                .rmsnorm_rows_dev(&self.stream, &self.x_res, None, &mut self.sc_a, HIDDEN, cfg.eps, nb)?;
            self.gemv.gemv_f32_rows_dev(
                &self.stream, &self.layers[l].router_wf_d, &self.sc_a,
                &mut self.router_logits_dev, HIDDEN, cfg.n_experts, nb,
            )?;
            lap!(shared_mlp_us);

            if self.gpu_routing {
                // === fully device-side routing: no per-layer sync ===
                self.gemv
                    .quantize_q8(&self.stream, &self.sc_ri, HIDDEN, &mut self.q8_routed, HIDDEN, nb)?;
                self.gemv.topk_router_dev(
                    &self.stream, &self.router_logits_dev, &self.layers[l].expert_scale_d,
                    &mut self.moe_ids_dev, &mut self.pair_wts, cfg.n_experts, cfg.topk, nb,
                )?;
                lap!(router_us);
                if let Some(ev) = self.pending_fetch.take() {
                    self.stream.wait(&ev)?;
                }
                self.gemv.lru_admit_dev(
                    &self.stream, &self.moe_ids_dev, l, cfg.n_experts, self.n_slots,
                    self.cache_dptr, self.banks_dptr, self.moe.eb,
                    &mut self.lru_map, &mut self.lru_slot_key, &mut self.lru_slot_last,
                    &mut self.lru_clock, &mut self.pair_bases,
                    &mut self.promote_src_dev, &mut self.promote_dst_dev, nb * cfg.topk,
                )?;
                let ev_a = self.stream.record_event(None)?;
                self.copy_stream.wait(&ev_a)?;
                self.gemv.promote_experts_dev(
                    &self.copy_stream, &self.promote_src_dev, &self.promote_dst_dev,
                    self.banks_dptr, self.cache_dptr, self.moe.eb, nb * cfg.topk,
                )?;
                // two-phase: misses are read from slots, so the fetch must land first
                let ev_f = self.copy_stream.record_event(None)?;
                self.stream.wait(&ev_f)?;
                lap!(embed_us); // reused: admit+promote cost bucket

                let n_pairs = nb * cfg.topk;
                {
                    let mut r = self.routed_out_dev.slice_mut(0..nb * HIDDEN);
                    self.stream.memset_zeros(&mut r)?;
                }
                let gu_rows_e = 2 * cfg.moe_inter;
                {
                    let (rbr, qo) = q4r_geom(HIDDEN);
                    self.gemv.gemv_q4r_ptr(
                        &self.stream, &self.pair_bases, 0, &self.pair_xidx, n_pairs,
                        &self.q8_routed, HIDDEN / 32, &mut self.pair_y_gu, gu_rows_e,
                        gu_rows_e, HIDDEN, rbr, qo,
                    )?;
                }
                self.gemv.gelu_mul_grouped(
                    &self.stream, &self.pair_y_gu, &mut self.pair_act, cfg.moe_inter, n_pairs,
                )?;
                self.gemv.quantize_q8(
                    &self.stream, &self.pair_act, cfg.moe_inter, &mut self.pair_act_q8,
                    cfg.moe_inter, n_pairs,
                )?;
                {
                    let (rbr, qo) = q4r_geom(cfg.moe_inter);
                    self.gemv.gemv_q4r_ptr(
                        &self.stream, &self.pair_bases, self.moe.gu_bytes, &self.dense_idx,
                        n_pairs, &self.pair_act_q8, cfg.moe_inter / 32, &mut self.pair_y_down,
                        HIDDEN, HIDDEN, cfg.moe_inter, rbr, qo,
                    )?;
                }
                self.gemv.reduce_pairs_weighted(
                    &self.stream, &self.pair_y_down, &self.pair_wts, &self.pair_seq,
                    &mut self.routed_out_dev, HIDDEN, n_pairs, nb,
                )?;
                {
                    let sh2 = self.ybufs.get(&HIDDEN).unwrap();
                    self.gemv.dual_combine_dev(
                        &self.stream, &mut self.x_res, sh2, &self.routed_out_dev,
                        &self.layers[l].post_ffw1_d, &self.layers[l].post_ffw2_d,
                        &self.layers[l].post_ffw_d, self.layers[l].layer_scalar,
                        HIDDEN, cfg.eps, nb,
                    )?;
                }
                lap!(moe_us);
                continue;
            }

            // --- the one sync per layer: router logits + routed_in download ---
            {
                let rl = self.router_logits_dev.slice(0..nb * cfg.n_experts);
                self.stream
                    .memcpy_dtoh(&rl, &mut router_logits[..nb * cfg.n_experts])?;
                let ri = self.sc_ri.slice(0..nb * HIDDEN);
                self.stream.memcpy_dtoh(&ri, &mut routed_in_all)?;
                self.stream.synchronize()?;
            }
            let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(nb);
            let mut all_wts: Vec<Vec<f32>> = Vec::with_capacity(nb);
            for i in 0..nb {
                let lg = &router_logits[i * cfg.n_experts..(i + 1) * cfg.n_experts];
                let mut order: Vec<usize> = (0..cfg.n_experts).collect();
                order.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
                let top = &order[..cfg.topk];
                let mx = lg[top[0]];
                let exps: Vec<f32> = top.iter().map(|&e| (lg[e] - mx).exp()).collect();
                let denom: f32 = exps.iter().sum();
                all_ids.push(top.iter().map(|&e| e as u32).collect());
                all_wts.push(
                    top.iter()
                        .zip(&exps)
                        .map(|(&e, &xv)| xv / denom * self.layers[l].per_expert_scale[e])
                        .collect(),
                );
            }
            lap!(router_us);

            // --- routed experts: batched pairs ---
            self.gemv
                .quantize_q8(&self.stream, &self.sc_ri, HIDDEN, &mut self.q8_routed, HIDDEN, nb)?;
            // hits read the VRAM slot; misses read the pinned host bank over
            // UVA this step while a background promote fills the slot for
            // future steps. The promote is only awaited at the NEXT layer's
            // moe section, so nothing here blocks on PCIe.
            if let Some(ev) = self.pending_fetch.take() {
                self.stream.wait(&ev)?;
            }
            self.pair_bases_h.clear();
            self.pair_xidx_h.clear();
            self.pair_wts_h.clear();
            self.pair_seq_h.clear();
            let mut cpu_pairs: Vec<(usize, u32, f32)> = Vec::new();
            let mut any_fetch = false;
            for (i, ids) in all_ids.iter().enumerate() {
                let lk = self.moe.cache.lookup(l as u32, ids);
                let miss_ids: Vec<u32> = lk.misses.iter().map(|&(e, _)| e).collect();
                let (fetch_ids, cpu_ids) = split_misses(&miss_ids, self.moe.fraction);
                for &e in &cpu_ids {
                    self.moe.cache.forget(l as u32, e);
                    let w = all_wts[i][ids.iter().position(|&x2| x2 == e).unwrap()];
                    cpu_pairs.push((i, e, w));
                }
                let slot_of = |e: u32| lk.misses.iter().find(|&&(me, _)| me == e).unwrap().1;
                for &e in &fetch_ids {
                    let cslot = slot_of(e) as usize;
                    let off = (l * cfg.n_experts + e as usize) * self.moe.eb;
                    let hs = self.moe.banks.as_slice();
                    let mut dst = self
                        .moe
                        .cache_buf
                        .slice_mut(cslot * self.moe.eb..(cslot + 1) * self.moe.eb);
                    self.copy_stream
                        .memcpy_htod(&hs[off..off + self.moe.eb], &mut dst)?;
                    any_fetch = true;
                }
                let fetch_set: Vec<u32> = fetch_ids.clone();
                for &(e, cslot) in lk
                    .hits
                    .iter()
                    .chain(fetch_ids.iter().map(|&e| (e, slot_of(e))).collect::<Vec<_>>().iter())
                {
                    let w = all_wts[i][ids.iter().position(|&x2| x2 == e).unwrap()];
                    let base = if fetch_set.contains(&e) {
                        // freshly missed: read host bank via UVA this step
                        self.banks_dptr + ((l * cfg.n_experts + e as usize) * self.moe.eb) as u64
                    } else {
                        self.cache_dptr + (cslot as usize * self.moe.eb) as u64
                    };
                    self.pair_bases_h.push(base);
                    self.pair_xidx_h.push(i as i32);
                    self.pair_wts_h.push(w);
                    self.pair_seq_h.push(i as i32);
                }
            }
            if any_fetch {
                self.pending_fetch = Some(self.copy_stream.record_event(None)?);
            }
            let n_pairs = self.pair_bases_h.len();
            {
                let mut r = self.routed_out_dev.slice_mut(0..nb * HIDDEN);
                self.stream.memset_zeros(&mut r)?;
            }
            if n_pairs > 0 {
                {
                    let mut bd = self.pair_bases.slice_mut(0..n_pairs);
                    self.stream.memcpy_htod(&self.pair_bases_h, &mut bd)?;
                    let mut xd = self.pair_xidx.slice_mut(0..n_pairs);
                    self.stream.memcpy_htod(&self.pair_xidx_h, &mut xd)?;
                    let mut wd = self.pair_wts.slice_mut(0..n_pairs);
                    self.stream.memcpy_htod(&self.pair_wts_h, &mut wd)?;
                    let mut qd = self.pair_seq.slice_mut(0..n_pairs);
                    self.stream.memcpy_htod(&self.pair_seq_h, &mut qd)?;
                }
                let gu_rows_e = 2 * cfg.moe_inter;
                {
                    let (rbr, qo) = q4r_geom(HIDDEN);
                    self.gemv.gemv_q4r_ptr(
                        &self.stream, &self.pair_bases, 0, &self.pair_xidx, n_pairs,
                        &self.q8_routed, HIDDEN / 32, &mut self.pair_y_gu, gu_rows_e,
                        gu_rows_e, HIDDEN, rbr, qo,
                    )?;
                }
                self.gemv.gelu_mul_grouped(
                    &self.stream, &self.pair_y_gu, &mut self.pair_act, cfg.moe_inter, n_pairs,
                )?;
                self.gemv.quantize_q8(
                    &self.stream, &self.pair_act, cfg.moe_inter, &mut self.pair_act_q8,
                    cfg.moe_inter, n_pairs,
                )?;
                {
                    let (rbr, qo) = q4r_geom(cfg.moe_inter);
                    self.gemv.gemv_q4r_ptr(
                        &self.stream, &self.pair_bases, self.moe.gu_bytes, &self.dense_idx,
                        n_pairs, &self.pair_act_q8, cfg.moe_inter / 32, &mut self.pair_y_down,
                        HIDDEN, HIDDEN, cfg.moe_inter, rbr, qo,
                    )?;
                }
                self.gemv.reduce_pairs_weighted(
                    &self.stream, &self.pair_y_down, &self.pair_wts, &self.pair_seq,
                    &mut self.routed_out_dev, HIDDEN, n_pairs, nb,
                )?;
            }

            // CPU expert pairs -> upload + add into routed_out_dev
            if !cpu_pairs.is_empty() {
                let inter = cfg.moe_inter;
                let x8s: Vec<q4_0::Q8Vec> = (0..nb)
                    .map(|i| q4_0::Q8Vec::quantize(&routed_in_all[i * HIDDEN..(i + 1) * HIDDEN]))
                    .collect();
                let banks = self.moe.banks.as_slice();
                let eb = self.moe.eb;
                let gu_bytes = self.moe.gu_bytes;
                let n_experts = cfg.n_experts;
                let parts: Vec<(usize, Vec<f32>)> = cpu_pairs
                    .par_iter()
                    .map(|&(i, e, w)| {
                        let off = (l * n_experts + e as usize) * eb;
                        let mut ygu = vec![0f32; 2 * inter];
                        q4_0::gemv_q8_r(&banks[off..off + gu_bytes], &x8s[i], &mut ygu);
                        let mut act = vec![0f32; inter];
                        for j in 0..inter {
                            act[j] = gelu_tanh(ygu[j]) * ygu[j + inter];
                        }
                        let mut ydn = vec![0f32; HIDDEN];
                        q4_0::gemv_q8_r(
                            &banks[off + gu_bytes..off + eb],
                            &q4_0::Q8Vec::quantize(&act),
                            &mut ydn,
                        );
                        for v in ydn.iter_mut() {
                            *v *= w;
                        }
                        (i, ydn)
                    })
                    .collect();
                h_all[..nb * HIDDEN].fill(0.0);
                for (i, part) in parts {
                    for j in 0..HIDDEN {
                        h_all[i * HIDDEN + j] += part[j];
                    }
                }
                {
                    let mut v = self.sc_pf.slice_mut(0..nb * HIDDEN);
                    self.stream.memcpy_htod(&h_all[..nb * HIDDEN], &mut v)?;
                }
                self.gemv
                    .add_rows_dev(&self.stream, &mut self.routed_out_dev, &self.sc_pf, HIDDEN, nb)?;
            }

            // --- dual combine on GPU ---
            {
                let sh2 = self.ybufs.get(&HIDDEN).unwrap();
                self.gemv.dual_combine_dev(
                    &self.stream, &mut self.x_res, sh2, &self.routed_out_dev,
                    &self.layers[l].post_ffw1_d, &self.layers[l].post_ffw2_d,
                    &self.layers[l].post_ffw_d, self.layers[l].layer_scalar,
                    HIDDEN, cfg.eps, nb,
                )?;
            }
            lap!(moe_us);
        }

        // --- final norm + batched q4 lm_head + softcap ---
        self.gemv.rmsnorm_rows_dev(
            &self.stream, &self.x_res, Some(&self.final_norm_d), &mut self.sc_a,
            HIDDEN, cfg.eps, nb,
        )?;
        self.gemv
            .quantize_q8(&self.stream, &self.sc_a, HIDDEN, &mut self.q8_preff, HIDDEN, nb)?;
        {
            let (rbr, qo) = q4r_geom(HIDDEN);
            self.gemv.gemv_q4r_idx(
                &self.stream, &self.lm_head_q4, 0, 0, &self.zeros_b, &self.dense_idx,
                nb, &self.q8_preff, HIDDEN / 32, &mut self.logits_dev, cfg.vocab,
                cfg.vocab, HIDDEN, rbr, qo,
            )?;
        }
        lap!(combine_us);
        let cap = cfg.softcap;
        let mut out = Vec::with_capacity(nb);
        if self.want_sample {
            self.gemv.sample_tokens_dev(
                &self.stream, &self.logits_dev, &self.temps_dev, &mut self.rng_dev,
                &mut self.tok_out_dev, cfg.vocab, cap, nb,
            )?;
            for &(slot, _) in reqs {
                self.seq_pos[slot] += 1;
            }
        } else {
            let mut logits_all = vec![0f32; nb * cfg.vocab];
            self.stream
                .memcpy_dtoh(&self.logits_dev.slice(0..nb * cfg.vocab), &mut logits_all)?;
            self.stream.synchronize()?;
            for (i, &(slot, _)) in reqs.iter().enumerate() {
                let mut lg = logits_all[i * cfg.vocab..(i + 1) * cfg.vocab].to_vec();
                for v in lg.iter_mut() {
                    *v = (*v / cap).tanh() * cap;
                }
                out.push(lg);
                self.seq_pos[slot] += 1;
            }
        }
        self.prof.tokens += nb as u64;
        lap!(lm_head_us);
        Ok(out)
    }
}
