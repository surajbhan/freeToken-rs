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
            max_seq: 8192,
        })
    }

    pub fn head_dim(&self, layer: usize) -> usize {
        if self.swa[layer] { self.head_dim_swa } else { self.head_dim_full }
    }
    pub fn rope_base(&self, layer: usize) -> f32 {
        if self.swa[layer] { self.rope_base_swa } else { self.rope_base_full }
    }
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
    pub embed_gpu: CudaSlice<u8>,
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
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    inv_freq_swa: Vec<f32>,
    inv_freq_full: Vec<f32>,
    pub seq_len: usize,
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
    ) -> Result<Self> {
        let cfg = Config::from_gguf(g)?;
        let stream = ctx.default_stream();
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
            let qkv = stream.memcpy_stod(&qkv_host)?;
            let qkv_rows = q_rows + 2 * kv_rows;

            let o = stream.memcpy_stod(g.tensor_data(&t("attn_output.weight"))?)?;

            let gd = g.tensor_data(&t("ffn_gate.weight"))?;
            let ud = g.tensor_data(&t("ffn_up.weight"))?;
            let mut gu_host = Vec::with_capacity(gd.len() + ud.len());
            gu_host.extend_from_slice(gd);
            gu_host.extend_from_slice(ud);
            let gate_up = stream.memcpy_stod(&gu_host)?;
            let down = stream.memcpy_stod(g.tensor_data(&t("ffn_down.weight"))?)?;

            layers.push(LayerWeights {
                qkv,
                qkv_rows,
                o,
                gate_up,
                down,
                attn_norm: to_f32(g, &t("attn_norm.weight"))?,
                q_norm: to_f32(g, &t("attn_q_norm.weight"))?,
                k_norm: to_f32(g, &t("attn_k_norm.weight"))?,
                post_attn_norm: to_f32(g, &t("post_attention_norm.weight"))?,
                ffn_norm: to_f32(g, &t("ffn_norm.weight"))?,
                pre_ffw2: to_f32(g, &t("pre_ffw_norm_2.weight"))?,
                post_ffw: to_f32(g, &t("post_ffw_norm.weight"))?,
                post_ffw1: to_f32(g, &t("post_ffw_norm_1.weight"))?,
                post_ffw2: to_f32(g, &t("post_ffw_norm_2.weight"))?,
                layer_scalar: to_f32(g, &t("layer_output_scale.weight"))?[0],
                router_w: to_f32(g, &t("ffn_gate_inp.weight"))?,
                router_scale: to_f32(g, &t("ffn_gate_inp.scale"))?,
                per_expert_scale: to_f32(g, &t("ffn_down_exps.scale"))?,
            });
        }

        // embedding (q6_k, row per token) doubles as the tied lm_head on GPU
        let et = g.tensor("token_embd.weight")?;
        anyhow::ensure!(et.ggml_type == 14, "embedding must be q6_k");
        let embed_cpu = g.tensor_data("token_embd.weight")?.to_vec();
        let embed_gpu = stream.memcpy_stod(&embed_cpu)?;
        let final_norm = to_f32(g, "output_norm.weight")?;

        // MoE expert banks
        let gu_rows = 2 * cfg.moe_inter;
        let gu_bytes = rb_h * gu_rows;
        let down_bytes = q4_0::row_bytes(cfg.moe_inter) * HIDDEN;
        let eb = gu_bytes + down_bytes;
        let mut banks = HostBanks::new(ctx, cfg.n_layers * cfg.n_experts * eb)?;
        {
            let hs = banks.as_mut_slice();
            for l in 0..cfg.n_layers {
                let gu = g.tensor_data(&format!("blk.{l}.ffn_gate_up_exps.weight"))?;
                let dn = g.tensor_data(&format!("blk.{l}.ffn_down_exps.weight"))?;
                for e in 0..cfg.n_experts {
                    let dst =
                        &mut hs[(l * cfg.n_experts + e) * eb..(l * cfg.n_experts + e + 1) * eb];
                    dst[..gu_bytes].copy_from_slice(&gu[e * gu_bytes..(e + 1) * gu_bytes]);
                    dst[gu_bytes..].copy_from_slice(&dn[e * down_bytes..(e + 1) * down_bytes]);
                }
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
        Ok(Self {
            prof: Profile::default(),
            zero_slot: stream.alloc_zeros::<i32>(1)?,
            xbufs: HashMap::new(),
            q8bufs: HashMap::new(),
            ybufs: HashMap::new(),
            act_dev: stream.alloc_zeros::<f32>(cfg.ffn)?,
            logits_dev: stream.alloc_zeros::<f32>(cfg.vocab)?,
            k_cache: kv_dims.iter().map(|d| vec![0f32; cfg.max_seq * d]).collect(),
            v_cache: kv_dims.iter().map(|d| vec![0f32; cfg.max_seq * d]).collect(),
            inv_freq_swa: build_inv_freq(cfg.head_dim_swa, cfg.rope_base_swa),
            inv_freq_full: build_inv_freq(cfg.head_dim_full, cfg.rope_base_full),
            seq_len: 0,
            cfg,
            layers,
            final_norm,
            embed_cpu,
            embed_gpu,
            moe,
            gemv,
            stream,
            copy_stream,
        })
    }

    /// One decode step for the token at position `self.seq_len`. Returns
    /// softcapped logits.
    pub fn forward_token(&mut self, token: u32) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let pos = self.seq_len;
        anyhow::ensure!(pos < cfg.max_seq, "context overflow");

        let mut t = Instant::now();
        macro_rules! lap {
            ($field:ident) => {{
                let now = Instant::now();
                self.prof.$field += now.duration_since(t).as_micros() as u64;
                t = now;
            }};
        }
        // embedding row * sqrt(hidden)
        let erb = q6k::row_bytes(HIDDEN);
        let mut x = vec![0f32; HIDDEN];
        q6k::dequantize_row(
            &self.embed_cpu[token as usize * erb..(token as usize + 1) * erb],
            &mut x,
        );
        let scale = (HIDDEN as f32).sqrt();
        for v in x.iter_mut() {
            *v *= scale;
        }

        lap!(embed_us);
        let mut h = vec![0f32; HIDDEN];
        for l in 0..cfg.n_layers {
            let hd = cfg.head_dim(l);
            let kvh = cfg.kv_heads[l];
            let q_rows = cfg.n_heads * hd;
            let kv_dim = kvh * hd;
            let kv_start = if cfg.swa[l] {
                (pos + 1).saturating_sub(cfg.window)
            } else {
                0
            };

            // --- attention sandwich ---
            t = Instant::now();
            rmsnorm_into(&x, Some(&self.layers[l].attn_norm), cfg.eps, &mut h);
            lap!(attn_norms_us);
            let mut qkv = vec![0f32; self.layers[l].qkv_rows];
            dense_gemv(
                &self.stream,
                &self.gemv,
                &mut self.xbufs,
                &mut self.q8bufs,
                &mut self.ybufs,
                &self.zero_slot,
                &self.layers[l].qkv,
                &h,
                self.layers[l].qkv_rows,
                HIDDEN,
                &mut qkv,
            )?;
            lap!(qkv_gemv_us);

            let (q, rest) = qkv.split_at_mut(q_rows);
            let (k, v) = rest.split_at_mut(kv_dim);
            for qh in q.chunks_exact_mut(hd) {
                rmsnorm_inplace(qh, Some(&self.layers[l].q_norm), cfg.eps);
            }
            for kh in k.chunks_exact_mut(hd) {
                rmsnorm_inplace(kh, Some(&self.layers[l].k_norm), cfg.eps);
            }
            for vh in v.chunks_exact_mut(hd) {
                rmsnorm_inplace(vh, None, cfg.eps);
            }
            let inv_freq = if cfg.swa[l] { &self.inv_freq_swa } else { &self.inv_freq_full };
            rope_neox(q, hd, pos, inv_freq);
            rope_neox(k, hd, pos, inv_freq);

            self.k_cache[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k);
            self.v_cache[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(v);
            lap!(attn_norms_us);

            // GQA attention on CPU (sm_scale = 1.0 per reference config)
            let group = cfg.n_heads / kvh;
            let kc = &self.k_cache[l];
            let vc = &self.v_cache[l];
            let mut o = vec![0f32; q_rows];
            o.par_chunks_exact_mut(hd)
                .enumerate()
                .for_each(|(qh_idx, oh)| {
                    let gh = qh_idx / group;
                    let qh = &q[qh_idx * hd..(qh_idx + 1) * hd];
                    let n = pos + 1 - kv_start;
                    let mut sc = vec![0f32; n];
                    for (i, t) in (kv_start..=pos).enumerate() {
                        let kt = &kc[t * kv_dim + gh * hd..t * kv_dim + (gh + 1) * hd];
                        sc[i] = qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>();
                    }
                    let m = sc.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut denom = 0f32;
                    for s in sc.iter_mut() {
                        *s = (*s - m).exp();
                        denom += *s;
                    }
                    for (i, t) in (kv_start..=pos).enumerate() {
                        let vt = &vc[t * kv_dim + gh * hd..t * kv_dim + (gh + 1) * hd];
                        let wgt = sc[i] / denom;
                        for (od, vv) in oh.iter_mut().zip(vt) {
                            *od += wgt * vv;
                        }
                    }
                });
            lap!(attn_cpu_us);

            let mut attn = vec![0f32; HIDDEN];
            dense_gemv(
                &self.stream,
                &self.gemv,
                &mut self.xbufs,
                &mut self.q8bufs,
                &mut self.ybufs,
                &self.zero_slot,
                &self.layers[l].o,
                &o,
                HIDDEN,
                q_rows,
                &mut attn,
            )?;
            lap!(o_gemv_us);
            rmsnorm_inplace(&mut attn, Some(&self.layers[l].post_attn_norm), cfg.eps);

            // x = residual + attn; pre_ff = norm(x)
            for i in 0..HIDDEN {
                x[i] += attn[i];
            }
            let mut pre_ff = vec![0f32; HIDDEN];
            rmsnorm_into(&x, Some(&self.layers[l].ffn_norm), cfg.eps, &mut pre_ff);

            // --- shared MLP: gate_up -> gelu*up -> down, all on GPU, 1 sync ---
            let gu_rows = 2 * cfg.ffn;
            for (map_k, len) in [(HIDDEN, HIDDEN), (cfg.ffn, cfg.ffn)] {
                if !self.q8bufs.contains_key(&map_k) {
                    self.q8bufs
                        .insert(map_k, self.stream.alloc_zeros::<u8>(len / 32 * ft_cuda::Q8_BLK)?);
                }
            }
            if !self.xbufs.contains_key(&HIDDEN) {
                self.xbufs.insert(HIDDEN, self.stream.alloc_zeros::<f32>(HIDDEN)?);
            }
            for rows in [gu_rows, HIDDEN] {
                if !self.ybufs.contains_key(&rows) {
                    self.ybufs.insert(rows, self.stream.alloc_zeros::<f32>(rows)?);
                }
            }
            {
                let xd = self.xbufs.get_mut(&HIDDEN).unwrap();
                self.stream.memcpy_htod(&pre_ff, xd)?;
            }
            {
                let xd = self.xbufs.get(&HIDDEN).unwrap();
                let q8 = self.q8bufs.get_mut(&HIDDEN).unwrap();
                self.gemv.quantize_q8(&self.stream, xd, 0, q8, HIDDEN, 1)?;
            }
            {
                let q8 = self.q8bufs.get(&HIDDEN).unwrap();
                let yd = self.ybufs.get_mut(&gu_rows).unwrap();
                self.gemv.gemv_grouped_q8(
                    &self.stream, &self.layers[l].gate_up, 0, 0, &self.zero_slot, 1,
                    q8, 0, yd, gu_rows, gu_rows, HIDDEN,
                )?;
            }
            {
                let yd = self.ybufs.get(&gu_rows).unwrap();
                self.gemv
                    .gelu_mul_grouped(&self.stream, yd, &mut self.act_dev, cfg.ffn, 1)?;
                let q8a = self.q8bufs.get_mut(&cfg.ffn).unwrap();
                self.gemv
                    .quantize_q8(&self.stream, &self.act_dev, 0, q8a, cfg.ffn, 1)?;
            }
            let mut shared = vec![0f32; HIDDEN];
            {
                let q8a = self.q8bufs.get(&cfg.ffn).unwrap();
                let yd = self.ybufs.get_mut(&HIDDEN).unwrap();
                self.gemv.gemv_grouped_q8(
                    &self.stream, &self.layers[l].down, 0, 0, &self.zero_slot, 1,
                    q8a, 0, yd, HIDDEN, HIDDEN, cfg.ffn,
                )?;
                let yd = self.ybufs.get(&HIDDEN).unwrap();
                self.stream.memcpy_dtoh(yd, &mut shared)?;
                self.stream.synchronize()?;
            }
            lap!(shared_mlp_us);

            // --- router (CPU) ---
            let mut rh = vec![0f32; HIDDEN];
            rmsnorm_into(&x, None, cfg.eps, &mut rh);
            let root = (HIDDEN as f32).powf(-0.5);
            for i in 0..HIDDEN {
                rh[i] *= self.layers[l].router_scale[i] * root;
            }
            let rw = &self.layers[l].router_w;
            let mut logits: Vec<(f32, usize)> = (0..cfg.n_experts)
                .into_par_iter()
                .map(|e| {
                    let row = &rw[e * HIDDEN..(e + 1) * HIDDEN];
                    let mut acc = [0f32; 8];
                    for (cr, cx) in row.chunks_exact(8).zip(rh.chunks_exact(8)) {
                        for j in 0..8 {
                            acc[j] += cr[j] * cx[j];
                        }
                    }
                    (acc.iter().sum::<f32>(), e)
                })
                .collect();
            logits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let top = &logits[..cfg.topk];
            let mx = top[0].0;
            let exps: Vec<f32> = top.iter().map(|(v, _)| (v - mx).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let routed_ids: Vec<u32> = top.iter().map(|&(_, e)| e as u32).collect();
            let routed_wts: Vec<f32> = top
                .iter()
                .zip(&exps)
                .map(|(&(_, e), &xv)| xv / denom * self.layers[l].per_expert_scale[e])
                .collect();

            lap!(router_us);
            // --- routed experts through the offload engine ---
            let mut routed_in = vec![0f32; HIDDEN];
            rmsnorm_into(&x, Some(&self.layers[l].pre_ffw2), cfg.eps, &mut routed_in);
            let routed = self.moe_forward(l, &routed_ids, &routed_wts, &routed_in)?;
            lap!(moe_us);

            // --- dual-rmsnorm combine ---
            let mut n1 = vec![0f32; HIDDEN];
            rmsnorm_into(&shared, Some(&self.layers[l].post_ffw1), cfg.eps, &mut n1);
            let mut n2 = vec![0f32; HIDDEN];
            rmsnorm_into(&routed, Some(&self.layers[l].post_ffw2), cfg.eps, &mut n2);
            for i in 0..HIDDEN {
                n1[i] += n2[i];
            }
            let combined = n1.clone();
            rmsnorm_into(&combined, Some(&self.layers[l].post_ffw), cfg.eps, &mut n1);
            let ls = self.layers[l].layer_scalar;
            for i in 0..HIDDEN {
                x[i] = (x[i] + n1[i]) * ls;
            }
            lap!(combine_us);
        }

        // final norm + tied q6_k lm_head + softcap
        let xin = x.clone();
        rmsnorm_into(&xin, Some(&self.final_norm), cfg.eps, &mut x);
        if !self.xbufs.contains_key(&HIDDEN) {
            self.xbufs
                .insert(HIDDEN, self.stream.alloc_zeros::<f32>(HIDDEN)?);
        }
        let xd = self.xbufs.get_mut(&HIDDEN).unwrap();
        self.stream.memcpy_htod(&x, xd)?;
        {
            let xd = self.xbufs.get(&HIDDEN).unwrap();
            let q8 = self.q8bufs.get_mut(&HIDDEN).unwrap();
            self.gemv.quantize_q8(&self.stream, xd, 0, q8, HIDDEN, 1)?;
        }
        let q8 = self.q8bufs.get(&HIDDEN).unwrap();
        self.gemv
            .gemv_q6k_q8(&self.stream, &self.embed_gpu, q8, &mut self.logits_dev, cfg.vocab, HIDDEN)?;
        let mut logits = vec![0f32; cfg.vocab];
        self.stream.memcpy_dtoh(&self.logits_dev, &mut logits)?;
        self.stream.synchronize()?;
        let cap = cfg.softcap;
        for v in logits.iter_mut() {
            *v = (*v / cap).tanh() * cap;
        }
        lap!(lm_head_us);
        self.prof.tokens += 1;
        self.seq_len += 1;
        Ok(logits)
    }

    /// Hybrid expert forward for one layer: returns the routed-branch output.
    fn moe_forward(
        &mut self,
        layer: usize,
        ids: &[u32],
        wts: &[f32],
        x_in: &[f32],
    ) -> Result<Vec<f32>> {
        let inter = self.cfg.moe_inter;
        let lk = self.moe.cache.lookup(layer as u32, ids);
        let miss_ids: Vec<u32> = lk.misses.iter().map(|&(e, _)| e).collect();
        let (fetch_ids, cpu_ids) = split_misses(&miss_ids, self.moe.fraction);
        // CPU-served misses never land in their assigned GPU slot: un-admit
        // them or the next route becomes a false hit on stale slot contents.
        for &e in &cpu_ids {
            self.moe.cache.forget(layer as u32, e);
        }
        let slot_of = |e: u32| lk.misses.iter().find(|&&(me, _)| me == e).unwrap().1;
        let wt_of = |e: u32| wts[ids.iter().position(|&i| i == e).unwrap()];

        for &e in &fetch_ids {
            let slot = slot_of(e) as usize;
            let off = (layer * self.cfg.n_experts + e as usize) * self.moe.eb;
            let hs = self.moe.banks.as_slice();
            let mut dst = self
                .moe
                .cache_buf
                .slice_mut(slot * self.moe.eb..(slot + 1) * self.moe.eb);
            self.copy_stream
                .memcpy_htod(&hs[off..off + self.moe.eb], &mut dst)?;
        }
        let ev = self.copy_stream.record_event(None)?;
        self.stream.wait(&ev)?;

        let gpu: Vec<(u32, u32)> = lk
            .hits
            .iter()
            .copied()
            .chain(fetch_ids.iter().map(|&e| (e, slot_of(e))))
            .collect();
        let n = gpu.len();
        self.stream.memset_zeros(&mut self.moe.out_dev)?;
        self.stream.memcpy_htod(x_in, &mut self.moe.x_dev)?;
        if n > 0 {
            let slots: Vec<i32> = gpu.iter().map(|&(_, s)| s as i32).collect();
            let gwts: Vec<f32> = gpu.iter().map(|&(e, _)| wt_of(e)).collect();
            {
                let mut sd = self.moe.slots_dev.slice_mut(0..n);
                self.stream.memcpy_htod(&slots, &mut sd)?;
                let mut wd = self.moe.wts_dev.slice_mut(0..n);
                self.stream.memcpy_htod(&gwts, &mut wd)?;
            }
            let gu_rows = 2 * inter;
            self.gemv
                .quantize_q8(&self.stream, &self.moe.x_dev, 0, &mut self.moe.x_q8, HIDDEN, 1)?;
            self.gemv.gemv_grouped_q8(
                &self.stream,
                &self.moe.cache_buf,
                self.moe.eb,
                0,
                &self.moe.slots_dev,
                n,
                &self.moe.x_q8,
                0,
                &mut self.moe.y_gu,
                gu_rows,
                gu_rows,
                HIDDEN,
            )?;
            self.gemv
                .gelu_mul_grouped(&self.stream, &self.moe.y_gu, &mut self.moe.act, inter, n)?;
            self.gemv
                .quantize_q8(&self.stream, &self.moe.act, inter, &mut self.moe.act_q8, inter, n)?;
            self.gemv.gemv_grouped_q8(
                &self.stream,
                &self.moe.cache_buf,
                self.moe.eb,
                self.moe.gu_bytes,
                &self.moe.slots_dev,
                n,
                &self.moe.act_q8,
                inter / 32,
                &mut self.moe.y_down,
                HIDDEN,
                HIDDEN,
                inter,
            )?;
            self.gemv.reduce_expert_weighted(
                &self.stream,
                &self.moe.y_down,
                &self.moe.wts_dev,
                &mut self.moe.out_dev,
                HIDDEN,
                n,
            )?;
        }

        let mut cpu_part = vec![0f32; HIDDEN];
        if !cpu_ids.is_empty() {
            let x8 = q4_0::Q8Vec::quantize(x_in);
            let mut ygu = vec![0f32; 2 * inter];
            let mut act = vec![0f32; inter];
            let mut ydn = vec![0f32; HIDDEN];
            for &e in &cpu_ids {
                let off = (layer * self.cfg.n_experts + e as usize) * self.moe.eb;
                let hs = self.moe.banks.as_slice();
                q4_0::gemv_q8(&hs[off..off + self.moe.gu_bytes], &x8, &mut ygu);
                for i in 0..inter {
                    act[i] = gelu_tanh(ygu[i]) * ygu[i + inter];
                }
                q4_0::gemv(&hs[off + self.moe.gu_bytes..off + self.moe.eb], &act, &mut ydn);
                let w = wt_of(e);
                for i in 0..HIDDEN {
                    cpu_part[i] += w * ydn[i];
                }
            }
        }

        let mut out = vec![0f32; HIDDEN];
        self.stream.memcpy_dtoh(&self.moe.out_dev, &mut out)?;
        self.stream.synchronize()?;
        for i in 0..HIDDEN {
            out[i] += cpu_part[i];
        }
        Ok(out)
    }
}
