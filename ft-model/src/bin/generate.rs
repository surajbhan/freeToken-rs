//! Greedy/sampled generation from a Gemma-4 Q4_0 GGUF through the
//! freeToken-rs engine.
//!
//! Usage: generate gguf=/path/model.gguf prompt="..." [n=64] [slots=1024]
//!        [fraction=0.2] [temp=0.0] [raw=1 (skip chat template)]

use anyhow::Result;
use cudarc::driver::CudaContext;
use ft_gguf::Gguf;
use ft_model::{tokenizer::Tokenizer, Model};
use std::io::Write;
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
    let path = arg_s("gguf", "");
    anyhow::ensure!(!path.is_empty(), "gguf=<path> required");
    let prompt = arg_s("prompt", "Why is the sky blue?");
    let n = arg("n", 64.0) as usize;
    let slots = arg("slots", 1024.0) as usize;
    let fraction = arg("fraction", 0.2);
    let temp = arg("temp", 0.0) as f32;
    let raw = arg("raw", 0.0) != 0.0;

    let t0 = Instant::now();
    let g = Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let ctx = CudaContext::new(0)?;
    let mut model = Model::load(&g, &ctx, slots, fraction, 1)?;
    model.gpu_routing = arg("groute", 0.0) != 0.0;
    let graphed = arg("graph", 0.0) != 0.0;
    eprintln!(
        "loaded {} layers, {} experts, cache {slots} slots in {:.1}s",
        model.cfg.n_layers,
        model.cfg.n_experts,
        t0.elapsed().as_secs_f64()
    );

    let text = if raw { prompt.clone() } else { tok.chat_prompt(&prompt) };
    let mut ids = vec![tok.bos];
    ids.extend(tok.encode_with_specials(&text));
    eprintln!("prompt: {} tokens", ids.len());

    // prefill (sequential; on-device sampling, 4-byte downloads)
    let t1 = Instant::now();
    let mut next = 0u32;
    for &id in &ids {
        next = if graphed {
            model.forward_sample_graphed(&[(0, id)], &[temp])?[0]
        } else {
            model.forward_sample(&[(0, id)], &[temp])?[0]
        };
    }
    eprintln!(
        "prefill: {:.2}s ({:.0} ms/tok)",
        t1.elapsed().as_secs_f64(),
        t1.elapsed().as_secs_f64() * 1000.0 / ids.len() as f64
    );

    // decode
    let t2 = Instant::now();
    let mut out_ids = Vec::new();
    for _ in 0..n {
        if next == tok.eos || Some(next) == tok.eot {
            break;
        }
        out_ids.push(next);
        print!("{}", tok.decode(&[next]));
        std::io::stdout().flush()?;
        next = if graphed {
            model.forward_sample_graphed(&[(0, next)], &[temp])?[0]
        } else {
            model.forward_sample(&[(0, next)], &[temp])?[0]
        };
    }
    println!();
    let dt = t2.elapsed().as_secs_f64();
    eprintln!(
        "decode: {} tokens in {:.2}s -> {:.2} tok/s | cache hit rate {:.1}%",
        out_ids.len(),
        dt,
        out_ids.len() as f64 / dt,
        model.moe.cache.hit_rate() * 100.0
    );
    eprintln!("{}", model.prof.report());
    Ok(())
}
