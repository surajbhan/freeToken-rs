//! Batched-vs-single equivalence test: seq 0's continuation must be
//! identical whether it decodes alone or alongside another sequence.
use anyhow::Result;
use cudarc::driver::CudaContext;
use ft_gguf::Gguf;
use ft_model::{tokenizer::Tokenizer, Model};

fn arg_s(name: &str, default: &str) -> String {
    std::env::args()
        .find_map(|a| a.strip_prefix(&format!("{name}=")).map(str::to_string))
        .unwrap_or_else(|| default.to_string())
}

fn greedy(l: &[f32]) -> u32 {
    l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
}

fn main() -> Result<()> {
    let g = Gguf::open(arg_s("gguf", ""))?;
    let tok = Tokenizer::from_gguf(&g)?;
    let ctx = CudaContext::new(0)?;
    let mut m = Model::load(&g, &ctx, 2048, 0.43, 2)?;

    // sanity: dense_idx content
    let idx = m.debug_dense_idx()?;
    println!("dense_idx[..8] = {:?}", &idx[..8]);

    let p0: Vec<u32> = {
        let mut v = vec![tok.bos];
        v.extend(tok.encode_with_specials(&tok.chat_prompt("What is 7 times 8?")));
        v
    };
    let p1: Vec<u32> = {
        let mut v = vec![tok.bos];
        v.extend(tok.encode_with_specials(&tok.chat_prompt("Name the largest planet.")));
        v
    };

    // run A: seq0 alone
    m.reset_slot(0);
    let mut lg = Vec::new();
    for &t in &p0 {
        lg = m.forward_batch(&[(0, t)])?.pop().unwrap();
    }
    let mut a_text = Vec::new();
    for _ in 0..8 {
        let n = greedy(&lg);
        a_text.push(n);
        lg = m.forward_batch(&[(0, n)])?.pop().unwrap();
    }
    println!("A (solo): {:?}", tok.decode(&a_text));

    // run B: seq0 + seq1 batched
    m.reset_slot(0);
    m.reset_slot(1);
    let mut lg0 = Vec::new();
    for &t in &p0 {
        lg0 = m.forward_batch(&[(0, t)])?.pop().unwrap();
    }
    let mut lg1 = Vec::new();
    for &t in &p1 {
        lg1 = m.forward_batch(&[(1, t)])?.pop().unwrap();
    }
    let mut b0 = Vec::new();
    let mut b1 = Vec::new();
    for _ in 0..8 {
        let n0 = greedy(&lg0);
        let n1 = greedy(&lg1);
        b0.push(n0);
        b1.push(n1);
        let outs = m.forward_batch(&[(0, n0), (1, n1)])?;
        lg1 = outs[1].clone();
        lg0 = outs[0].clone();
    }
    println!("B seq0 (batched): {:?}", tok.decode(&b0));
    println!("B seq1 (batched): {:?}", tok.decode(&b1));
    println!("seq0 match: {}", a_text == b0);
    Ok(())
}
