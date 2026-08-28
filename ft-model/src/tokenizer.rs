//! Gemma-4 GGUF tokenizer: SentencePiece-flavored BPE. Vocabulary + merge
//! ranks come straight from the GGUF metadata; spaces become ▁, unknown
//! characters fall back to <0xXX> byte tokens.

use anyhow::{bail, Context, Result};
use ft_gguf::{Gguf, Value};
use std::collections::HashMap;

pub struct Tokenizer {
    pub vocab: Vec<String>,
    id_of: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), u32>,
    pub bos: u32,
    pub eos: u32,
    /// <end_of_turn>, the chat-format stop token
    pub eot: Option<u32>,
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let toks = match g.meta.get("tokenizer.ggml.tokens") {
            Some(Value::Arr(v)) => v,
            _ => bail!("missing tokenizer.ggml.tokens"),
        };
        let vocab: Vec<String> = toks
            .iter()
            .map(|t| t.as_str().unwrap_or_default().to_string())
            .collect();
        let id_of: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        let mut merge_rank = HashMap::new();
        if let Some(Value::Arr(merges)) = g.meta.get("tokenizer.ggml.merges") {
            for (rank, m) in merges.iter().enumerate() {
                let m = m.as_str().unwrap_or_default();
                // merges are "left right"; pieces may themselves contain
                // spaces only via ▁, so the single ASCII space is the split
                if let Some(sp) = m.rfind(' ') {
                    merge_rank.insert(
                        (m[..sp].to_string(), m[sp + 1..].to_string()),
                        rank as u32,
                    );
                }
            }
        }
        let get_id = |k: &str| g.meta.get(k).and_then(|v| v.as_u64()).map(|v| v as u32);
        Ok(Self {
            bos: get_id("tokenizer.ggml.bos_token_id").context("bos id")?,
            eos: get_id("tokenizer.ggml.eos_token_id").context("eos id")?,
            eot: id_of.get("<turn|>").copied(),
            vocab,
            id_of,
            merge_rank,
        })
    }

    /// BPE over the ▁-normalized text. Special tokens (chat markers) must be
    /// spliced by the caller via `encode_with_specials`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let norm = text.replace(' ', "\u{2581}");
        let mut pieces: Vec<String> = norm.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..pieces.len().saturating_sub(1) {
                if let Some(&r) = self
                    .merge_rank
                    .get(&(pieces[i].clone(), pieces[i + 1].clone()))
                {
                    if best.map_or(true, |(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            match best {
                Some((_, i)) => {
                    let merged = format!("{}{}", pieces[i], pieces[i + 1]);
                    pieces[i] = merged;
                    pieces.remove(i + 1);
                }
                None => break,
            }
        }
        let mut ids = Vec::new();
        for p in pieces {
            if let Some(&id) = self.id_of.get(&p) {
                ids.push(id);
            } else {
                // byte fallback
                for b in p.bytes() {
                    if let Some(&id) = self.id_of.get(&format!("<0x{b:02X}>")) {
                        ids.push(id);
                    }
                }
            }
        }
        ids
    }

    /// Encode text that may contain special tokens like <start_of_turn>:
    /// exact vocab hits for the special strings, BPE for everything between.
    pub fn encode_with_specials(&self, text: &str) -> Vec<u32> {
        let specials = [
            "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>",
            "<bos>", "<eos>",
        ];
        let mut ids = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            let mut first: Option<(usize, &str)> = None;
            for sp in specials {
                if let Some(pos) = rest.find(sp) {
                    if first.map_or(true, |(fp, _)| pos < fp) {
                        first = Some((pos, sp));
                    }
                }
            }
            match first {
                Some((pos, sp)) => {
                    if pos > 0 {
                        ids.extend(self.encode(&rest[..pos]));
                    }
                    match self.id_of.get(sp) {
                        Some(&id) => ids.push(id),
                        None => ids.extend(self.encode(sp)),
                    }
                    rest = &rest[pos + sp.len()..];
                }
                None => {
                    ids.extend(self.encode(rest));
                    break 'outer;
                }
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let piece = &self.vocab[id as usize];
            if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&piece[3..5], 16) {
                    bytes.push(b);
                    continue;
                }
            }
            bytes.extend(piece.replace('\u{2581}', " ").as_bytes());
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Gemma-4 canonical chat format for a single user turn (as rendered by
    /// the GGUF's own chat template with add_generation_prompt, no thinking).
    pub fn chat_prompt(&self, user: &str) -> String {
        format!(
            "<|turn>user\n{user}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        )
    }
}
