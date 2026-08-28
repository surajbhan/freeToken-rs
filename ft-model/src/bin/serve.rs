//! OpenAI-compatible serving for freeToken-rs.
//!
//! POST /v1/chat/completions (stream + non-stream), GET /v1/models.
//! The model lives on one worker thread; requests are serialized through a
//! channel (bs=1 decode engine). Usage:
//!
//!   serve gguf=/path/model.gguf [port=8080] [slots=1024] [fraction=0.2]

use axum::{
    extract::State,
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use cudarc::driver::CudaContext;
use ft_gguf::Gguf;
use ft_model::{tokenizer::Tokenizer, Model};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::{mpsc, Arc};
use std::time::{SystemTime, UNIX_EPOCH};

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

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
}

struct GenJob {
    prompt_ids: Vec<u32>,
    max_new: usize,
    temperature: f32,
    /// per-token text; None terminates with the finish reason
    tx: mpsc::Sender<Option<String>>,
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<GenJob>,
    model_name: String,
}

fn sample(logits: &[f32], temperature: f32, rng: &mut u64) -> u32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
    }
    let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let probs: Vec<f32> = logits.iter().map(|&l| ((l - mx) / temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let mut r = (*rng >> 11) as f32 / (1u64 << 53) as f32 * sum;
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

struct ActiveSeq {
    slot: usize,
    tx: mpsc::Sender<Option<String>>,
    temperature: f32,
    remaining: usize,
    next_token: u32,
}

/// Continuous-batching worker: new requests are prefilled on admission (one
/// sequence at a time), then all active sequences decode together — one
/// forward_batch per step. Finished sequences free their slot immediately.
fn worker(mut model: Model, tok: Tokenizer, rx: mpsc::Receiver<GenJob>) {
    let max_batch = model.max_batch;
    let mut active: Vec<ActiveSeq> = Vec::new();
    let mut free_slots: Vec<usize> = (0..max_batch).rev().collect();
    loop {
        // admit while slots are free (block only when fully idle)
        loop {
            let job = if active.is_empty() {
                match rx.recv() {
                    Ok(j) => j,
                    Err(_) => return,
                }
            } else if free_slots.is_empty() {
                break;
            } else {
                match rx.try_recv() {
                    Ok(j) => j,
                    Err(_) => break,
                }
            };
            let slot = free_slots.pop().unwrap();
            model.reset_slot(slot);
            let budget = model.cfg.max_seq.saturating_sub(job.max_new + 2);
            let prompt = &job.prompt_ids[..job.prompt_ids.len().min(budget)];
            let mut next = 0u32;
            let mut ok = !prompt.is_empty();
            for &id in prompt {
                match if model.gpu_routing {
                    model.forward_sample_graphed(&[(slot, id)], &[job.temperature])
                } else {
                    model.forward_sample(&[(slot, id)], &[job.temperature])
                } {
                    Ok(t) => next = t[0],
                    Err(e) => {
                        eprintln!("prefill error: {e:#}");
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                let _ = job.tx.send(None);
                free_slots.push(slot);
                continue;
            }
            active.push(ActiveSeq {
                slot,
                tx: job.tx,
                temperature: job.temperature,
                remaining: job.max_new,
                next_token: next,
            });
        }

        // one batched decode step (tokens pre-sampled on device)
        let mut reqs: Vec<(usize, u32)> = Vec::new();
        let mut temps: Vec<f32> = Vec::new();
        let mut keep: Vec<bool> = Vec::with_capacity(active.len());
        for seq in active.iter_mut() {
            let next = seq.next_token;
            let stop = next == tok.eos
                || Some(next) == tok.eot
                || seq.remaining == 0
                || seq.tx.send(Some(tok.decode(&[next]))).is_err();
            if stop {
                let _ = seq.tx.send(None);
                keep.push(false);
            } else {
                seq.remaining -= 1;
                reqs.push((seq.slot, next));
                temps.push(seq.temperature);
                keep.push(true);
            }
        }
        let mut ki = keep.iter();
        active.retain(|seq| {
            let k = *ki.next().unwrap();
            if !k {
                free_slots.push(seq.slot);
            }
            k
        });
        if reqs.is_empty() {
            continue;
        }
        match if model.gpu_routing {
            model.forward_sample_graphed(&reqs, &temps)
        } else {
            model.forward_sample(&reqs, &temps)
        } {
            Ok(toks) => {
                for (seq, t2) in active.iter_mut().zip(toks) {
                    seq.next_token = t2;
                }
            }
            Err(e) => {
                eprintln!("decode error: {e:#}");
                for seq in active.drain(..) {
                    let _ = seq.tx.send(None);
                    free_slots.push(seq.slot);
                }
            }
        }
    }
}

fn build_prompt(tok: &Tokenizer, messages: &[ChatMessage]) -> Vec<u32> {
    let mut text = String::new();
    for m in messages {
        let role = match m.role.as_str() {
            "assistant" => "model",
            "system" => "user", // gemma folds system guidance into a user turn
            r => r,
        };
        text.push_str(&format!("<|turn>{role}\n{}<turn|>\n", m.content));
    }
    text.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    let mut ids = vec![tok.bos];
    ids.extend(tok.encode_with_specials(&text));
    ids
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

async fn chat(State(st): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Response {
    let tok_holder = TOKENIZER.get().unwrap();
    let prompt_ids = build_prompt(tok_holder, &req.messages);
    let max_new = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(512)
        .min(4096);
    let temperature = req.temperature.unwrap_or(0.0);
    let (tx, rx) = mpsc::channel::<Option<String>>();
    if st
        .jobs
        .send(GenJob { prompt_ids, max_new, temperature, tx })
        .is_err()
    {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "worker gone").into_response();
    }

    let id = format!("chatcmpl-{}", now());
    let model_name = st.model_name.clone();
    if req.stream {
        let stream = sse_stream(rx, id, model_name);
        Sse::new(stream).into_response()
    } else {
        let text = tokio::task::spawn_blocking(move || {
            let mut out = String::new();
            while let Ok(Some(t)) = rx.recv() {
                out.push_str(&t);
            }
            out
        })
        .await
        .unwrap_or_default();
        Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": now(),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }))
        .into_response()
    }
}

fn sse_stream(
    rx: mpsc::Receiver<Option<String>>,
    id: String,
    model: String,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (atx, arx) = tokio::sync::mpsc::unbounded_channel::<Option<String>>();
    tokio::task::spawn_blocking(move || {
        while let Ok(item) = rx.recv() {
            let end = item.is_none();
            if atx.send(item).is_err() || end {
                break;
            }
        }
    });
    let created = now();
    futures::stream::unfold(
        (arx, false, id, model),
        move |(mut arx, done, id, model)| async move {
            if done {
                return None;
            }
            match arx.recv().await {
                Some(Some(text)) => {
                    let ev = Event::default().data(
                        json!({
                            "id": id, "object": "chat.completion.chunk", "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                        })
                        .to_string(),
                    );
                    Some((Ok(ev), (arx, false, id, model)))
                }
                _ => {
                    let ev = Event::default().data("[DONE]");
                    Some((Ok(ev), (arx, true, id, model)))
                }
            }
        },
    )
}

async fn models(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": st.model_name, "object": "model", "created": now(), "owned_by": "freetoken-rs"}]
    }))
}

static TOKENIZER: std::sync::OnceLock<Tokenizer> = std::sync::OnceLock::new();

fn main() -> anyhow::Result<()> {
    let path = arg_s("gguf", "");
    anyhow::ensure!(!path.is_empty(), "gguf=<path> required");
    let port = arg("port", 8080.0) as u16;
    let slots = arg("slots", 1024.0) as usize;
    let fraction = arg("fraction", 0.2);
    let model_name = std::path::Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("freetoken-rs")
        .to_string();

    eprintln!("loading {path} ...");
    let g = Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    TOKENIZER.set(tok).ok();
    let tok2 = Tokenizer::from_gguf(&g)?;
    let ctx = CudaContext::new(0)?;
    let batch = arg("batch", 4.0) as usize;
    let mut model = Model::load(&g, &ctx, slots, fraction, batch)?;
    model.gpu_routing = arg("groute", 0.0) != 0.0;
    drop(g);
    eprintln!("model loaded; serving on 0.0.0.0:{port}");

    let (jtx, jrx) = mpsc::channel::<GenJob>();
    std::thread::spawn(move || worker(model, tok2, jrx));

    let state = Arc::new(AppState { jobs: jtx, model_name });
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let app = Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/v1/models", get(models))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })
}
