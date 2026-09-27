//! OpenAI-compatible serving for freeToken-rs.
//!
//!   POST /v1/chat/completions   chat (stream + non-stream)
//!   POST /v1/completions        legacy text completion, raw prompt
//!   GET  /v1/models[/{id}]      the single loaded model
//!   GET  /health
//!
//! Supported request fields: messages/prompt, max_tokens,
//! max_completion_tokens, temperature, stop (<= 4), n (=1), stream,
//! stream_options.include_usage, echo (completions). Other fields (top_p,
//! seed, user, ...) are accepted and ignored. Responses carry real token
//! usage and finish_reason ("stop" / "length"); errors use OpenAI's
//! `{"error": {...}}` shape.
//!
//! The model lives on one worker thread that continuous-batches requests.
//! Usage:
//!
//!   serve gguf=/path/model.gguf [port=8080] [slots=1024] [fraction=0.2]
//!         [batch=4] [groute=0] [api_key=KEY | env FT_API_KEY]

use axum::{
    extract::{rejection::JsonRejection, Path, Request, State},
    http::{header, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use cudarc::driver::CudaContext;
use ft_gguf::Gguf;
use ft_model::openai::{
    check_common, error_body, render_chat, ChatRequest, CompletionRequest, FinishReason,
    StopMatcher, Usage, Utf8Stream,
};
use ft_model::{tokenizer::Tokenizer, Model};
use futures::stream::Stream;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
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

/// Worker -> request messages.
enum WorkerEvent {
    Token(u32),
    /// generation ended; None means the engine failed
    Done(Option<FinishReason>),
}

struct GenJob {
    prompt_ids: Vec<u32>,
    max_new: usize,
    temperature: f32,
    tx: mpsc::Sender<WorkerEvent>,
}

struct AppState {
    jobs: mpsc::Sender<GenJob>,
    tok: Tokenizer,
    model_name: String,
    max_seq: usize,
    api_key: Option<String>,
    created: u64,
}

struct ActiveSeq {
    slot: usize,
    tx: mpsc::Sender<WorkerEvent>,
    temperature: f32,
    remaining: usize,
    next_token: u32,
}

/// Continuous-batching worker: new requests are prefilled on admission (one
/// sequence at a time), then all active sequences decode together — one
/// forward_batch per step. Finished sequences free their slot immediately;
/// a dropped receiver (client gone, stop sequence hit) cancels its sequence.
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
            // the handler already sized max_new to fit; this is a safety net
            let budget = model.cfg.max_seq.saturating_sub(job.max_new);
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
                let _ = job.tx.send(WorkerEvent::Done(None));
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
            let finish = if next == tok.eos || Some(next) == tok.eot {
                Some(FinishReason::Stop)
            } else if seq.remaining == 0 {
                Some(FinishReason::Length)
            } else {
                None
            };
            let stop = match finish {
                Some(r) => {
                    let _ = seq.tx.send(WorkerEvent::Done(Some(r)));
                    true
                }
                None => seq.tx.send(WorkerEvent::Token(next)).is_err(),
            };
            if stop {
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
                    let _ = seq.tx.send(WorkerEvent::Done(None));
                    free_slots.push(seq.slot);
                }
            }
        }
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

fn new_id(prefix: &str) -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    format!("{prefix}-{:x}{:06x}", now(), SEQ.fetch_add(1, Ordering::Relaxed))
}

fn api_error(status: StatusCode, message: &str, param: Option<&str>, code: Option<&str>) -> Response {
    let kind = match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::NOT_FOUND => "not_found_error",
        s if s.is_server_error() => "server_error",
        _ => "invalid_request_error",
    };
    (status, Json(error_body(message, kind, param, code))).into_response()
}

fn bad_request(message: &str, param: Option<&str>) -> Response {
    api_error(StatusCode::BAD_REQUEST, message, param, None)
}

/// Decoded output of one generation, as seen by a request handler.
enum Out {
    Text(String),
    Done { finish: FinishReason, completion_tokens: usize },
    Failed,
}

/// A validated generation: prompt tokens, output budget, stop sequences.
struct Gen {
    prompt_ids: Vec<u32>,
    max_new: usize,
    temperature: f32,
    stops: Vec<String>,
}

impl Gen {
    fn new(
        st: &AppState,
        prompt_ids: Vec<u32>,
        max_tokens: Option<usize>,
        default_max: usize,
        temperature: Option<f32>,
        stops: Vec<String>,
    ) -> Result<Self, Response> {
        let room = st.max_seq.saturating_sub(prompt_ids.len());
        if room == 0 {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                &format!(
                    "This model's maximum context length is {} tokens. However, your prompt is {} tokens.",
                    st.max_seq,
                    prompt_ids.len()
                ),
                Some("messages"),
                Some("context_length_exceeded"),
            ));
        }
        // greedy by default (OpenAI defaults to 1.0; plain temperature sampling
        // without top-p is noticeably worse, so keep the engine's default)
        let temperature = temperature.unwrap_or(0.0);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(bad_request("temperature must be between 0 and 2", Some("temperature")));
        }
        Ok(Self {
            max_new: max_tokens.unwrap_or(default_max).min(room),
            prompt_ids,
            temperature,
            stops,
        })
    }

    /// Submit to the worker; decoded text, with stop sequences applied,
    /// arrives on the returned channel. Dropping it cancels generation.
    fn start(self, st: &Arc<AppState>) -> Result<tokio::sync::mpsc::UnboundedReceiver<Out>, Response> {
        let (tx, rx) = mpsc::channel::<WorkerEvent>();
        let job = GenJob {
            prompt_ids: self.prompt_ids,
            max_new: self.max_new,
            temperature: self.temperature,
            tx,
        };
        if st.jobs.send(job).is_err() {
            return Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, "inference worker is gone", None, None));
        }
        let (otx, orx) = tokio::sync::mpsc::unbounded_channel::<Out>();
        let st = st.clone();
        let stops = self.stops;
        tokio::task::spawn_blocking(move || {
            let mut utf8 = Utf8Stream::default();
            let mut stop = StopMatcher::new(stops);
            let mut n = 0usize;
            let emit = |s: String| s.is_empty() || otx.send(Out::Text(s)).is_ok();
            let end = loop {
                match rx.recv() {
                    Ok(WorkerEvent::Token(id)) => {
                        n += 1;
                        let (text, hit) = stop.push(&utf8.push(&st.tok.decode_bytes(&[id])));
                        if !emit(text) {
                            return; // client went away; dropping rx cancels
                        }
                        if hit {
                            break Out::Done { finish: FinishReason::Stop, completion_tokens: n };
                        }
                    }
                    Ok(WorkerEvent::Done(Some(finish))) => {
                        let (text, hit) = stop.push(&utf8.finish());
                        let text = if hit { text } else { text + &stop.finish() };
                        let _ = emit(text);
                        let finish = if hit { FinishReason::Stop } else { finish };
                        break Out::Done { finish, completion_tokens: n };
                    }
                    Ok(WorkerEvent::Done(None)) | Err(_) => break Out::Failed,
                }
            };
            let _ = otx.send(end);
        });
        Ok(orx)
    }
}

/// Collect a whole generation (non-streaming responses).
async fn collect(mut rx: tokio::sync::mpsc::UnboundedReceiver<Out>) -> Result<(String, FinishReason, usize), Response> {
    let mut text = String::new();
    while let Some(o) = rx.recv().await {
        match o {
            Out::Text(t) => text.push_str(&t),
            Out::Done { finish, completion_tokens } => return Ok((text, finish, completion_tokens)),
            Out::Failed => break,
        }
    }
    Err(api_error(StatusCode::INTERNAL_SERVER_ERROR, "generation failed", None, None))
}

/// Chunk shape differs between the two endpoints.
#[derive(Clone, Copy)]
enum Kind {
    Chat,
    Completion,
}

/// SSE stream in OpenAI's chunk format, ending with `data: [DONE]`.
fn sse(
    kind: Kind,
    rx: tokio::sync::mpsc::UnboundedReceiver<Out>,
    id: String,
    model: String,
    prompt_tokens: usize,
    include_usage: bool,
    prefix: Option<String>,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let created = now();
    let usage_null = if include_usage { Some(serde_json::Value::Null) } else { None };
    let chunk = move |text: Option<&str>, role: bool, finish: Option<FinishReason>| {
        let choice = match kind {
            Kind::Chat => {
                let mut delta = serde_json::Map::new();
                if role {
                    delta.insert("role".into(), json!("assistant"));
                }
                if let Some(t) = text {
                    delta.insert("content".into(), json!(t));
                }
                json!({"index": 0, "delta": delta, "logprobs": null, "finish_reason": finish})
            }
            Kind::Completion => {
                json!({"index": 0, "text": text.unwrap_or(""), "logprobs": null, "finish_reason": finish})
            }
        };
        let object = match kind {
            Kind::Chat => "chat.completion.chunk",
            Kind::Completion => "text_completion",
        };
        let mut v = json!({"id": id, "object": object, "created": created, "model": model, "choices": [choice]});
        if let Some(u) = &usage_null {
            v["usage"] = u.clone();
        }
        v
    };
    let usage_chunk = {
        let chunk = chunk.clone();
        move |completion_tokens: usize| {
            let mut v = chunk(None, false, None);
            v["choices"] = json!([]);
            v["usage"] = json!(Usage::new(prompt_tokens, completion_tokens));
            v
        }
    };
    let data = |v: serde_json::Value| Event::default().data(v.to_string());

    enum Phase {
        Start,
        Body,
        End,
    }
    let init = (rx, Phase::Start, Vec::<Event>::new());
    futures::stream::unfold(init, move |(mut rx, mut phase, mut queue)| {
        let chunk = chunk.clone();
        let usage_chunk = usage_chunk.clone();
        let prefix = prefix.clone();
        async move {
            loop {
                if !queue.is_empty() {
                    let ev = queue.remove(0);
                    return Some((Ok(ev), (rx, phase, queue)));
                }
                match phase {
                    Phase::End => return None,
                    Phase::Start => {
                        phase = Phase::Body;
                        match kind {
                            Kind::Chat => queue.push(data(chunk(Some(""), true, None))),
                            Kind::Completion => {
                                if let Some(p) = &prefix {
                                    queue.push(data(chunk(Some(p), false, None)));
                                }
                            }
                        }
                    }
                    Phase::Body => match rx.recv().await {
                        Some(Out::Text(t)) => queue.push(data(chunk(Some(&t), false, None))),
                        Some(Out::Done { finish, completion_tokens }) => {
                            let last = match kind {
                                Kind::Chat => chunk(None, false, Some(finish)),
                                Kind::Completion => chunk(Some(""), false, Some(finish)),
                            };
                            queue.push(data(last));
                            if include_usage {
                                queue.push(data(usage_chunk(completion_tokens)));
                            }
                            queue.push(Event::default().data("[DONE]"));
                            phase = Phase::End;
                        }
                        Some(Out::Failed) | None => {
                            queue.push(data(error_body("generation failed", "server_error", None, None)));
                            queue.push(Event::default().data("[DONE]"));
                            phase = Phase::End;
                        }
                    },
                }
            }
        }
    })
}

async fn chat(State(st): State<Arc<AppState>>, payload: Result<Json<ChatRequest>, JsonRejection>) -> Response {
    let req = match payload {
        Ok(Json(r)) => r,
        Err(e) => return bad_request(&e.body_text(), None),
    };
    let stops = match check_common(req.n, &req.stop) {
        Ok(s) => s,
        Err((msg, param)) => return bad_request(&msg, Some(param)),
    };
    let text = match render_chat(&req.messages) {
        Ok(t) => t,
        Err(msg) => return bad_request(&msg, Some("messages")),
    };
    let mut prompt_ids = vec![st.tok.bos];
    prompt_ids.extend(st.tok.encode_with_specials(&text));
    let prompt_tokens = prompt_ids.len();
    let max_tokens = req.max_completion_tokens.or(req.max_tokens);
    let gen = match Gen::new(&st, prompt_ids, max_tokens, usize::MAX, req.temperature, stops) {
        Ok(g) => g,
        Err(r) => return r,
    };
    let rx = match gen.start(&st) {
        Ok(rx) => rx,
        Err(r) => return r,
    };
    let id = new_id("chatcmpl");
    if req.stream {
        let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
        let s = sse(Kind::Chat, rx, id, st.model_name.clone(), prompt_tokens, include_usage, None);
        return Sse::new(s).into_response();
    }
    let (text, finish, completion_tokens) = match collect(rx).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": now(),
        "model": st.model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "logprobs": null,
            "finish_reason": finish
        }],
        "usage": Usage::new(prompt_tokens, completion_tokens)
    }))
    .into_response()
}

async fn completions(
    State(st): State<Arc<AppState>>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let req = match payload {
        Ok(Json(r)) => r,
        Err(e) => return bad_request(&e.body_text(), None),
    };
    let stops = match check_common(req.n, &req.stop) {
        Ok(s) => s,
        Err((msg, param)) => return bad_request(&msg, Some(param)),
    };
    let mut prompts = req.prompt.into_vec();
    if prompts.len() != 1 {
        return bad_request("exactly one prompt string is supported", Some("prompt"));
    }
    let prompt = prompts.pop().unwrap();
    let mut prompt_ids = vec![st.tok.bos];
    prompt_ids.extend(st.tok.encode_with_specials(&prompt));
    let prompt_tokens = prompt_ids.len();
    // OpenAI's legacy default is 16 tokens
    let gen = match Gen::new(&st, prompt_ids, req.max_tokens, 16, req.temperature, stops) {
        Ok(g) => g,
        Err(r) => return r,
    };
    let rx = match gen.start(&st) {
        Ok(rx) => rx,
        Err(r) => return r,
    };
    let id = new_id("cmpl");
    let prefix = req.echo.then(|| prompt.clone());
    if req.stream {
        let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
        let s = sse(Kind::Completion, rx, id, st.model_name.clone(), prompt_tokens, include_usage, prefix);
        return Sse::new(s).into_response();
    }
    let (text, finish, completion_tokens) = match collect(rx).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    Json(json!({
        "id": id,
        "object": "text_completion",
        "created": now(),
        "model": st.model_name,
        "choices": [{
            "index": 0,
            "text": prefix.unwrap_or_default() + &text,
            "logprobs": null,
            "finish_reason": finish
        }],
        "usage": Usage::new(prompt_tokens, completion_tokens)
    }))
    .into_response()
}

fn model_card(st: &AppState) -> serde_json::Value {
    json!({"id": st.model_name, "object": "model", "created": st.created, "owned_by": "freetoken-rs"})
}

async fn models(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({"object": "list", "data": [model_card(&st)]}))
}

async fn model_info(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    if id == st.model_name {
        Json(model_card(&st)).into_response()
    } else {
        api_error(
            StatusCode::NOT_FOUND,
            &format!("The model '{id}' does not exist"),
            Some("model"),
            Some("model_not_found"),
        )
    }
}

/// Bearer-token check (when an API key is configured) plus permissive CORS
/// so browser-based OpenAI clients can talk to the server directly.
async fn gate(State(st): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let mut resp = if req.method() == Method::OPTIONS {
        StatusCode::NO_CONTENT.into_response()
    } else {
        let authorized = st.api_key.as_deref().is_none_or(|key| {
            req.headers()
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                == Some(key)
        });
        if authorized || req.uri().path() == "/health" {
            next.run(req).await
        } else {
            api_error(StatusCode::UNAUTHORIZED, "Incorrect API key provided", None, Some("invalid_api_key"))
        }
    };
    let h = resp.headers_mut();
    h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS"));
    h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static("*"));
    resp
}

fn main() -> anyhow::Result<()> {
    let path = arg_s("gguf", "");
    anyhow::ensure!(!path.is_empty(), "gguf=<path> required");
    let port = arg("port", 8080.0) as u16;
    let slots = arg("slots", 1024.0) as usize;
    let fraction = arg("fraction", 0.2);
    let api_key = Some(arg_s("api_key", &std::env::var("FT_API_KEY").unwrap_or_default()))
        .filter(|k| !k.is_empty());
    let model_name = std::path::Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("freetoken-rs")
        .to_string();

    eprintln!("loading {path} ...");
    let g = Gguf::open(&path)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let tok2 = Tokenizer::from_gguf(&g)?;
    let ctx = CudaContext::new(0)?;
    let batch = arg("batch", 4.0) as usize;
    let mut model = Model::load(&g, &ctx, slots, fraction, batch)?;
    model.gpu_routing = arg("groute", 0.0) != 0.0;
    let max_seq = model.cfg.max_seq;
    drop(g);
    eprintln!(
        "model loaded; serving '{model_name}' on 0.0.0.0:{port}{}",
        if api_key.is_some() { " (API key required)" } else { "" }
    );

    let (jtx, jrx) = mpsc::channel::<GenJob>();
    std::thread::spawn(move || worker(model, tok2, jrx));

    let state = Arc::new(AppState { jobs: jtx, tok, model_name, max_seq, api_key, created: now() });
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let app = Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/v1/completions", post(completions))
            .route("/v1/models", get(models))
            .route("/v1/models/{id}", get(model_info))
            .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
            .layer(middleware::from_fn_with_state(state.clone(), gate))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })
}
