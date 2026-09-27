//! OpenAI API wire types and the GPU-free pieces of the serving path:
//! request parsing, Gemma-4 chat templating, incremental UTF-8 decoding and
//! stop-sequence matching. `bin/serve.rs` wires these to the decode worker.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A string or an array of strings (`stop`, legacy `prompt`).
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum StrOrVec {
    One(String),
    Many(Vec<String>),
}

impl StrOrVec {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StrOrVec::One(s) => vec![s],
            StrOrVec::Many(v) => v,
        }
    }
}

/// Message content: a plain string or an array of typed parts.
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize, Clone, Debug)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Content>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// Fields shared by both endpoints. Unknown fields (`top_p`, `seed`, `user`,
/// `logit_bias`, ...) are accepted and ignored, as most OpenAI-compatible
/// servers do; sampling here is temperature-only.
#[derive(Deserialize, Clone, Debug)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stop: Option<StrOrVec>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: StrOrVec,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stop: Option<StrOrVec>,
    #[serde(default)]
    pub echo: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Serialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt: usize, completion: usize) -> Self {
        Self { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion }
    }
}

/// OpenAI-style error body: `{"error": {"message", "type", "param", "code"}}`.
pub fn error_body(message: &str, kind: &str, param: Option<&str>, code: Option<&str>) -> Value {
    json!({"error": {"message": message, "type": kind, "param": param, "code": code}})
}

/// Validate `n` and `stop` (OpenAI allows at most 4 stop sequences).
pub fn check_common(n: Option<usize>, stop: &Option<StrOrVec>) -> Result<Vec<String>, (String, &'static str)> {
    if let Some(n) = n {
        if n != 1 {
            return Err(("only n=1 is supported".into(), "n"));
        }
    }
    let stops: Vec<String> = stop
        .clone()
        .map(StrOrVec::into_vec)
        .unwrap_or_default()
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
    if stops.len() > 4 {
        return Err(("at most 4 stop sequences are allowed".into(), "stop"));
    }
    Ok(stops)
}

fn message_text(m: &ChatMessage) -> Result<String, String> {
    match &m.content {
        None => Ok(String::new()),
        Some(Content::Text(s)) => Ok(s.clone()),
        Some(Content::Parts(parts)) => {
            let mut out = String::new();
            for p in parts {
                match (p.kind.as_str(), &p.text) {
                    ("text", Some(t)) => out.push_str(t),
                    (k, _) => return Err(format!("unsupported content part type '{k}' (text only)")),
                }
            }
            Ok(out)
        }
    }
}

/// Render a conversation in the Gemma-4 chat format, ending with an open
/// model turn (thinking disabled). System/developer messages have no turn of
/// their own in Gemma, so they are prepended to the first user turn.
pub fn render_chat(messages: &[ChatMessage]) -> Result<String, String> {
    if messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    let mut system = String::new();
    let mut text = String::new();
    for m in messages {
        let body = message_text(m)?;
        let role = match m.role.as_str() {
            "system" | "developer" => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(&body);
                continue;
            }
            "assistant" => "model",
            "user" | "tool" | "function" => "user",
            r => return Err(format!("unsupported message role '{r}'")),
        };
        let body = if role == "user" && !system.is_empty() {
            let merged = format!("{system}\n\n{body}");
            system.clear();
            merged
        } else {
            body
        };
        text.push_str(&format!("<|turn>{role}\n{body}<turn|>\n"));
    }
    if !system.is_empty() {
        // system prompt with no user turn after it
        text.push_str(&format!("<|turn>user\n{system}<turn|>\n"));
    }
    text.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    Ok(text)
}

/// Turns a byte stream into text, holding back an incomplete trailing UTF-8
/// sequence (byte-fallback tokens can split a character across tokens).
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            // genuinely invalid bytes: emit lossily rather than stall
            Err(_) => {
                let s = String::from_utf8_lossy(&self.pending).into_owned();
                self.pending.clear();
                return s;
            }
        };
        let rest = self.pending.split_off(valid);
        String::from_utf8(std::mem::replace(&mut self.pending, rest)).unwrap()
    }

    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// Stop-sequence matcher over streamed text. `push` returns text that is
/// safe to emit (cannot be the start of a stop sequence) and whether a stop
/// sequence was hit; text from the stop sequence onward is never emitted.
pub struct StopMatcher {
    stops: Vec<String>,
    buf: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self { stops, buf: String::new() }
    }

    pub fn push(&mut self, text: &str) -> (String, bool) {
        self.buf.push_str(text);
        if self.stops.is_empty() {
            return (std::mem::take(&mut self.buf), false);
        }
        if let Some(pos) = self.stops.iter().filter_map(|s| self.buf.find(s.as_str())).min() {
            let out = self.buf[..pos].to_string();
            self.buf.clear();
            return (out, true);
        }
        // hold back the longest suffix that is a prefix of some stop
        let mut hold = 0;
        for s in &self.stops {
            for (i, _) in s.char_indices().skip(1) {
                if i > hold && self.buf.ends_with(&s[..i]) {
                    hold = i;
                }
            }
        }
        let cut = self.buf.len() - hold;
        let out = self.buf[..cut].to_string();
        self.buf.drain(..cut);
        (out, false)
    }

    /// Flush held-back text at end of generation.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.into(), content: Some(Content::Text(content.into())) }
    }

    #[test]
    fn chat_template_folds_system() {
        let t = render_chat(&[msg("system", "Be brief."), msg("user", "Hi"), msg("assistant", "Hello"), msg("user", "Sky?")]).unwrap();
        assert_eq!(
            t,
            "<|turn>user\nBe brief.\n\nHi<turn|>\n<|turn>model\nHello<turn|>\n<|turn>user\nSky?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    #[test]
    fn content_parts_parse() {
        let req: ChatRequest = serde_json::from_value(json!({
            "model": "x", "top_p": 0.9,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
        }))
        .unwrap();
        assert!(render_chat(&req.messages).unwrap().starts_with("<|turn>user\nab<turn|>"));
        let img: ChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        }))
        .unwrap();
        assert!(render_chat(&img.messages).is_err());
    }

    #[test]
    fn stop_matcher_across_chunks() {
        let mut m = StopMatcher::new(vec!["END".into()]);
        assert_eq!(m.push("hello E"), ("hello ".into(), false));
        assert_eq!(m.push("N"), ("".into(), false));
        assert_eq!(m.push("Dtrailing"), ("".into(), true));
        let mut m = StopMatcher::new(vec!["END".into()]);
        assert_eq!(m.push("xE"), ("x".into(), false));
        assert_eq!(m.push("y"), ("Ey".into(), false));
        assert_eq!(m.push("E"), ("".into(), false));
        assert_eq!(m.finish(), "E");
    }

    #[test]
    fn utf8_split_across_tokens() {
        let mut u = Utf8Stream::default();
        let e = "é".as_bytes();
        assert_eq!(u.push(&e[..1]), "");
        assert_eq!(u.push(&e[1..]), "é");
        assert_eq!(u.finish(), "");
    }

    #[test]
    fn validation() {
        assert!(check_common(Some(2), &None).is_err());
        let five = Some(StrOrVec::Many(vec!["a".into(); 5]));
        assert!(check_common(None, &five).is_err());
        assert_eq!(check_common(Some(1), &Some(StrOrVec::One("x".into()))).unwrap(), vec!["x"]);
    }
}
