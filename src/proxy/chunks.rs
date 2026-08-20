use bytes::{BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageAccumulator {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

impl UsageAccumulator {
    pub fn update_from_json(&mut self, usage_val: &Value) {
        if let Some(prompt) = usage_val.get("prompt_tokens").and_then(|v| v.as_u64()) {
            self.prompt_tokens = Some(prompt);
        }
        if let Some(completion) = usage_val.get("completion_tokens").and_then(|v| v.as_u64()) {
            self.completion_tokens = Some(completion);
        }
        if let Some(total) = usage_val.get("total_tokens").and_then(|v| v.as_u64()) {
            self.total_tokens = Some(total);
        }
    }

    pub fn to_json(&self) -> Option<Value> {
        if self.prompt_tokens.is_some()
            || self.completion_tokens.is_some()
            || self.total_tokens.is_some()
        {
            let mut obj = serde_json::Map::new();
            if let Some(p) = self.prompt_tokens {
                obj.insert("prompt_tokens".to_string(), Value::from(p));
            }
            if let Some(c) = self.completion_tokens {
                obj.insert("completion_tokens".to_string(), Value::from(c));
            }
            if let Some(t) = self.total_tokens {
                obj.insert("total_tokens".to_string(), Value::from(t));
            }
            Some(Value::Object(obj))
        } else {
            None
        }
    }
}

pub fn format_sse_data(json: &Value) -> Bytes {
    let mut out = BytesMut::with_capacity(64);
    out.put_slice(b"data: ");
    let _ = serde_json::to_writer((&mut out).writer(), json);
    out.put_slice(b"\n\n");
    out.freeze()
}

pub fn format_sse_done() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}

pub fn format_sse_comment(comment: &str) -> Bytes {
    let mut out = BytesMut::with_capacity(comment.len() + 4);
    out.put_slice(b": ");
    out.put_slice(comment.as_bytes());
    out.put_slice(b"\n\n");
    out.freeze()
}

pub fn build_chunk_with_content(
    downstream_id: &str,
    created: i64,
    model: &str,
    content: &str,
    include_role: bool,
) -> Value {
    let mut delta = serde_json::Map::new();
    if include_role {
        delta.insert("role".to_string(), Value::String("assistant".to_string()));
    }
    delta.insert("content".to_string(), Value::String(content.to_string()));

    serde_json::json!({
        "id": downstream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": Value::Null
            }
        ]
    })
}

pub fn build_chunk_with_reasoning(
    downstream_id: &str,
    created: i64,
    model: &str,
    reasoning: &str,
    field_name: &str,
    include_role: bool,
) -> Value {
    let mut delta = serde_json::Map::new();
    if include_role {
        delta.insert("role".to_string(), Value::String("assistant".to_string()));
    }
    delta.insert(field_name.to_string(), Value::String(reasoning.to_string()));

    serde_json::json!({
        "id": downstream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": Value::Null
            }
        ]
    })
}

pub fn build_chunk_with_finish(
    downstream_id: &str,
    created: i64,
    model: &str,
    finish_reason: &str,
    usage: Option<&UsageAccumulator>,
) -> Value {
    let mut root = serde_json::json!({
        "id": downstream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }
        ]
    });

    if let Some(usage) = usage {
        if let Some(usage_json) = usage.to_json() {
            if let Some(obj) = root.as_object_mut() {
                obj.insert("usage".to_string(), usage_json);
            }
        }
    }

    root
}

/// Rewrite an upstream chunk JSON object to align with downstream invariants (§6.5).
pub fn rewrite_upstream_chunk(
    chunk_json: &mut Value,
    downstream_id: &str,
    created: i64,
    model: &str,
    emitted_role: &mut bool,
    is_terminal: bool,
) {
    if let Some(obj) = chunk_json.as_object_mut() {
        obj.insert("id".to_string(), Value::String(downstream_id.to_string()));
        obj.insert("created".to_string(), Value::Number(created.into()));
        obj.insert("model".to_string(), Value::String(model.to_string()));
        if !obj.contains_key("object") {
            obj.insert(
                "object".to_string(),
                Value::String("chat.completion.chunk".to_string()),
            );
        }

        if let Some(choices) = obj.get_mut("choices").and_then(|c| c.as_array_mut()) {
            if let Some(choice0) = choices.first_mut().and_then(|c| c.as_object_mut()) {
                // Role delta handling
                if let Some(delta) = choice0.get_mut("delta").and_then(|d| d.as_object_mut()) {
                    if delta.contains_key("role") {
                        if !*emitted_role {
                            *emitted_role = true;
                        } else {
                            delta.remove("role");
                        }
                    }
                }

                // Finish reason handling
                if !is_terminal {
                    choice0.insert("finish_reason".to_string(), Value::Null);
                }
            }
        }
    }
}

/// Extract content delta, reasoning delta, tool call deltas, finish_reason, and usage from a chunk.
#[derive(Debug, Default)]
pub struct ParsedChunkDetails {
    pub has_role: bool,
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub reasoning_field: Option<String>,
    pub tool_calls: Option<Vec<Value>>,
    pub finish_reason: Option<String>,
    pub model: Option<String>,
    pub usage: Option<Value>,
}

pub fn parse_chunk_details(chunk_json: &Value) -> ParsedChunkDetails {
    let mut details = ParsedChunkDetails::default();

    if let Some(model) = chunk_json.get("model").and_then(|m| m.as_str()) {
        details.model = Some(model.to_string());
    }

    if let Some(usage) = chunk_json.get("usage") {
        if !usage.is_null() {
            details.usage = Some(usage.clone());
        }
    }

    if let Some(choices) = chunk_json.get("choices").and_then(|c| c.as_array()) {
        if let Some(choice0) = choices.first() {
            if let Some(finish_reason) = choice0.get("finish_reason").and_then(|f| f.as_str()) {
                details.finish_reason = Some(finish_reason.to_string());
            }

            if let Some(delta) = choice0.get("delta") {
                details.has_role = delta.get("role").is_some();
                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        details.content = Some(content.to_string());
                    }
                }

                // Check reasoning content in standard or provider fields
                if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                    if !reasoning.is_empty() {
                        details.reasoning = Some(reasoning.to_string());
                        details.reasoning_field = Some("reasoning_content".to_string());
                    }
                }
                if details.reasoning.is_none() {
                    if let Some(reasoning) = delta.get("reasoning").and_then(|r| r.as_str()) {
                        if !reasoning.is_empty() {
                            details.reasoning = Some(reasoning.to_string());
                            details.reasoning_field = Some("reasoning".to_string());
                        }
                    }
                }

                if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                    if !tool_calls.is_empty() {
                        details.tool_calls = Some(tool_calls.clone());
                    }
                }
            }
        }
    }

    details
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §6.6: the three frame shapes, pinned byte-for-byte. These bytes are the
    /// entire contract with the downstream client, and each helper builds them
    /// by a different route, so assert the exact output of all three.
    #[test]
    fn test_sse_frame_shapes_are_exact() {
        assert_eq!(&format_sse_done()[..], b"data: [DONE]\n\n");
        assert_eq!(&format_sse_comment("retrying")[..], b": retrying\n\n");
        assert_eq!(&format_sse_comment("")[..], b": \n\n");
        assert_eq!(
            &format_sse_data(&serde_json::json!({"a": 1, "b": "x"}))[..],
            b"data: {\"a\":1,\"b\":\"x\"}\n\n"
        );
    }

    #[test]
    fn test_format_sse_data_frames_compact_json() {
        let chunk = build_chunk_with_content("id-1", 7, "m", "hi", true);
        let framed = format_sse_data(&chunk);

        assert!(framed.starts_with(b"data: "));
        assert!(framed.ends_with(b"\n\n"));
        // Byte-identical to serialising separately, and compact (no newlines
        // inside the payload, which would split the SSE frame).
        assert_eq!(
            framed,
            format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()).as_bytes()
        );
        assert_eq!(framed.iter().filter(|&&b| b == b'\n').count(), 2);
    }

    #[test]
    fn test_chunk_rewriting_role_suppression() {
        let mut chunk = serde_json::json!({
            "id": "upstream-1",
            "created": 100,
            "model": "upstream-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Hi"
                    },
                    "finish_reason": null
                }
            ]
        });

        let mut emitted_role = false;
        rewrite_upstream_chunk(
            &mut chunk,
            "downstream-fixed",
            200,
            "my-model",
            &mut emitted_role,
            false,
        );
        assert_eq!(chunk["id"], "downstream-fixed");
        assert_eq!(chunk["created"], 200);
        assert_eq!(chunk["model"], "my-model");
        assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
        assert!(emitted_role);

        // Next chunk on retry has role again:
        let mut chunk2 = serde_json::json!({
            "id": "upstream-2",
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": " there"
                    }
                }
            ]
        });
        rewrite_upstream_chunk(
            &mut chunk2,
            "downstream-fixed",
            200,
            "my-model",
            &mut emitted_role,
            false,
        );
        assert!(chunk2["choices"][0]["delta"].get("role").is_none());
        assert_eq!(chunk2["choices"][0]["delta"]["content"], " there");
    }

    #[test]
    fn test_finish_reason_suppression_when_not_terminal() {
        let mut chunk = serde_json::json!({
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        });

        let mut emitted_role = true;
        rewrite_upstream_chunk(
            &mut chunk,
            "downstream-id",
            200,
            "my-model",
            &mut emitted_role,
            false,
        );
        assert!(chunk["choices"][0]["finish_reason"].is_null());

        let mut chunk_term = serde_json::json!({
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        });
        rewrite_upstream_chunk(
            &mut chunk_term,
            "downstream-id",
            200,
            "my-model",
            &mut emitted_role,
            true,
        );
        assert_eq!(chunk_term["choices"][0]["finish_reason"], "stop");
    }
}
