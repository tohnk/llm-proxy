use crate::config::ContinuationMode;
use serde_json::Value;

pub const INSTRUCTED_USER_PROMPT: &str =
    "Continue your previous response from exactly where it stopped. Do not repeat any text, do not restate or summarise, do not acknowledge this instruction — output only the continuation.";

/// Build the request body for an upstream attempt.
/// If `accumulated_text` is empty, the original request body is cloned unmodified.
/// Otherwise, messages are extended according to the continuation mode and max_tokens is adjusted.
pub fn build_continuation_request(
    original_body: &Value,
    accumulated_text: &str,
    mode: ContinuationMode,
) -> Value {
    if accumulated_text.is_empty() {
        return original_body.clone();
    }

    let mut body = original_body.clone();

    // 1. Rewrite `messages`
    if let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        match mode {
            ContinuationMode::Prefix => {
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": accumulated_text
                }));
            }
            ContinuationMode::Instructed => {
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": accumulated_text
                }));
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": INSTRUCTED_USER_PROMPT
                }));
            }
        }
    }

    // 2. Adjust max_tokens or max_completion_tokens if present
    let estimated_tokens_emitted = ((accumulated_text.chars().count() / 4).max(1)) as u64;

    for key in ["max_tokens", "max_completion_tokens"] {
        if let Some(budget) = body.get_mut(key) {
            if let Some(remaining) = budget.as_u64() {
                *budget = Value::from(remaining.saturating_sub(estimated_tokens_emitted).max(1));
            }
        }
    }

    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_accumulated_returns_original() {
        let orig = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "custom_field": "custom_value"
        });
        let result = build_continuation_request(&orig, "", ContinuationMode::Prefix);
        assert_eq!(result, orig);
    }

    #[test]
    fn test_prefix_continuation() {
        let orig = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Count to 10"}],
            "max_tokens": 100
        });
        let result = build_continuation_request(&orig, "1, 2, 3, 4", ContinuationMode::Prefix);
        let msgs = result["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "1, 2, 3, 4");
        // 10 chars / 4 = 2 tokens subtracted -> max_tokens should be 98
        assert_eq!(result["max_tokens"], 98);
    }

    #[test]
    fn test_continuation_budget_floors_estimate_at_one_token() {
        let orig = serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10
        });

        let result = build_continuation_request(&orig, "abc", ContinuationMode::Prefix);

        assert_eq!(result["max_tokens"], 9);
    }

    #[test]
    fn test_instructed_continuation() {
        let orig = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Count to 10"}]
        });
        let result = build_continuation_request(&orig, "1, 2, 3, 4", ContinuationMode::Instructed);
        let msgs = result["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "1, 2, 3, 4");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"], INSTRUCTED_USER_PROMPT);
    }
}
