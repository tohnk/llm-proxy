use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AttemptOutcome {
    Terminal(String),
    Retryable(Cause),
    Fatal(ApiError),
}

#[derive(Debug, Clone, Error)]
pub enum Cause {
    #[error("HTTP status {status}: {body}")]
    HttpStatus {
        status: StatusCode,
        retry_after: Option<Duration>,
        body: String,
    },
    #[error("Transport error: {0}")]
    Transport(String),
    #[error("First chunk timeout expired after {0:?}")]
    FirstChunkTimeout(Duration),
    #[error("Idle chunk timeout expired after {0:?}")]
    IdleTimeout(Duration),
    #[error("Upstream stream ended abruptly without [DONE] or finish_reason")]
    StreamEndedUnexpectedly,
    #[error("Finish reason length resumed")]
    LengthFinishResumed,
}

impl Cause {
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Cause::HttpStatus { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
#[error("Upstream API error ({status}): {message}")]
pub struct ApiError {
    #[serde(skip)]
    pub status: StatusCode,
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    /// Kept as the upstream's own JSON value rather than a stringified copy:
    /// OpenAI's `code` may be a string or a number, and re-encoding it changes
    /// the type a client matches on.
    pub code: Option<serde_json::Value>,
    pub param: Option<serde_json::Value>,
    #[serde(skip)]
    pub raw_body: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIErrorWrapper {
    pub error: OpenAIErrorDetails,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIErrorDetails {
    pub message: String,
    #[serde(rename = "type", default = "default_error_type")]
    pub error_type: String,
    pub param: Option<serde_json::Value>,
    pub code: Option<serde_json::Value>,
}

fn default_error_type() -> String {
    "api_error".to_string()
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            error_type: Some("api_error".to_string()),
            code: None,
            param: None,
            raw_body: None,
        }
    }

    pub fn from_status_and_body(status: StatusCode, body: &str) -> Self {
        if let Ok(parsed) = serde_json::from_str::<OpenAIErrorWrapper>(body) {
            Self {
                status,
                message: parsed.error.message,
                error_type: Some(parsed.error.error_type),
                code: parsed.error.code,
                param: parsed.error.param,
                raw_body: Some(body.to_string()),
            }
        } else {
            Self {
                status,
                message: if body.trim().is_empty() {
                    status
                        .canonical_reason()
                        .unwrap_or("Unknown upstream error")
                        .to_string()
                } else {
                    body.to_string()
                },
                error_type: Some(if status.is_client_error() {
                    "invalid_request_error".to_string()
                } else {
                    "api_error".to_string()
                }),
                code: None,
                param: None,
                raw_body: Some(body.to_string()),
            }
        }
    }

    pub fn to_openai_json(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.error_type.clone().unwrap_or_else(|| "api_error".to_string()),
                "param": self.param,
                "code": self.code,
            }
        })
    }
}

/// Phrases that can only describe an oversized prompt or context window.
/// Retrying these is pointless: the request is too big and resuming makes it
/// bigger, so they are fatal whatever status code carried them (§7).
pub fn is_unambiguous_context_length_failure(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("context_length_exceeded")
        || lower.contains("maximum context length")
        || lower.contains("context window")
        || lower.contains("context limit")
        || lower.contains("prompt is too long")
        || lower.contains("string_too_long")
        || lower.contains("reduce your prompt")
        || lower.contains("reduce the length of the messages")
}

/// Phrases that usually mean "your request was too large", but that rate-limit
/// messages borrow as well — token-per-minute quotas are also described as a
/// "token limit" or "too many tokens". Fatal on most statuses, but never
/// sufficient on its own to condemn a 429 (see `classify_http_response`).
pub fn is_ambiguous_size_failure(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("too many tokens") || lower.contains("token limit")
}

/// Content-policy and safety refusals. Retrying re-submits the same content, so
/// these are fatal (§7).
pub fn is_content_policy_failure(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("content_policy")
        || lower.contains("content policy")
        || lower.contains("content filter")
        || lower.contains("content_filter")
        || lower.contains("content safety")
        || lower.contains("safety policy")
        || lower.contains("safety_policy")
        || lower.contains("safety setting")
        || lower.contains("safety violation")
        || lower.contains("safety filter")
        || lower.contains("blocked by safety")
        || lower.contains("prohibited content")
        || lower.contains("violates our usage guidelines")
        || lower.contains("harm_category")
}

pub fn is_context_or_policy_failure(body: &str) -> bool {
    is_unambiguous_context_length_failure(body)
        || is_ambiguous_size_failure(body)
        || is_content_policy_failure(body)
}

pub fn classify_http_response(
    status: StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> AttemptOutcome {
    if status.is_success() {
        // Successful status is not an error; downstream logic handles the stream
        return AttemptOutcome::Terminal("stop".to_string());
    }

    // §7: a context-length or content-policy failure is fatal regardless of the
    // HTTP status that carried it.
    //
    // 429 is deliberately excluded from the general scan. A rate limit is the
    // one error whose own vocabulary overlaps this heuristic — quota messages
    // routinely say "token limit" or "too many tokens" because quotas are
    // metered in tokens per minute — and misreading one as fatal turns a
    // recoverable throttle into a failed response. Only an unambiguous
    // context-length message can condemn a 429.
    let body_is_fatal = if status == StatusCode::TOO_MANY_REQUESTS {
        is_unambiguous_context_length_failure(body)
    } else {
        is_context_or_policy_failure(body)
    };
    if body_is_fatal {
        return AttemptOutcome::Fatal(ApiError::from_status_and_body(status, body));
    }

    let retry_after = parse_retry_after(headers);

    match status.as_u16() {
        // Retryable status codes according to spec (§7)
        408 | 409 | 425 | 429 | 500 | 502 | 503 | 504 => {
            AttemptOutcome::Retryable(Cause::HttpStatus {
                status,
                retry_after,
                body: body.to_string(),
            })
        }
        // Fatal status codes (400, 401, 403, 404, 413, 422, and other client/unretryable codes)
        _ => AttemptOutcome::Fatal(ApiError::from_status_and_body(status, body)),
    }
}

/// Extract the upstream's retry hint.
///
/// `retry-after-ms` is preferred when present: RFC 9110 defines `Retry-After`
/// delay-seconds as `1*DIGIT`, so a provider that wants sub-second precision
/// cannot express it in the standard header and sends this one instead (OpenAI
/// and Azure OpenAI both do). Falls back to `Retry-After` in seconds or
/// HTTP-date form.
pub fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    if let Some(ms) = headers
        .get("retry-after-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        return Some(Duration::from_millis(ms));
    }

    let header_val = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let trimmed = header_val.trim();
    // Check if integer seconds
    if let Ok(secs) = trimmed.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    // Try parsing as HTTP-Date (e.g. IMF-fixdate: "Sun, 06 Nov 1994 08:49:37 GMT")
    // or standard date format
    if let Ok(datetime) = httpdate::parse_http_date(trimmed) {
        if let Ok(duration) = datetime.duration_since(std::time::SystemTime::now()) {
            return Some(duration);
        } else {
            return Some(Duration::from_secs(0));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};

    #[test]
    fn test_classify_success() {
        let headers = HeaderMap::new();
        let outcome = classify_http_response(StatusCode::OK, &headers, "");
        assert!(matches!(outcome, AttemptOutcome::Terminal(fr) if fr == "stop"));
    }

    #[test]
    fn test_classify_retryable_statuses() {
        let headers = HeaderMap::new();
        for code in [408, 409, 425, 429, 500, 502, 503, 504] {
            let status = StatusCode::from_u16(code).unwrap();
            let outcome = classify_http_response(status, &headers, "transient error");
            assert!(
                matches!(outcome, AttemptOutcome::Retryable(_)),
                "Status {code} should be retryable"
            );
        }
    }

    #[test]
    fn test_classify_fatal_statuses() {
        let headers = HeaderMap::new();
        for code in [400, 401, 403, 404, 413, 422] {
            let status = StatusCode::from_u16(code).unwrap();
            let outcome = classify_http_response(status, &headers, "client error");
            assert!(
                matches!(outcome, AttemptOutcome::Fatal(_)),
                "Status {code} should be fatal"
            );
        }
    }

    #[test]
    fn test_classify_context_length_override_fatal() {
        let headers = HeaderMap::new();
        // Even with a 500 or 429 status code, context length exceeded message is Fatal (§7)
        let body = r#"{"error":{"message":"This model's maximum context length is 8192 tokens. However, your messages resulted in 9000 tokens."}}"#;
        let outcome = classify_http_response(StatusCode::INTERNAL_SERVER_ERROR, &headers, body);
        assert!(matches!(outcome, AttemptOutcome::Fatal(_)));

        let outcome_429 = classify_http_response(
            StatusCode::TOO_MANY_REQUESTS,
            &headers,
            r#"{"error":{"message":"prompt is too long"}}"#,
        );
        assert!(matches!(outcome_429, AttemptOutcome::Fatal(_)));
    }

    #[test]
    fn test_classify_content_policy_override_fatal() {
        let headers = HeaderMap::new();
        for status in [
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::BAD_GATEWAY,
        ] {
            let body =
                r#"{"error":{"message":"Request was blocked due to safety_policy violation."}}"#;
            assert!(
                matches!(
                    classify_http_response(status, &headers, body),
                    AttemptOutcome::Fatal(_)
                ),
                "{status} with a safety message should be fatal"
            );
        }
    }

    /// A 429 is a throttle. Quota messages are metered in tokens and therefore
    /// share vocabulary with context-length errors, so the body scan must not be
    /// allowed to turn a recoverable rate limit into a fatal failure.
    #[test]
    fn test_429_rate_limit_bodies_stay_retryable() {
        let headers = HeaderMap::new();
        for body in [
            r#"{"error":{"message":"Rate limit reached: you exceeded your token limit for this model."}}"#,
            r#"{"error":{"message":"Too many tokens per minute. Please slow down."}}"#,
            r#"{"error":{"message":"Quota exceeded for tokens-per-minute."}}"#,
            r#"{"error":{"message":"Resource has been exhausted (e.g. check quota)."}}"#,
            // Even a safety-flavoured 429 is retried rather than condemned: a
            // real content refusal does not arrive as a rate limit.
            r#"{"error":{"message":"blocked by content safety filter"}}"#,
        ] {
            assert!(
                matches!(
                    classify_http_response(StatusCode::TOO_MANY_REQUESTS, &headers, body),
                    AttemptOutcome::Retryable(_)
                ),
                "429 body should stay retryable: {body}"
            );
        }
    }

    /// The one exception: a 429 whose message unmistakably describes an
    /// oversized context is still fatal, because retrying only resends it.
    #[test]
    fn test_429_unambiguous_context_length_is_still_fatal() {
        let headers = HeaderMap::new();
        for body in [
            r#"{"error":{"message":"This model's maximum context length is 8192 tokens."}}"#,
            r#"{"error":{"message":"context_length_exceeded"}}"#,
            r#"{"error":{"message":"prompt is too long"}}"#,
            r#"{"error":{"message":"Please reduce your prompt and try again."}}"#,
        ] {
            assert!(
                matches!(
                    classify_http_response(StatusCode::TOO_MANY_REQUESTS, &headers, body),
                    AttemptOutcome::Fatal(_)
                ),
                "429 body should be fatal: {body}"
            );
        }
    }

    /// The ambiguous phrases remain fatal on every status except 429, where a
    /// rate limit is the more likely reading.
    #[test]
    fn test_ambiguous_size_phrases_stay_fatal_off_429() {
        let headers = HeaderMap::new();
        let body = r#"{"error":{"message":"Request rejected: too many tokens."}}"#;
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::SERVICE_UNAVAILABLE,
        ] {
            assert!(
                matches!(
                    classify_http_response(status, &headers, body),
                    AttemptOutcome::Fatal(_)
                ),
                "{status} with an oversize message should be fatal"
            );
        }
    }

    /// An upstream `code`/`param` must reach the client with its original JSON
    /// type: a string stays an unquoted string, a number stays a number, and an
    /// absent field stays null.
    #[test]
    fn test_error_body_preserves_code_and_param_types() {
        let body = r#"{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key","param":"model"}}"#;
        let json = ApiError::from_status_and_body(StatusCode::UNAUTHORIZED, body).to_openai_json();
        assert_eq!(json["error"]["code"], serde_json::json!("invalid_api_key"));
        assert_eq!(json["error"]["param"], serde_json::json!("model"));
        assert_eq!(json["error"]["message"], "bad key");
        assert_eq!(json["error"]["type"], "invalid_request_error");

        let numeric =
            r#"{"error":{"message":"slow down","type":"api_error","code":429,"param":null}}"#;
        let json =
            ApiError::from_status_and_body(StatusCode::TOO_MANY_REQUESTS, numeric).to_openai_json();
        assert_eq!(
            json["error"]["code"],
            serde_json::json!(429),
            "a numeric code must not be coerced to a string"
        );
        assert!(json["error"]["param"].is_null());

        let bare = ApiError::new(StatusCode::BAD_GATEWAY, "upstream unreachable").to_openai_json();
        assert!(bare["error"]["code"].is_null());
        assert!(bare["error"]["param"].is_null());
    }

    /// `retry-after-ms` carries sub-second precision that `Retry-After` cannot
    /// express, and wins when both are present.
    #[test]
    fn test_parse_retry_after_ms_header() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after-ms", HeaderValue::from_static("1500"));
        assert_eq!(
            parse_retry_after(&headers),
            Some(Duration::from_millis(1500))
        );

        headers.insert(RETRY_AFTER, HeaderValue::from_static("9"));
        assert_eq!(
            parse_retry_after(&headers),
            Some(Duration::from_millis(1500)),
            "retry-after-ms should take precedence"
        );

        // A malformed value falls through to the standard header.
        headers.insert("retry-after-ms", HeaderValue::from_static("soon"));
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(9)));
    }

    #[test]
    fn test_parse_retry_after_seconds() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("15"));
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(15)));
    }

    #[test]
    fn test_parse_retry_after_http_date() {
        let mut headers = HeaderMap::new();
        // Construct a future date (e.g. now + 60s)
        let future = std::time::SystemTime::now() + Duration::from_secs(60);
        let date_str = httpdate::fmt_http_date(future);
        headers.insert(RETRY_AFTER, HeaderValue::from_str(&date_str).unwrap());
        let parsed = parse_retry_after(&headers).expect("Should parse valid future HTTP date");
        assert!(parsed.as_secs() >= 55 && parsed.as_secs() <= 65);
    }
}
