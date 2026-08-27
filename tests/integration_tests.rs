use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;

use llm_proxy::config::{Config, ContinuationMode};
use llm_proxy::create_app;
use llm_proxy::mock::{create_mock_app, default_corpus, MockServerState};
use llm_proxy::proxy::continuation::INSTRUCTED_USER_PROMPT;

struct TestHarness {
    pub _mock_addr: SocketAddr,
    pub proxy_addr: SocketAddr,
    pub mock_state: MockServerState,
    pub client: Client,
}

impl TestHarness {
    pub async fn setup(customize_config: impl FnOnce(&mut Config)) -> Self {
        let mock_state = MockServerState::default();
        let mock_app = create_mock_app(mock_state.clone());
        let mock_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mock_addr = mock_listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(mock_listener, mock_app).await.unwrap();
        });

        let mut config = Config::default();
        config.server.bind = "127.0.0.1:0".to_string();
        config.upstream.base_url = format!("http://{mock_addr}/v1");
        config.upstream.api_key_env = "".to_string();
        config.resume.backoff_initial_ms = 20;
        config.resume.backoff_max_ms = 50;
        config.resume.backoff_jitter = 0.0;
        config.resume.max_attempts = 6;
        config.resume.first_chunk_timeout_secs = 2;
        config.resume.idle_timeout_secs = 2;

        customize_config(&mut config);

        let proxy_app = create_app(Arc::new(config));
        let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy_listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(proxy_listener, proxy_app).await.unwrap();
        });

        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .unwrap();

        Self {
            _mock_addr: mock_addr,
            proxy_addr,
            mock_state,
            client,
        }
    }
}

struct StreamResult {
    pub text: String,
    pub reasoning_text: String,
    pub chunks: Vec<Value>,
    pub done_count: usize,
    pub finish_reasons: Vec<String>,
    pub roles: Vec<String>,
    pub tool_calls: Vec<Value>,
    pub errors: Vec<Value>,
    pub usages: Vec<Value>,
}

async fn collect_stream_response(client: &Client, url: &str, body: Value) -> StreamResult {
    let resp = client
        .post(url)
        .json(&body)
        .send()
        .await
        .expect("Failed to send request to proxy");

    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let mut event_stream = resp.bytes_stream().eventsource();
    let mut text = String::new();
    let mut reasoning_text = String::new();
    let mut chunks = Vec::new();
    let mut done_count = 0;
    let mut finish_reasons = Vec::new();
    let mut roles = Vec::new();
    let mut tool_calls = Vec::new();
    let mut errors = Vec::new();
    let mut usages = Vec::new();

    while let Some(event_res) = event_stream.next().await {
        let event = event_res.expect("SSE event error");
        let data = event.data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            done_count += 1;
            continue;
        }

        if let Ok(json) = serde_json::from_str::<Value>(data) {
            chunks.push(json.clone());

            if let Some(err) = json.get("error") {
                errors.push(err.clone());
            }
            if let Some(usage) = json.get("usage") {
                usages.push(usage.clone());
            }

            if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                if let Some(choice0) = choices.first() {
                    if let Some(fr) = choice0.get("finish_reason").and_then(|f| f.as_str()) {
                        finish_reasons.push(fr.to_string());
                    }
                    if let Some(delta) = choice0.get("delta") {
                        if let Some(role) = delta.get("role").and_then(|r| r.as_str()) {
                            roles.push(role.to_string());
                        }
                        if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
                            text.push_str(c);
                        }
                        if let Some(r) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                            reasoning_text.push_str(r);
                        }
                        if let Some(r) = delta.get("reasoning").and_then(|r| r.as_str()) {
                            reasoning_text.push_str(r);
                        }
                        if let Some(tcs) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                            for tc in tcs {
                                tool_calls.push(tc.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    StreamResult {
        text,
        reasoning_text,
        chunks,
        done_count,
        finish_reasons,
        roles,
        tool_calls,
        errors,
        usages,
    }
}

// 1. Clean passthrough — no failure; output matches corpus, one [DONE].
#[tokio::test]
async fn test_1_clean_passthrough() {
    let harness = TestHarness::setup(|_| {}).await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert_eq!(result.done_count, 1);
    assert!(result.finish_reasons.contains(&"stop".to_string()));
}

#[tokio::test]
async fn test_streaming_usage_is_preserved_on_terminal_chunk() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?include_usage=true",
            config.upstream.base_url
        );
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.usages.len(), 1);
    assert_eq!(result.usages[0]["prompt_tokens"], 10);
    assert_eq!(result.usages[0]["completion_tokens"], 20);
    assert_eq!(result.usages[0]["total_tokens"], 30);
}

// 2. Single mid-stream cut — output matches corpus exactly, no duplicated or missing text.
#[tokio::test]
async fn test_2_single_midstream_cut() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=10&cut_first_n_requests=1",
            config.upstream.base_url
        );
    })
    .await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert_eq!(result.done_count, 1);
}

// 3. Three consecutive cuts — same assertion; verifies repeated resumption.
#[tokio::test]
async fn test_3_three_consecutive_cuts() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=10&cut_first_n_requests=3",
            config.upstream.base_url
        );
        config.resume.max_attempts = 6;
    })
    .await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert_eq!(result.done_count, 1);
}

// 4. Overlap dedup — repeat_overlap=80; output matches corpus with no repetition.
#[tokio::test]
async fn test_4_overlap_dedup() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=15&cut_first_n_requests=1&repeat_overlap=80",
            config.upstream.base_url
        );
        config.resume.dedup_window_chars = 240;
    })
    .await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert_eq!(result.done_count, 1);
}

// 5. Stall — stall_after=5 with idle_timeout_secs=1; proxy recovers and completes.
#[tokio::test]
async fn test_5_stall_recovery() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?stall_after=5&stall_first_n_requests=1",
            config.upstream.base_url
        );
        config.resume.idle_timeout_secs = 1;
    })
    .await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert_eq!(result.done_count, 1);
}

// 6. 429 then success — retried, Retry-After honoured, output complete.
#[tokio::test]
async fn test_6_429_then_success() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=429&fail_first=2&retry_after_secs=1",
            config.upstream.base_url
        );
    })
    .await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        3
    );
}

// 7. 400 fatal — no retry loop, error propagated, exactly one [DONE].
#[tokio::test]
async fn test_7_400_fatal() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=400&fail_first=10",
            config.upstream.base_url
        );
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Bad request"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    // Should stop immediately without retry loop
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    // Verify that error details are propagated as an error chunk
    assert!(!result.errors.is_empty());
    assert_eq!(result.errors[0]["type"], "invalid_request_error");
}

// 8. Give-up — mock always cuts; proxy stops at max_attempts, clean finish chunk and [DONE], no marker in chat.
#[tokio::test]
async fn test_8_give_up() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url =
            format!("{}/chat/completions?cut_after=5", config.upstream.base_url);
        config.resume.max_attempts = 3;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    assert!(result.finish_reasons.contains(&"stop".to_string()));
    // Assert partial text across attempts was preserved without corruption (3 attempts * 5 lines = 15 lines)
    let mut expected_partial = String::new();
    for i in 1..=15 {
        expected_partial.push_str(&format!("Line {i}\n"));
    }
    assert_eq!(result.text, expected_partial);
    // Assert NO proxy synthetic failure marker is in the chat output
    assert!(!result.text.contains("[proxy:"));
    assert!(!result.text.contains("upstream failed"));
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        3
    );
}

// 9. Chunk identity — across a resumed response, every chunk carries the same id, created, and model, and delta.role appears exactly once.
#[tokio::test]
async fn test_9_chunk_identity() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=10&cut_first_n_requests=1",
            config.upstream.base_url
        );
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert!(!result.chunks.is_empty());
    let first_id = result.chunks[0]["id"].as_str().unwrap().to_string();
    let first_created = result.chunks[0]["created"].as_i64().unwrap();
    let first_model = result.chunks[0]["model"].as_str().unwrap().to_string();

    // Upstream sends different ids per attempt ("chatcmpl-mock-chunk-attempt-1" and "chatcmpl-mock-chunk-attempt-2"),
    // so verify that the proxy rewrote ALL chunks to have identical downstream id and metadata.
    assert!(first_id.starts_with("chatcmpl-"));
    for chunk in &result.chunks {
        assert_eq!(chunk["id"].as_str().unwrap(), first_id);
        assert_eq!(chunk["created"].as_i64().unwrap(), first_created);
        assert_eq!(chunk["model"].as_str().unwrap(), first_model);
    }

    assert_eq!(result.roles.len(), 1);
    assert_eq!(result.roles[0], "assistant");
}

// 10. Tool call across a cut — cut mid-arguments; assert the client receives one complete, valid tool call and never a partial one.
#[tokio::test]
async fn test_10_tool_call_across_cut() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?tool_call=true&cut_after=1&cut_first_n_requests=1",
            config.upstream.base_url
        );
        config.tools.buffer_tool_calls = true;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is AAPL stock?"}],
            "tools": [{"type": "function", "function": {"name": "lookup_stock"}}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    assert!(result.finish_reasons.contains(&"tool_calls".to_string()));
    assert!(!result.tool_calls.is_empty());

    let tc = &result.tool_calls[0];
    assert_eq!(tc["function"]["name"], "lookup_stock");
    let args_str = tc["function"]["arguments"].as_str().unwrap();
    let parsed_args: Value = serde_json::from_str(args_str).expect("Valid JSON tool args");
    assert_eq!(parsed_args["ticker"], "AAPL");
}

// 11. Client disconnect — drop the client mid-stream; assert the mock observes its connection closing.
#[tokio::test]
async fn test_11_client_disconnect() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!("{}/chat/completions", config.upstream.base_url);
    })
    .await;

    let resp = harness
        .client
        .post(format!("http://{}/v1/chat/completions", harness.proxy_addr))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    let mut stream = resp.bytes_stream().eventsource();
    // Read just 2 events then drop the stream
    let _ = stream.next().await;
    let _ = stream.next().await;
    drop(stream);

    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        harness
            .mock_state
            .client_disconnect_counter
            .load(std::sync::atomic::Ordering::SeqCst)
            > 0
    );
}

// 12. Non-streaming retry — stream: false with fail_first=2; correct body returned.
#[tokio::test]
async fn test_12_non_streaming_retry() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?fail_first=2&status=503",
            config.upstream.base_url
        );
    })
    .await;

    let resp = harness
        .client
        .post(format!("http://{}/v1/chat/completions", harness.proxy_addr))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let json: Value = resp.json().await.unwrap();
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        3
    );
}

// 13. First chunk timeout recovery — slow first chunk causes first_chunk_timeout, proxy retries and completes.
#[tokio::test]
async fn test_13_first_chunk_timeout_recovery() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?slow_first_chunk=2&cut_first_n_requests=1",
            config.upstream.base_url
        );
        config.resume.first_chunk_timeout_secs = 1;
    })
    .await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

// 14. ContinuationMode::Instructed with upstream restart — upstream ignores assistant turn and restarts, proxy dedup recovers.
#[tokio::test]
async fn test_14_instructed_mode_with_restart() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=10&cut_first_n_requests=1&restart=true",
            config.upstream.base_url
        );
        config.resume.mode = ContinuationMode::Instructed;
        config.resume.dedup_window_chars = 240;
    })
    .await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert_eq!(result.done_count, 1);
}

// 15. resume_on_length_finish = true — upstream reports finish_reason "length", proxy resumes and completes.
#[tokio::test]
async fn test_15_resume_on_length_finish() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=10&cut_first_n_requests=1&finish_reason=length",
            config.upstream.base_url
        );
        config.resume.resume_on_length_finish = true;
    })
    .await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

// 16. Reasoning content forwarding — reasoning is emitted downstream but not included in continuation messages.
#[tokio::test]
async fn test_16_reasoning_content_forwarding() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?reasoning=true&cut_after=10&cut_first_n_requests=1",
            config.upstream.base_url
        );
    })
    .await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert!(!result.reasoning_text.is_empty());
    assert!(result.reasoning_text.contains("Thinking step 1"));
    let last_body = harness
        .mock_state
        .last_received_body
        .lock()
        .await
        .clone()
        .unwrap();
    assert!(!last_body.to_string().contains("Thinking step"));
    assert_eq!(result.done_count, 1);
}

// 17. A tool-call stream cut before any text is emitted.
//
// §9 suppresses resumption for tool requests only "if a failure occurs after
// text/content has begun emitting". The mock's tool fixture emits no text, so
// that condition is not met and resumption correctly proceeds. The case where
// text *has* been emitted needs a raw upstream — the mock always serves its tool
// fixture when `tools` is present — and lives in
// `raw_upstream_tests::test_disable_resume_for_tool_responses_after_text`.
#[tokio::test]
async fn test_17_tool_stream_cut_before_any_text_still_resumes() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=1&cut_first_n_requests=1",
            config.upstream.base_url
        );
        config.tools.disable_resume_for_tool_responses = true;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Tool request"}],
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2,
        "no text was emitted, so §9 does not suppress resumption"
    );
    assert_eq!(
        result.tool_calls.len(),
        1,
        "only the retry's assembled call should be emitted"
    );
    assert_eq!(result.finish_reasons, vec!["tool_calls".to_string()]);
    assert_eq!(result.done_count, 1);
}

// 18. resume.enabled = false — cuts mid-stream, proxy immediately terminates with stop and [DONE].
#[tokio::test]
async fn test_18_resume_disabled() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url =
            format!("{}/chat/completions?cut_after=10", config.upstream.base_url);
        config.resume.enabled = false;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
}

// 19. Retry-After delay is measured and honored.
#[tokio::test]
async fn test_19_retry_after_measured_delay() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=429&fail_first=1&retry_after_secs=1",
            config.upstream.base_url
        );
        config.resume.backoff_max_ms = 5000;
        config.resume.backoff_initial_ms = 10;
    })
    .await;

    let start = Instant::now();
    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    let elapsed = start.elapsed();
    assert_eq!(result.done_count, 1);
    assert!(
        elapsed >= Duration::from_millis(900),
        "Expected elapsed >= 900ms, was {:?}",
        elapsed
    );
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

#[tokio::test]
async fn test_health_endpoint() {
    let harness = TestHarness::setup(|_| {}).await;
    let resp = harness
        .client
        .get(format!("http://{}/health", harness.proxy_addr))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let json: Value = resp.json().await.unwrap();
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn test_models_endpoint() {
    let harness = TestHarness::setup(|_| {}).await;
    let resp = harness
        .client
        .get(format!("http://{}/v1/models", harness.proxy_addr))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let json: Value = resp.json().await.unwrap();
    assert!(json.get("data").is_some());
}

#[tokio::test]
async fn test_unknown_endpoint_404() {
    let harness = TestHarness::setup(|_| {}).await;
    let resp = harness
        .client
        .get(format!("http://{}/v1/unknown_route", harness.proxy_addr))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    let json: Value = resp.json().await.unwrap();
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

// 20. Wall-clock give-up (max_wall_clock_secs) — proxy stops cleanly when wall clock ceiling is exceeded.
#[tokio::test]
async fn test_20_wall_clock_give_up() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url =
            format!("{}/chat/completions?cut_after=5", config.upstream.base_url);
        config.resume.max_wall_clock_secs = 1;
        config.resume.backoff_initial_ms = 600;
        config.resume.max_attempts = 100;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    assert!(result.finish_reasons.contains(&"stop".to_string()));
    let reqs = harness
        .mock_state
        .request_counter
        .load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        reqs < 10,
        "Expected wall clock cutoff before 10 attempts, ran {reqs}"
    );
}

// 21. Keepalive comments — SSE comment ': retrying\n\n' emitted during backoff > 10s.
#[tokio::test]
async fn test_21_keepalive_comments() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?fail_first=1&status=429&retry_after_secs=11",
            config.upstream.base_url
        );
        config.resume.backoff_max_ms = 15000;
    })
    .await;

    let resp = harness
        .client
        .post(format!("http://{}/v1/chat/completions", harness.proxy_addr))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let mut byte_stream = resp.bytes_stream();
    let mut seen_retrying_comment = false;
    let mut raw_text = String::new();

    while let Some(chunk_res) = byte_stream.next().await {
        let chunk = chunk_res.unwrap();
        let text = String::from_utf8_lossy(&chunk);
        raw_text.push_str(&text);
        if text.contains(": retrying") {
            seen_retrying_comment = true;
        }
    }

    assert!(
        seen_retrying_comment,
        "Expected raw SSE comment ': retrying' in stream. Raw text: {}",
        raw_text
    );
    assert!(raw_text.contains("data: [DONE]"));
}

// 22. Retry-After as HTTP-date and clamping to backoff_max_ms.
#[tokio::test]
async fn test_22_retry_after_http_date_and_clamping() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?fail_first=1&status=429&retry_after_http_date_secs=10",
            config.upstream.base_url
        );
        config.resume.backoff_max_ms = 1000;
    })
    .await;

    let start = Instant::now();
    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    let elapsed = start.elapsed();
    assert_eq!(result.done_count, 1);
    assert!(
        elapsed >= Duration::from_millis(900) && elapsed < Duration::from_secs(5),
        "Elapsed was {:?}",
        elapsed
    );
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

// 23. Context-length error message sniffing causes immediate Fatal outcome without retrying.
#[tokio::test]
async fn test_23_context_length_fatal_sniffing() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!("{}/chat/completions?status=500&fail_first=5&error_message=maximum+context+length+exceeded", config.upstream.base_url);
    }).await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Huge prompt"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    assert!(!result.errors.is_empty());
}

// 24. buffer_tool_calls = false passthrough mode.
#[tokio::test]
async fn test_24_buffer_tool_calls_false_passthrough() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?tool_call=true",
            config.upstream.base_url
        );
        config.tools.buffer_tool_calls = false;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Lookup AAPL"}],
            "tools": [{"type": "function", "function": {"name": "lookup_stock"}}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    assert_eq!(result.roles.len(), 1);
    assert_eq!(result.roles[0], "assistant");
    assert!(result.tool_calls.len() >= 2);
}

// 25. Multiple parallel tool calls assembled correctly.
#[tokio::test]
async fn test_25_multi_tool_call_assembly() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?tool_call=true&multi_tool_call=true",
            config.upstream.base_url
        );
        config.tools.buffer_tool_calls = true;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Stock and weather"}],
            "tools": [
                {"type": "function", "function": {"name": "lookup_stock"}},
                {"type": "function", "function": {"name": "lookup_weather"}}
            ],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    assert_eq!(result.tool_calls.len(), 2);
    assert_eq!(result.tool_calls[0]["function"]["name"], "lookup_stock");
    assert_eq!(result.tool_calls[1]["function"]["name"], "lookup_weather");
}

// 26. Authorization header contract — client header discarded, proxy key injected.
#[tokio::test]
async fn test_26_auth_header_contract() {
    let harness = TestHarness::setup(|config| {
        config.upstream.api_key = Some("proxy_secret_upstream_key".to_string());
    })
    .await;

    let resp = harness
        .client
        .post(format!("http://{}/v1/chat/completions", harness.proxy_addr))
        .header("Authorization", "Bearer client_dummy_key_to_discard")
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let recorded_auth = harness.mock_state.last_auth_header.lock().await.clone();
    assert_eq!(
        recorded_auth,
        Some("Bearer proxy_secret_upstream_key".to_string())
    );
}

// 27. Custom request parameters survive round trip through continuation.
#[tokio::test]
async fn test_27_custom_parameters_preserved() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=5&cut_first_n_requests=1",
            config.upstream.base_url
        );
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true,
            "temperature": 0.42,
            "custom_passthrough_param": "special_value"
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    let last_body = harness
        .mock_state
        .last_received_body
        .lock()
        .await
        .clone()
        .unwrap();
    assert_eq!(last_body["temperature"], 0.42);
    assert_eq!(last_body["custom_passthrough_param"], "special_value");
}

// 28. Non-streaming give-up and fatal error propagation.
#[tokio::test]
async fn test_28_non_streaming_give_up_and_fatal() {
    // 1. Fatal 400 stops immediately
    let harness_fatal = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=400&fail_first=5",
            config.upstream.base_url
        );
    })
    .await;

    let resp_fatal = harness_fatal
        .client
        .post(format!(
            "http://{}/v1/chat/completions",
            harness_fatal.proxy_addr
        ))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Bad request"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp_fatal.status(), reqwest::StatusCode::BAD_REQUEST);
    let fatal_body: Value = resp_fatal.json().await.unwrap();
    assert_eq!(fatal_body["error"]["code"], 400);
    assert_eq!(
        harness_fatal
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );

    // 2. Retryable 500 retries until max_attempts give-up
    let harness_giveup = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=500&fail_first=10",
            config.upstream.base_url
        );
        config.resume.max_attempts = 3;
    })
    .await;

    let resp_giveup = harness_giveup
        .client
        .post(format!(
            "http://{}/v1/chat/completions",
            harness_giveup.proxy_addr
        ))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Server error"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp_giveup.status(),
        reqwest::StatusCode::INTERNAL_SERVER_ERROR
    );
    let giveup_body: Value = resp_giveup.json().await.unwrap();
    assert_eq!(giveup_body["error"]["code"], 500);
    assert_eq!(
        harness_giveup
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        3
    );
}

// 29. Models endpoint retry on 5xx status.
#[tokio::test]
async fn test_29_models_endpoint_retry_on_5xx() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/models?status=503&fail_first=1",
            config.upstream.base_url
        );
    })
    .await;

    let resp = harness
        .client
        .get(format!("http://{}/v1/models", harness.proxy_addr))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let json: Value = resp.json().await.unwrap();
    assert!(json.get("data").is_some());
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

#[tokio::test]
async fn test_models_endpoint_preserves_error_body() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/models?status=400&fail_first=1",
            config.upstream.base_url
        );
    })
    .await;

    let resp = harness
        .client
        .get(format!("http://{}/v1/models", harness.proxy_addr))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], 400);
    assert_eq!(
        body["error"]["message"],
        "Mock models upstream simulated error 400"
    );
}

// 30. Retryable HTTP status codes 502/504 retry and complete.
#[tokio::test]
async fn test_30_retryable_502_status() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=502&fail_first=2",
            config.upstream.base_url
        );
    })
    .await;
    let expected = default_corpus();

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, expected);
    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        3
    );
}

#[tokio::test]
async fn test_first_chunk_timeout_includes_http_dispatch() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?slow_response=2&slow_response_first_n_requests=1",
            config.upstream.base_url
        );
        config.resume.first_chunk_timeout_secs = 1;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

#[tokio::test]
async fn test_resume_disabled_still_retries_pre_stream_timeout() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?slow_response=2&slow_response_first_n_requests=1",
            config.upstream.base_url
        );
        config.resume.first_chunk_timeout_secs = 1;
        config.resume.enabled = false;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

#[tokio::test]
async fn test_resume_disabled_still_retries_initial_http_error() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=503&fail_first=1",
            config.upstream.base_url
        );
        config.resume.enabled = false;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

#[tokio::test]
async fn test_reasoning_field_is_preserved() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?reasoning=true&reasoning_field=reasoning",
            config.upstream.base_url
        );
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert!(result.reasoning_text.contains("Thinking step 1"));
    assert!(result
        .chunks
        .iter()
        .any(|chunk| { chunk["choices"][0]["delta"].get("reasoning").is_some() }));
    assert!(!result.chunks.iter().any(|chunk| {
        chunk["choices"][0]["delta"]
            .get("reasoning_content")
            .is_some()
    }));
}

#[tokio::test]
async fn test_role_only_chunk_is_forwarded_once() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?role_only=true",
            config.upstream.base_url
        );
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.roles, vec!["assistant"]);
    assert_eq!(result.chunks[0]["choices"][0]["delta"]["role"], "assistant");
}

#[tokio::test]
async fn test_resume_disabled_stops_after_role_chunk_failure() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?role_only=true&cut_after=0",
            config.upstream.base_url
        );
        config.resume.enabled = false;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
}

#[tokio::test]
async fn test_nonstreaming_wall_clock_cancels_dispatch() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?slow_response=2&slow_response_first_n_requests=1",
            config.upstream.base_url
        );
        config.resume.max_wall_clock_secs = 1;
    })
    .await;

    let resp = harness
        .client
        .post(format!("http://{}/v1/chat/completions", harness.proxy_addr))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_GATEWAY);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
}

// ---------------------------------------------------------------------------
// Retryable / fatal HTTP status matrices (§7)
//
// The classifier is unit-tested in `src/error.rs`, but until now only 400, 429,
// 502 and 503 were ever driven through the whole proxy. These two tests walk the
// complete lists from §7 end to end.
// ---------------------------------------------------------------------------

// Every retryable status in §7 backs off, resumes, and produces the full body.
#[tokio::test]
async fn test_all_retryable_statuses_recover_end_to_end() {
    for code in [408u16, 409, 425, 429, 500, 502, 503, 504] {
        let harness = TestHarness::setup(|config| {
            config.upstream.base_url = format!(
                "{}/chat/completions?status={code}&fail_first=1",
                config.upstream.base_url
            );
        })
        .await;

        let result = collect_stream_response(
            &harness.client,
            &format!("http://{}/v1/chat/completions", harness.proxy_addr),
            serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Count lines"}],
                "stream": true
            }),
        )
        .await;

        assert_eq!(
            result.text,
            default_corpus(),
            "status {code} should have resumed to a complete body"
        );
        assert_eq!(result.done_count, 1, "status {code} emitted != 1 [DONE]");
        assert!(
            result.errors.is_empty(),
            "status {code} leaked an error chunk downstream: {:?}",
            result.errors
        );
        assert_eq!(
            result.finish_reasons,
            vec!["stop".to_string()],
            "status {code} finish reasons"
        );
        assert_eq!(
            harness
                .mock_state
                .request_counter
                .load(std::sync::atomic::Ordering::SeqCst),
            2,
            "status {code} should have taken exactly one retry"
        );
    }
}

// Every fatal status in §7 propagates immediately without a second attempt.
#[tokio::test]
async fn test_all_fatal_statuses_abort_without_retry() {
    for code in [400u16, 401, 403, 404, 413, 422] {
        let harness = TestHarness::setup(|config| {
            config.upstream.base_url = format!(
                "{}/chat/completions?status={code}&fail_first=10",
                config.upstream.base_url
            );
        })
        .await;

        let result = collect_stream_response(
            &harness.client,
            &format!("http://{}/v1/chat/completions", harness.proxy_addr),
            serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Count lines"}],
                "stream": true
            }),
        )
        .await;

        assert_eq!(result.done_count, 1, "status {code} emitted != 1 [DONE]");
        assert_eq!(
            result.errors.len(),
            1,
            "status {code} should surface exactly one error chunk"
        );
        assert_eq!(
            result.errors[0]["message"],
            format!("Mock upstream simulated error {code}"),
            "status {code} should preserve the upstream message"
        );
        assert!(
            result.text.is_empty(),
            "status {code} should not emit content"
        );
        assert_eq!(
            harness
                .mock_state
                .request_counter
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "status {code} must not be retried"
        );
    }
}

// The same matrix on the non-streaming path (§10): retryable statuses retry,
// fatal statuses return the upstream status and body verbatim.
#[tokio::test]
async fn test_non_streaming_status_matrix() {
    for code in [408u16, 409, 425, 429, 500, 502, 503, 504] {
        let harness = TestHarness::setup(|config| {
            config.upstream.base_url = format!(
                "{}/chat/completions?status={code}&fail_first=1",
                config.upstream.base_url
            );
        })
        .await;

        let resp = harness
            .client
            .post(format!("http://{}/v1/chat/completions", harness.proxy_addr))
            .json(&serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": false
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            reqwest::StatusCode::OK,
            "status {code} should have been retried to success"
        );
        let body: Value = resp.json().await.unwrap();
        assert_eq!(
            body["choices"][0]["message"]["content"],
            "Line 1\nLine 2\nLine 3\n"
        );
    }

    for code in [400u16, 401, 403, 404, 413, 422] {
        let harness = TestHarness::setup(|config| {
            config.upstream.base_url = format!(
                "{}/chat/completions?status={code}&fail_first=10",
                config.upstream.base_url
            );
        })
        .await;

        let resp = harness
            .client
            .post(format!("http://{}/v1/chat/completions", harness.proxy_addr))
            .json(&serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": false
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(
            resp.status().as_u16(),
            code,
            "status {code} should be passed through verbatim"
        );
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["error"]["code"], code);
        assert_eq!(
            harness
                .mock_state
                .request_counter
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "status {code} must not be retried"
        );
    }
}

// ---------------------------------------------------------------------------
// Downstream response headers and CORS (§4, §6.6)
// ---------------------------------------------------------------------------

// §6.6: the exact SSE header set Jan.ai relies on to avoid buffering.
#[tokio::test]
async fn test_streaming_response_headers() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!("{}/chat/completions", config.upstream.base_url);
    })
    .await;

    let resp = harness
        .client
        .post(format!("http://{}/v1/chat/completions", harness.proxy_addr))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let headers = resp.headers();
    assert_eq!(headers["content-type"], "text/event-stream");
    assert_eq!(headers["cache-control"], "no-cache");
    assert_eq!(headers["x-accel-buffering"], "no");
    // hyper manages `Connection` on HTTP/1.1 keep-alive responses, so assert the
    // stream is genuinely chunked rather than asserting the hop-by-hop header.
    assert!(
        headers.get("content-length").is_none(),
        "a streaming response must not be length-delimited"
    );
}

// §4: no CORS headers unless origins are configured. Jan.ai and other desktop
// clients send no `Origin`, so permissive headers would only let any page a
// browser loads spend the configured account's quota through this listener.
#[tokio::test]
async fn test_no_cors_headers_by_default() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!("{}/chat/completions", config.upstream.base_url);
    })
    .await;

    let simple = harness
        .client
        .get(format!("http://{}/health", harness.proxy_addr))
        .header("Origin", "https://evil.example")
        .send()
        .await
        .unwrap();

    assert_eq!(simple.status(), reqwest::StatusCode::OK);
    assert!(
        !simple.headers().contains_key("access-control-allow-origin"),
        "an unconfigured proxy must not grant browser access"
    );

    // A native client with no Origin is unaffected.
    let native = harness
        .client
        .get(format!("http://{}/health", harness.proxy_addr))
        .send()
        .await
        .unwrap();
    assert_eq!(native.status(), reqwest::StatusCode::OK);
}

// ...and an explicitly allowed origin does get them.
#[tokio::test]
async fn test_configured_origin_is_allowed() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!("{}/chat/completions", config.upstream.base_url);
        config.server.allowed_origins = vec!["http://localhost:1420".to_string()];
    })
    .await;

    let preflight = harness
        .client
        .request(
            reqwest::Method::OPTIONS,
            format!("http://{}/v1/chat/completions", harness.proxy_addr),
        )
        .header("Origin", "http://localhost:1420")
        .header("Access-Control-Request-Method", "POST")
        .header("Access-Control-Request-Headers", "content-type")
        .send()
        .await
        .unwrap();

    assert!(preflight.status().is_success());
    assert_eq!(
        preflight.headers()["access-control-allow-origin"],
        "http://localhost:1420"
    );

    // An origin outside the list is not granted access.
    let other = harness
        .client
        .get(format!("http://{}/health", harness.proxy_addr))
        .header("Origin", "https://evil.example")
        .send()
        .await
        .unwrap();
    assert!(!other.headers().contains_key("access-control-allow-origin"));
}

// §5: unknown routes return 404 with an OpenAI-shaped error body, and CORS still
// applies so the browser client sees the 404 rather than a network error.
#[tokio::test]
async fn test_unknown_endpoint_error_shape() {
    let harness = TestHarness::setup(|_| {}).await;

    let resp = harness
        .client
        .get(format!("http://{}/v1/embeddings", harness.proxy_addr))
        .header("Origin", "http://localhost:1420")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
    assert!(
        !resp.headers().contains_key("access-control-allow-origin"),
        "no CORS grant should be issued for an unconfigured origin"
    );

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"].is_string());
    assert!(body["error"].get("param").is_some());
    assert!(body["error"].get("code").is_some());
}

// ---------------------------------------------------------------------------
// §7 body sniffing vs. rate limits
// ---------------------------------------------------------------------------

// A 429 phrased in token terms is a throttle, not an oversized request. The
// proxy must back off and resume rather than condemn the response.
#[tokio::test]
async fn test_429_token_quota_message_is_retried_not_fatal() {
    for message in [
        "Rate+limit+reached:+token+limit+exceeded+for+this+model",
        "Too+many+tokens+per+minute,+please+slow+down",
        "Quota+exceeded+for+tokens-per-minute",
    ] {
        let harness = TestHarness::setup(|config| {
            config.upstream.base_url = format!(
                "{}/chat/completions?status=429&fail_first=1&error_message={message}",
                config.upstream.base_url
            );
        })
        .await;

        let result = collect_stream_response(
            &harness.client,
            &format!("http://{}/v1/chat/completions", harness.proxy_addr),
            serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Count lines"}],
                "stream": true
            }),
        )
        .await;

        assert_eq!(result.text, default_corpus(), "429 body {message:?}");
        assert!(
            result.errors.is_empty(),
            "429 body {message:?} was treated as fatal: {:?}",
            result.errors
        );
        assert_eq!(result.done_count, 1);
        assert_eq!(
            harness
                .mock_state
                .request_counter
                .load(std::sync::atomic::Ordering::SeqCst),
            2,
            "429 body {message:?} should have been retried exactly once"
        );
    }
}

// The exception: a 429 that unmistakably reports an oversized context is still
// fatal, because every retry only resends the same too-large prompt.
#[tokio::test]
async fn test_429_context_length_message_is_still_fatal() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=429&fail_first=5&error_message=maximum+context+length+is+8192+tokens",
            config.upstream.base_url
        );
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Huge prompt"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.done_count, 1);
    assert_eq!(result.errors.len(), 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "an oversized context must not be retried"
    );
}

// Rate limits without a Retry-After header escalate exponentially rather than
// hammering the upstream at a fixed interval (§7).
#[tokio::test]
async fn test_rate_limit_backoff_escalates_without_retry_after() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=429&fail_first=3",
            config.upstream.base_url
        );
        config.resume.backoff_initial_ms = 200;
        config.resume.backoff_multiplier = 2.0;
        config.resume.backoff_max_ms = 10_000;
        config.resume.backoff_jitter = 0.0;
    })
    .await;

    let started = Instant::now();
    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;
    let elapsed = started.elapsed();

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        4
    );
    // 200 + 400 + 800 = 1400ms of backoff; a fixed 200ms interval would be 600ms.
    assert!(
        elapsed >= Duration::from_millis(1_300),
        "backoff did not escalate across rate-limited attempts: {elapsed:?}"
    );
}

// An upstream that answers `Retry-After: 1` to every attempt has not recovered.
// Honouring it literally would spend the whole attempt budget in a few seconds;
// the hint must act as a floor while the exponential curve keeps escalating.
#[tokio::test]
async fn test_repeated_retry_after_still_escalates() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=429&fail_first=3&retry_after_secs=1",
            config.upstream.base_url
        );
        config.resume.backoff_initial_ms = 1_000;
        config.resume.backoff_multiplier = 2.0;
        config.resume.backoff_max_ms = 30_000;
        config.resume.backoff_jitter = 0.0;
    })
    .await;

    let started = Instant::now();
    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;
    let elapsed = started.elapsed();

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        4
    );
    // max(1s,1s) + max(1s,2s) + max(1s,4s) = 7s of backoff. Obeying the hint
    // verbatim would have retried three times inside 3s.
    assert!(
        elapsed >= Duration::from_millis(6_500),
        "repeated Retry-After did not escalate: {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// Error-body shape, body limits, and retry hints
// ---------------------------------------------------------------------------

// A fatal upstream error must reach the client with its `code` and `param`
// carrying their original JSON types — a string stays an unquoted string, a
// number stays a number — not a re-encoded string of the JSON representation.
#[tokio::test]
async fn test_fatal_error_chunk_preserves_code_and_param_types() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=400&fail_first=5",
            config.upstream.base_url
        );
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Bad request"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(result.errors.len(), 1);
    let err = &result.errors[0];
    // The mock sends `"code": 400` as a JSON number.
    assert_eq!(
        err["code"],
        serde_json::json!(400),
        "numeric code was re-typed: {err}"
    );
    assert!(err["param"].is_null(), "null param was re-typed: {err}");
    assert!(err["message"].is_string());
    assert_eq!(err["type"], "invalid_request_error");
}

// A conversation with a large prompt must not be rejected by the downstream
// body limit: axum's 2 MiB default truncates realistic long chats.
#[tokio::test]
async fn test_large_request_body_is_accepted() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!("{}/chat/completions", config.upstream.base_url);
    })
    .await;

    let big_prompt = "x".repeat(4 * 1024 * 1024);
    let resp = harness
        .client
        .post(format!("http://{}/v1/chat/completions", harness.proxy_addr))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": big_prompt}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::OK,
        "a 4 MiB prompt should be forwarded, not rejected"
    );
}

// ...but the limit is still enforced, and is configurable.
#[tokio::test]
async fn test_request_body_limit_is_configurable() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!("{}/chat/completions", config.upstream.base_url);
        config.server.max_body_bytes = 64 * 1024;
    })
    .await;

    let resp = harness
        .client
        .post(format!("http://{}/v1/chat/completions", harness.proxy_addr))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "y".repeat(256 * 1024)}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);
}

// ---------------------------------------------------------------------------
// Continuation request shape (§6.4)
//
// Pins the exact `messages` array each mode sends upstream. Google's
// OpenAI-compatibility endpoint rejects any request whose messages end with an
// assistant turn — "Requests ending with a model turn are not supported" — so
// the trailing role is a compatibility contract, not an internal detail.
// ---------------------------------------------------------------------------

async fn continuation_body_after_a_cut(mode: ContinuationMode) -> Value {
    let harness = TestHarness::setup(move |config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=5&cut_first_n_requests=1",
            config.upstream.base_url
        );
        config.resume.mode = mode;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;
    assert_eq!(result.text, default_corpus(), "resumption should complete");

    let body = harness
        .mock_state
        .last_received_body
        .lock()
        .await
        .clone()
        .expect("upstream should have received a continuation request");
    body
}

#[tokio::test]
async fn test_prefix_continuation_ends_with_the_assistant_turn() {
    let body = continuation_body_after_a_cut(ContinuationMode::Prefix).await;
    let messages = body["messages"].as_array().unwrap();

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    assert!(messages[1]["content"]
        .as_str()
        .unwrap()
        .starts_with("Line 1\n"));
}

#[tokio::test]
async fn test_instructed_continuation_never_ends_with_an_assistant_turn() {
    let body = continuation_body_after_a_cut(ContinuationMode::Instructed).await;
    let messages = body["messages"].as_array().unwrap();

    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    assert!(messages[1]["content"]
        .as_str()
        .unwrap()
        .starts_with("Line 1\n"));

    // The load-bearing assertion: upstreams that refuse a trailing model turn
    // must see a user turn last.
    assert_eq!(
        messages[2]["role"], "user",
        "instructed mode must not end on an assistant turn"
    );
    assert_eq!(messages[2]["content"], INSTRUCTED_USER_PROMPT);
}

// The shipped default must work against the shipped default upstream.
#[tokio::test]
async fn test_default_continuation_mode_is_compatible_with_default_upstream() {
    assert_eq!(Config::default().resume.mode, ContinuationMode::Instructed);

    let body = continuation_body_after_a_cut(Config::default().resume.mode).await;
    let messages = body["messages"].as_array().unwrap();
    assert_ne!(
        messages.last().unwrap()["role"],
        "assistant",
        "the default mode builds a request Gemini would reject with a 400"
    );
}

// ---------------------------------------------------------------------------
// Give-up behaviour, failed generations, and the congestion deadline (§7, §7.3)
// ---------------------------------------------------------------------------

// Nothing was ever emitted, so there is no partial answer for a clean `stop` to
// protect — and a silent empty response tells the user nothing. The real
// upstream error must reach the client instead.
#[tokio::test]
async fn test_give_up_with_no_content_surfaces_the_upstream_error() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=503&fail_first=100&error_message=The+model+is+overloaded",
            config.upstream.base_url
        );
        config.resume.max_attempts = 3;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert!(result.text.is_empty());
    assert_eq!(
        result.errors.len(),
        1,
        "the client must be told why it failed"
    );
    assert_eq!(result.errors[0]["message"], "The model is overloaded");
    assert!(
        result.finish_reasons.is_empty(),
        "an empty response must not be reported as a completed one"
    );
    assert_eq!(result.done_count, 1);
}

// ...but once text has gone out, the partial answer is protected: a clean stop,
// with no error chunk appended to corrupt what the user already sees.
#[tokio::test]
async fn test_give_up_with_partial_content_still_ends_cleanly() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url =
            format!("{}/chat/completions?cut_after=5", config.upstream.base_url);
        config.resume.max_attempts = 2;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert!(
        !result.text.is_empty(),
        "partial text should have been kept"
    );
    assert!(
        result.errors.is_empty(),
        "a partial answer must not be corrupted"
    );
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert_eq!(result.done_count, 1);
}

// A malformed function call is a failed generation, not a finished response, so
// it is retried — and the non-OpenAI reason never reaches the client.
#[tokio::test]
async fn test_malformed_function_call_is_retried_and_normalised() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?finish_reason=function_call_filter:+MALFORMED_FUNCTION_CALL&finish_reason_first_n_requests=1",
            config.upstream.base_url
        );
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2,
        "a failed generation should cost one retry"
    );
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert_eq!(result.done_count, 1);
}

// Every finish reason leaving the proxy is one an OpenAI client understands.
#[tokio::test]
async fn test_vendor_finish_reasons_are_normalised_downstream() {
    for (upstream, expected) in [("SAFETY", "content_filter"), ("MAX_TOKENS", "length")] {
        let harness = TestHarness::setup(|config| {
            config.upstream.base_url = format!(
                "{}/chat/completions?finish_reason={upstream}&finish_reason_first_n_requests=1",
                config.upstream.base_url
            );
        })
        .await;

        let result = collect_stream_response(
            &harness.client,
            &format!("http://{}/v1/chat/completions", harness.proxy_addr),
            serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Count lines"}],
                "stream": true
            }),
        )
        .await;

        assert_eq!(
            result.finish_reasons,
            vec![expected.to_string()],
            "upstream {upstream} should normalise to {expected}"
        );
        assert_eq!(result.done_count, 1);
    }
}

// §7.3: while the upstream is only reporting congestion, the proxy keeps trying
// past the normal ceiling; any other kind of failure drops it back.
#[tokio::test]
async fn test_congestion_extends_the_deadline_but_other_failures_do_not() {
    let overloaded = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=503&fail_first=1000",
            config.upstream.base_url
        );
        config.resume.max_attempts = 10_000;
        config.resume.max_wall_clock_secs = 1;
        config.resume.overloaded_max_wall_clock_secs = 4;
    })
    .await;

    let started = Instant::now();
    let result = collect_stream_response(
        &overloaded.client,
        &format!("http://{}/v1/chat/completions", overloaded.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;
    let congested_elapsed = started.elapsed();

    assert_eq!(result.done_count, 1);
    assert!(
        congested_elapsed >= Duration::from_secs(3),
        "congestion should have extended past the 1s ceiling, took {congested_elapsed:?}"
    );

    // Control: the stream dies mid-transfer instead, which is a fault rather
    // than congestion, so the normal ceiling applies.
    let broken = TestHarness::setup(|config| {
        config.upstream.base_url =
            format!("{}/chat/completions?cut_after=0", config.upstream.base_url);
        config.resume.max_attempts = 10_000;
        config.resume.max_wall_clock_secs = 1;
        config.resume.overloaded_max_wall_clock_secs = 4;
    })
    .await;

    let started = Instant::now();
    let result = collect_stream_response(
        &broken.client,
        &format!("http://{}/v1/chat/completions", broken.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;
    let broken_elapsed = started.elapsed();

    assert_eq!(result.done_count, 1);
    assert!(
        broken_elapsed < Duration::from_secs(3),
        "a non-congestion failure must not get the extended deadline, took {broken_elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// No-progress cutoff (§7.5)
// ---------------------------------------------------------------------------

// A cause that repeats identically while emitting nothing is not transient.
// Retrying it only turns a fast failure into a slow one with the same outcome,
// so the proxy stops after `max_no_progress_attempts`.
#[tokio::test]
async fn test_repeated_failure_with_no_progress_stops_early() {
    let harness = TestHarness::setup(|config| {
        // cut_after=0 with a finish reason emits the terminal chunk immediately:
        // a failed generation carrying no content, exactly as observed from
        // Gemini's MALFORMED_FUNCTION_CALL.
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=0&finish_reason=function_call_filter:+MALFORMED_FUNCTION_CALL",
            config.upstream.base_url
        );
        config.resume.max_attempts = 50;
        config.resume.max_no_progress_attempts = 3;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        3,
        "should stop at the no-progress limit, not run to max_attempts"
    );
    assert!(result.text.is_empty());
    assert_eq!(result.errors.len(), 1, "the client must be told why");
    assert_eq!(result.done_count, 1);
}

// Congestion is exempt: a busy upstream cannot emit content, and cutting it off
// after three tries would defeat §7.3 entirely.
#[tokio::test]
async fn test_repeated_congestion_is_exempt_from_the_no_progress_cutoff() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=503&fail_first=1000",
            config.upstream.base_url
        );
        config.resume.max_attempts = 8;
        config.resume.max_no_progress_attempts = 3;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        8,
        "503s should keep retrying to max_attempts, not stop at 3"
    );
    assert_eq!(result.done_count, 1);
}

// An attempt that emitted new text made progress, so the counter resets and
// resumption continues — repeated mid-stream cuts must not be cut off at three.
#[tokio::test]
async fn test_progress_resets_the_no_progress_counter() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url =
            format!("{}/chat/completions?cut_after=5", config.upstream.base_url);
        config.resume.max_attempts = 6;
        config.resume.max_no_progress_attempts = 3;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        6,
        "each attempt advanced the response, so all attempts should be used"
    );
    assert!(
        result.text.starts_with("Line 1\n"),
        "partial content should still have been delivered"
    );
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert_eq!(result.done_count, 1);
}

// The attempt limits are checked at the top of the loop, so without an explicit
// guard the last failed attempt sleeps out a full backoff before discovering it
// had no retry left — up to 30s of dead waiting at the end of a doomed request.
#[tokio::test]
async fn test_final_failed_attempt_does_not_sleep() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!(
            "{}/chat/completions?status=503&fail_first=100",
            config.upstream.base_url
        );
        config.resume.max_attempts = 2;
        config.resume.backoff_initial_ms = 1_500;
        config.resume.backoff_max_ms = 5_000;
        config.resume.backoff_multiplier = 2.0;
        config.resume.backoff_jitter = 0.0;
    })
    .await;

    let started = Instant::now();
    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;
    let elapsed = started.elapsed();

    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
    assert_eq!(result.done_count, 1);
    // Only the backoff between attempts 1 and 2 should be paid: ~1.5s, not the
    // ~4.5s that a trailing 3s sleep would add.
    assert!(
        elapsed < Duration::from_millis(2_800),
        "the final failure slept before giving up: {elapsed:?}"
    );
}

// §6.3: an upstream that ignores the prefix and regenerates the whole response
// after more than `dedup_window_chars` were already emitted. The tail window
// cannot see this overlap — the retry's opening no longer touches the end of
// what was emitted — so without head-window detection every character emitted
// so far is delivered a second time.
#[tokio::test]
async fn test_full_restart_beyond_the_dedup_window_is_deduplicated() {
    let harness = TestHarness::setup(|config| {
        // 40 lines is well past the 240-character window.
        config.upstream.base_url = format!(
            "{}/chat/completions?cut_after=40&cut_first_n_requests=1&restart=true",
            config.upstream.base_url
        );
        config.resume.dedup_window_chars = 240;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    let expected = default_corpus();
    assert!(
        expected.chars().count() > 240,
        "fixture must exceed the dedup window to be meaningful"
    );
    assert_eq!(
        harness
            .mock_state
            .request_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        2
    );
    assert_eq!(
        result.text, expected,
        "the replayed prefix was not deduplicated"
    );
    assert_eq!(result.text.matches("Line 1\n").count(), 1);
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert_eq!(result.done_count, 1);
}

// The accumulation cap protects memory without truncating real generations: it
// is far above any legitimate response, and hitting it ends the stream honestly
// rather than silently.
#[tokio::test]
async fn test_accumulated_text_cap_ends_the_stream() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!("{}/chat/completions", config.upstream.base_url);
        config.resume.max_accumulated_bytes = 64;
    })
    .await;

    let result = collect_stream_response(
        &harness.client,
        &format!("http://{}/v1/chat/completions", harness.proxy_addr),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Count lines"}],
            "stream": true
        }),
    )
    .await;

    assert!(
        result.text.len() <= 64,
        "the cap must actually bound memory"
    );
    assert!(
        !result.text.is_empty(),
        "text emitted before the cap is kept"
    );
    assert_eq!(
        result.finish_reasons,
        vec!["length".to_string()],
        "truncation must be reported honestly, not as a clean stop"
    );
    assert_eq!(result.done_count, 1);
}

// §5 / M13: body-extraction failures happen before the handler runs, and axum's
// own plain-text rejections are not what an OpenAI-compatible client parses.
#[tokio::test]
async fn test_malformed_request_bodies_get_openai_shaped_errors() {
    let harness = TestHarness::setup(|config| {
        config.upstream.base_url = format!("{}/chat/completions", config.upstream.base_url);
        config.server.max_body_bytes = 4096;
    })
    .await;
    let url = format!("http://{}/v1/chat/completions", harness.proxy_addr);

    // Malformed JSON.
    let resp = harness
        .client
        .post(&url)
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = resp.json().await.expect("rejection must still be JSON");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"].is_string());

    // Wrong content type.
    let resp = harness
        .client
        .post(&url)
        .header("content-type", "text/plain")
        .body("{\"model\":\"m\"}")
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "a JSON body must be declared as JSON"
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");

    // Oversized body.
    let resp = harness
        .client
        .post(&url)
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "x".repeat(64 * 1024)}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);
    let body: Value = resp.json().await.expect("413 must be OpenAI-shaped too");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}
