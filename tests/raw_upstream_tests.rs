//! Tests that need byte-level control of the upstream connection.
//!
//! `src/mock.rs` speaks through axum, so every failure it can simulate ends the
//! response body cleanly — the proxy always sees a well-formed EOF. The wire
//! failures §2 and §7 actually care about (TCP reset, broken pipe, incomplete
//! chunked message, TLS failure), malformed SSE framing, and the abort-on-client-
//! disconnect requirement of §11 can only be exercised from a raw socket, which
//! is what this file provides.

use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use llm_proxy::config::Config;
use llm_proxy::create_app;
use llm_proxy::mock::default_corpus;

// ---------------------------------------------------------------------------
// Raw upstream harness
// ---------------------------------------------------------------------------

type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

struct RawRequest {
    /// 1-based index of this request across the lifetime of the upstream.
    num: u32,
    body: Value,
}

struct RawUpstream {
    addr: SocketAddr,
    requests: Arc<AtomicU32>,
}

impl RawUpstream {
    fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    /// How many HTTP requests the proxy has issued so far.
    fn requests(&self) -> u32 {
        self.requests.load(Ordering::SeqCst)
    }
}

/// Bind a raw TCP upstream. Each accepted connection is parsed as one HTTP/1.1
/// request and then handed, socket and all, to `handler`.
async fn spawn_raw_upstream<F>(handler: F) -> RawUpstream
where
    F: Fn(RawRequest, TcpStream) -> BoxFuture + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handler = Arc::new(handler);
    let counter = Arc::new(AtomicU32::new(0));
    let requests = Arc::clone(&counter);

    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            let handler = Arc::clone(&handler);
            let counter = Arc::clone(&counter);
            tokio::spawn(async move {
                let Some(body) = read_http_request(&mut sock).await else {
                    return;
                };
                let num = counter.fetch_add(1, Ordering::SeqCst) + 1;
                handler(RawRequest { num, body }, sock).await;
            });
        }
    });

    RawUpstream { addr, requests }
}

/// Read one HTTP request off the socket and return its JSON body.
/// Returns `None` if the peer never sent a parseable request (e.g. a TLS
/// ClientHello, or a connection that closed first).
async fn read_http_request(sock: &mut TcpStream) -> Option<Value> {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];

    let head_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
    };

    let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
    let content_length: usize = head
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);

    while buf.len() < head_end + content_length {
        let n = sock.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
    }

    let body = &buf[head_end..head_end + content_length];
    if body.is_empty() {
        // A bodyless request (e.g. `GET /v1/models`) is still a valid request.
        return Some(Value::Null);
    }
    serde_json::from_slice(body).ok()
}

const SSE_HEAD: &str = "HTTP/1.1 200 OK\r\n\
                        Content-Type: text/event-stream\r\n\
                        Cache-Control: no-cache\r\n\
                        Transfer-Encoding: chunked\r\n\
                        Connection: close\r\n\r\n";

/// Write one HTTP chunked-transfer frame containing `payload`.
async fn write_frame(sock: &mut TcpStream, payload: &str) {
    let framed = format!("{:x}\r\n{}\r\n", payload.len(), payload);
    let _ = sock.write_all(framed.as_bytes()).await;
    let _ = sock.flush().await;
}

async fn write_sse(sock: &mut TcpStream, chunk: &Value) {
    write_frame(sock, &format!("data: {chunk}\n\n")).await;
}

async fn write_done(sock: &mut TcpStream) {
    write_frame(sock, "data: [DONE]\n\n").await;
}

/// Terminate the chunked body and half-close, so the proxy sees a clean end of
/// message rather than a truncated one.
async fn end_stream_cleanly(sock: &mut TcpStream) {
    let _ = sock.write_all(b"0\r\n\r\n").await;
    let _ = sock.flush().await;
    let _ = sock.shutdown().await;
}

/// Announce a chunk larger than what we then write, and hang up. hyper reports
/// this as an incomplete message — the "incomplete message" wire failure of §7.
async fn truncate_mid_chunk(mut sock: TcpStream) {
    let _ = sock.write_all(b"40\r\ndata: {\"id\":\"trunc\"").await;
    let _ = sock.flush().await;
    drop(sock);
}

/// Hang up with SO_LINGER 0, which makes the kernel send RST instead of FIN —
/// the "connection reset" wire failure of §7. `set_linger` is deprecated because
/// a non-zero linger blocks the thread on drop; a zero linger is exactly the
/// non-blocking, reset-now case, which is what this test needs.
#[allow(deprecated)]
async fn reset_connection(sock: TcpStream) {
    // Give the peer a moment to drain what was already written, so the reset
    // lands on an established, idle connection rather than racing the payload.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let _ = sock.set_linger(Some(Duration::ZERO));
    drop(sock);
}

// ---------------------------------------------------------------------------
// Corpus helpers — mirror src/mock.rs so continuation attempts resume correctly
// ---------------------------------------------------------------------------

/// Number of characters the proxy says it has already emitted, taken from the
/// trailing assistant message of a continuation request (§6.4).
fn resume_offset(body: &Value) -> usize {
    body.get("messages")
        .and_then(|m| m.as_array())
        .into_iter()
        .flatten()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        .map(|c| c.chars().count())
        .unwrap_or(0)
}

fn corpus_lines_from(offset: usize) -> Vec<String> {
    let chars: Vec<char> = default_corpus().chars().collect();
    let remaining: String = chars.get(offset..).unwrap_or_default().iter().collect();
    remaining.split_inclusive('\n').map(String::from).collect()
}

fn content_chunk(content: &str, with_role: bool) -> Value {
    let mut delta = serde_json::Map::new();
    if with_role {
        delta.insert("role".into(), json!("assistant"));
    }
    delta.insert("content".into(), json!(content));
    json!({
        "id": "upstream-raw",
        "object": "chat.completion.chunk",
        "created": 1690000000,
        "model": "raw-mock-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": Value::Null}]
    })
}

fn finish_chunk(reason: &str) -> Value {
    json!({
        "id": "upstream-raw",
        "object": "chat.completion.chunk",
        "created": 1690000000,
        "model": "raw-mock-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]
    })
}

fn tool_delta_chunk(tool_calls: Value, with_role: bool) -> Value {
    let mut delta = serde_json::Map::new();
    if with_role {
        delta.insert("role".into(), json!("assistant"));
    }
    delta.insert("tool_calls".into(), tool_calls);
    json!({
        "id": "upstream-raw",
        "object": "chat.completion.chunk",
        "created": 1690000000,
        "model": "raw-mock-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": Value::Null}]
    })
}

/// Stream the remainder of the corpus from `offset`, then finish cleanly.
async fn serve_rest_of_corpus(sock: &mut TcpStream, offset: usize) {
    for (i, line) in corpus_lines_from(offset).iter().enumerate() {
        write_sse(sock, &content_chunk(line, i == 0)).await;
    }
    write_sse(sock, &finish_chunk("stop")).await;
    write_done(sock).await;
    end_stream_cleanly(sock).await;
}

// ---------------------------------------------------------------------------
// Proxy harness
// ---------------------------------------------------------------------------

async fn spawn_proxy(base_url: String, customize: impl FnOnce(&mut Config)) -> SocketAddr {
    let mut config = Config::default();
    config.server.bind = "127.0.0.1:0".to_string();
    config.upstream.base_url = base_url;
    config.upstream.api_key_env = String::new();
    config.resume.backoff_initial_ms = 20;
    config.resume.backoff_max_ms = 50;
    config.resume.backoff_jitter = 0.0;
    config.resume.max_attempts = 6;
    config.resume.first_chunk_timeout_secs = 5;
    config.resume.idle_timeout_secs = 5;

    customize(&mut config);

    let app = create_app(Arc::new(config));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

#[derive(Default)]
struct StreamResult {
    text: String,
    reasoning: String,
    chunk_ids: Vec<String>,
    done_count: usize,
    finish_reasons: Vec<String>,
    roles: Vec<String>,
    tool_call_chunks: Vec<Value>,
    errors: Vec<Value>,
}

async fn collect(client: &Client, proxy: SocketAddr, body: Value) -> StreamResult {
    let resp = client
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("proxy request failed");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let mut out = StreamResult::default();
    let mut events = resp.bytes_stream().eventsource();
    while let Some(event) = events.next().await {
        let data = event.expect("SSE decode error").data;
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            out.done_count += 1;
            continue;
        }
        let json: Value = serde_json::from_str(data).expect("downstream chunk must be JSON");

        if let Some(err) = json.get("error") {
            out.errors.push(err.clone());
            continue;
        }
        if let Some(id) = json.get("id").and_then(|i| i.as_str()) {
            out.chunk_ids.push(id.to_string());
        }
        let Some(choice0) = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
        else {
            continue;
        };
        if let Some(fr) = choice0.get("finish_reason").and_then(|f| f.as_str()) {
            out.finish_reasons.push(fr.to_string());
        }
        if let Some(delta) = choice0.get("delta") {
            if let Some(role) = delta.get("role").and_then(|r| r.as_str()) {
                out.roles.push(role.to_string());
            }
            if let Some(c) = delta.get("content").and_then(|c| c.as_str()) {
                out.text.push_str(c);
            }
            for field in ["reasoning_content", "reasoning"] {
                if let Some(r) = delta.get(field).and_then(|r| r.as_str()) {
                    out.reasoning.push_str(r);
                }
            }
            if delta.get("tool_calls").is_some() {
                out.tool_call_chunks.push(delta["tool_calls"].clone());
            }
        }
    }
    out
}

fn test_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .unwrap()
}

fn chat_body() -> Value {
    json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Count lines"}],
        "stream": true
    })
}

// ---------------------------------------------------------------------------
// §7 wire failures
// ---------------------------------------------------------------------------

/// A chunked body that stops mid-frame is an incomplete message, not an EOF.
/// The proxy must classify it as retryable and resume seamlessly.
#[tokio::test]
async fn test_incomplete_chunked_message_is_retried() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            if req.num == 1 {
                let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
                for (i, line) in corpus_lines_from(0).iter().take(5).enumerate() {
                    write_sse(&mut sock, &content_chunk(line, i == 0)).await;
                }
                truncate_mid_chunk(sock).await;
            } else {
                let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
                serve_rest_of_corpus(&mut sock, resume_offset(&req.body)).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(result.roles.len(), 1, "role delta must be emitted once");
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert!(result.errors.is_empty());
}

/// A TCP reset (RST) mid-stream is the canonical "connection reset by peer"
/// failure of §2/§7. Whatever the reset discards, the resumed attempt must still
/// deliver a complete, non-duplicated body.
#[tokio::test]
async fn test_tcp_reset_mid_stream_is_retried() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            if req.num == 1 {
                let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
                for (i, line) in corpus_lines_from(0).iter().take(8).enumerate() {
                    write_sse(&mut sock, &content_chunk(line, i == 0)).await;
                }
                reset_connection(sock).await;
            } else {
                let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
                serve_rest_of_corpus(&mut sock, resume_offset(&req.body)).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(result.roles.len(), 1);
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
}

/// The connection dies before any response head arrives — a broken pipe during
/// dispatch. This exercises the pre-stream transport branch of the outer loop,
/// which no mock-driven test reaches.
#[tokio::test]
async fn test_connection_closed_before_response_head_is_retried() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            if req.num == 1 {
                // Close without writing a single response byte.
                drop(sock);
            } else {
                let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
                serve_rest_of_corpus(&mut sock, resume_offset(&req.body)).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert!(result.errors.is_empty());
}

/// A TLS error is retryable (§7), and when it never clears the proxy gives up.
/// Nothing was ever emitted downstream here, so §7's "clean stop" — which exists
/// to protect a partial answer — would leave the client with a silent, empty,
/// apparently-successful response. The real upstream error goes out instead.
#[tokio::test]
async fn test_tls_failure_is_retried_then_surfaces_the_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let connections = Arc::new(AtomicU32::new(0));
    let seen = Arc::clone(&connections);

    // Speak plaintext at a client expecting TLS: the handshake always fails.
    tokio::spawn(async move {
        while let Ok((sock, _)) = listener.accept().await {
            seen.fetch_add(1, Ordering::SeqCst);
            drop(sock);
        }
    });

    let proxy = spawn_proxy(format!("https://127.0.0.1:{}/v1", addr.port()), |config| {
        config.resume.max_attempts = 3;
    })
    .await;

    let result = collect(&test_client(), proxy, chat_body()).await;

    assert!(result.text.is_empty());
    assert_eq!(
        result.errors.len(),
        1,
        "a give-up with nothing emitted must tell the client why"
    );
    assert!(
        result.errors[0]["message"].is_string(),
        "the surfaced error carries the upstream cause: {:?}",
        result.errors[0]
    );
    assert!(
        result.finish_reasons.is_empty(),
        "no synthetic `stop` should claim the empty response succeeded"
    );
    assert_eq!(
        result.done_count, 1,
        "the stream still terminates exactly once"
    );
    assert_eq!(
        connections.load(Ordering::SeqCst),
        3,
        "a TLS failure must be retried up to max_attempts"
    );
}

// ---------------------------------------------------------------------------
// SSE framing robustness
// ---------------------------------------------------------------------------

/// Real upstreams and the reverse proxies in front of them interleave comment
/// heartbeats, non-chunk events, and multi-line `data:` fields. None of these
/// may corrupt the reassembled body or leak downstream.
#[tokio::test]
async fn test_malformed_and_heartbeat_sse_events_are_tolerated() {
    let upstream = spawn_raw_upstream(|_req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;

            // Comment-only heartbeat: no event is dispatched at all.
            write_frame(&mut sock, ": heartbeat\n\n").await;
            // Data that is not JSON must be skipped, not forwarded.
            write_frame(&mut sock, "data: not json at all\n\n").await;
            // A named event carrying an unrelated payload.
            write_frame(&mut sock, "event: ping\ndata: {\"noise\":true}\n\n").await;

            write_sse(&mut sock, &content_chunk("Hello ", true)).await;

            // A single `data:` field split across physical lines: the SSE decoder
            // rejoins them with \n, which is still valid JSON.
            write_frame(
                &mut sock,
                "data: {\"id\":\"u1\",\ndata:  \"choices\":[{\"index\":0,\
                 \"delta\":{\"content\":\"world\"},\"finish_reason\":null}]}\n\n",
            )
            .await;

            // An SSE field the proxy has no use for.
            write_frame(&mut sock, "retry: 3000\n\n").await;
            // An empty data field.
            write_frame(&mut sock, "data:\n\n").await;

            write_sse(&mut sock, &finish_chunk("stop")).await;
            write_done(&mut sock).await;
            end_stream_cleanly(&mut sock).await;
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(result.text, "Hello world");
    assert_eq!(result.done_count, 1);
    assert_eq!(result.roles.len(), 1);
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert!(result.errors.is_empty());
}

/// §8: comment heartbeats are upstream activity, so a stream that keeps itself
/// alive with `: keep-alive` past `idle_timeout_secs` must NOT be torn down and
/// re-issued. The whole body arrives on a single attempt.
#[tokio::test]
async fn test_heartbeat_comments_keep_the_stream_alive() {
    let upstream = spawn_raw_upstream(|_req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            let lines = corpus_lines_from(0);
            let (head, tail) = lines.split_at(4);

            for (i, line) in head.iter().enumerate() {
                write_sse(&mut sock, &content_chunk(line, i == 0)).await;
            }

            // Silence far longer than idle_timeout_secs, broken only by comments.
            for _ in 0..12 {
                write_frame(&mut sock, ": keep-alive\n\n").await;
                tokio::time::sleep(Duration::from_millis(250)).await;
            }

            for line in tail {
                write_sse(&mut sock, &content_chunk(line, false)).await;
            }
            write_sse(&mut sock, &finish_chunk("stop")).await;
            write_done(&mut sock).await;
            end_stream_cleanly(&mut sock).await;
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.resume.idle_timeout_secs = 1;
    })
    .await;

    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(result.roles.len(), 1);
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert_eq!(
        upstream.requests(),
        1,
        "a heartbeating stream must not be re-issued"
    );
}

/// Genuine silence — no bytes at all — still trips the idle watchdog, and the
/// retry resumes from the accumulated prefix rather than restarting.
#[tokio::test]
async fn test_genuine_silence_still_trips_the_idle_timeout() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            if req.num == 1 {
                for (i, line) in corpus_lines_from(0).iter().take(4).enumerate() {
                    write_sse(&mut sock, &content_chunk(line, i == 0)).await;
                }
                tokio::time::sleep(Duration::from_secs(10)).await;
            } else {
                serve_rest_of_corpus(&mut sock, resume_offset(&req.body)).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.resume.idle_timeout_secs = 1;
    })
    .await;

    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert_eq!(upstream.requests(), 2, "the stall should cost one retry");
}

/// §8 defines `first_chunk_timeout_secs` as an absolute budget from dispatch
/// until the first content/tool-call/reasoning delta. Heartbeats are liveness,
/// not progress, so they must not extend it.
#[tokio::test]
async fn test_heartbeats_do_not_extend_the_first_chunk_timeout() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            if req.num == 1 {
                // Alive, but never produces a first chunk.
                for _ in 0..40 {
                    write_frame(&mut sock, ": keep-alive\n\n").await;
                    tokio::time::sleep(Duration::from_millis(150)).await;
                }
            } else {
                serve_rest_of_corpus(&mut sock, resume_offset(&req.body)).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.resume.first_chunk_timeout_secs = 1;
        config.resume.idle_timeout_secs = 30;
    })
    .await;

    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(result.text, default_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(
        upstream.requests(),
        2,
        "the first-chunk deadline should have fired despite the heartbeats"
    );
}

// ---------------------------------------------------------------------------
// §9 tool-call edge cases
// ---------------------------------------------------------------------------

/// Two concurrent tool calls arriving as interleaved fragments must be keyed by
/// index and emitted as one consolidated delta in index order.
#[tokio::test]
async fn test_interleaved_fragmented_tool_calls_assemble_in_index_order() {
    let upstream = spawn_raw_upstream(|_req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            write_sse(
                &mut sock,
                &tool_delta_chunk(
                    json!([{"index": 1, "id": "call_b", "type": "function",
                            "function": {"name": "beta", "arguments": "{\"y\":"}}]),
                    true,
                ),
            )
            .await;
            write_sse(
                &mut sock,
                &tool_delta_chunk(
                    json!([{"index": 0, "id": "call_a", "type": "function",
                            "function": {"name": "alpha", "arguments": "{\"x\":"}}]),
                    false,
                ),
            )
            .await;
            write_sse(
                &mut sock,
                &tool_delta_chunk(
                    json!([{"index": 1, "function": {"arguments": " 2}"}}]),
                    false,
                ),
            )
            .await;
            write_sse(
                &mut sock,
                &tool_delta_chunk(
                    json!([{"index": 0, "function": {"arguments": " 1}"}}]),
                    false,
                ),
            )
            .await;
            write_sse(&mut sock, &finish_chunk("tool_calls")).await;
            write_done(&mut sock).await;
            end_stream_cleanly(&mut sock).await;
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(
        result.tool_call_chunks.len(),
        1,
        "buffered mode must emit exactly one consolidated tool-call delta"
    );
    let calls = result.tool_call_chunks[0].as_array().unwrap();
    assert_eq!(calls.len(), 2);

    assert_eq!(calls[0]["index"], 0);
    assert_eq!(calls[0]["id"], "call_a");
    assert_eq!(calls[0]["function"]["name"], "alpha");
    assert_eq!(calls[0]["function"]["arguments"], "{\"x\": 1}");

    assert_eq!(calls[1]["index"], 1);
    assert_eq!(calls[1]["id"], "call_b");
    assert_eq!(calls[1]["function"]["name"], "beta");
    assert_eq!(calls[1]["function"]["arguments"], "{\"y\": 2}");

    assert_eq!(result.finish_reasons, vec!["tool_calls".to_string()]);
    assert_eq!(result.done_count, 1);
    assert_eq!(result.roles.len(), 1);
}

/// §9: "If an attempt fails mid-tool-call, discard the uncommitted tool call
/// state and retry from the last committed text state." A half-received argument
/// string must not be spliced onto the arguments of the successful attempt.
#[tokio::test]
async fn test_partial_tool_call_state_is_discarded_across_attempts() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            if req.num == 1 {
                write_sse(
                    &mut sock,
                    &tool_delta_chunk(
                        json!([{"index": 0, "id": "call_partial", "type": "function",
                                "function": {"name": "lookup_stock", "arguments": "{\"ticker\":"}}]),
                        true,
                    ),
                )
                .await;
                truncate_mid_chunk(sock).await;
            } else {
                write_sse(
                    &mut sock,
                    &tool_delta_chunk(
                        json!([{"index": 0, "id": "call_final", "type": "function",
                                "function": {"name": "lookup_stock", "arguments": "{\"ticker\":"}}]),
                        true,
                    ),
                )
                .await;
                write_sse(
                    &mut sock,
                    &tool_delta_chunk(
                        json!([{"index": 0, "function": {"arguments": " \"AAPL\"}"}}]),
                        false,
                    ),
                )
                .await;
                write_sse(&mut sock, &finish_chunk("tool_calls")).await;
                write_done(&mut sock).await;
                end_stream_cleanly(&mut sock).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(result.tool_call_chunks.len(), 1);
    let calls = result.tool_call_chunks[0].as_array().unwrap();
    assert_eq!(calls.len(), 1, "the discarded attempt must not add a call");
    assert_eq!(calls[0]["id"], "call_final");
    assert_eq!(calls[0]["function"]["name"], "lookup_stock");
    assert_eq!(
        calls[0]["function"]["arguments"], "{\"ticker\": \"AAPL\"}",
        "arguments from the abandoned attempt must not be concatenated"
    );
    assert_eq!(result.finish_reasons, vec!["tool_calls".to_string()]);
    assert_eq!(result.done_count, 1);
}

/// `buffer_tool_calls = false` forwards deltas as they arrive. Across a cut the
/// proxy must still rewrite chunk identity, emit `role` once, and suppress the
/// intermediate finish reasons — but it does not (and per §9 is not asked to)
/// deduplicate the fragments the abandoned attempt already forwarded.
#[tokio::test]
async fn test_tool_call_passthrough_forwards_deltas_across_a_cut() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            if req.num == 1 {
                write_sse(
                    &mut sock,
                    &tool_delta_chunk(
                        json!([{"index": 0, "id": "call_x", "type": "function",
                                "function": {"name": "lookup_stock", "arguments": "{\"tic"}}]),
                        true,
                    ),
                )
                .await;
                truncate_mid_chunk(sock).await;
            } else {
                write_sse(
                    &mut sock,
                    &tool_delta_chunk(
                        json!([{"index": 0, "id": "call_x", "type": "function",
                                "function": {"name": "lookup_stock", "arguments": "{\"ticker\": \"AAPL\"}"}}]),
                        true,
                    ),
                )
                .await;
                write_sse(&mut sock, &finish_chunk("tool_calls")).await;
                write_done(&mut sock).await;
                end_stream_cleanly(&mut sock).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.tools.buffer_tool_calls = false;
    })
    .await;

    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(
        result.tool_call_chunks.len(),
        2,
        "passthrough forwards each upstream tool-call delta as it arrives"
    );
    assert_eq!(result.roles.len(), 1, "role must still be emitted once");
    assert_eq!(
        result.finish_reasons,
        vec!["tool_calls".to_string()],
        "intermediate finish reasons must stay suppressed"
    );
    assert_eq!(result.done_count, 1);

    // §6.5: every chunk carries the proxy's own id, never the upstream's.
    assert!(!result.chunk_ids.is_empty());
    assert!(
        result
            .chunk_ids
            .iter()
            .all(|id| id.starts_with("chatcmpl-") && id != "upstream-raw"),
        "chunk ids were not rewritten: {:?}",
        result.chunk_ids
    );
    assert_eq!(
        result
            .chunk_ids
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        1,
        "all chunks must share one downstream id"
    );
}

// ---------------------------------------------------------------------------
// §11 client disconnect
// ---------------------------------------------------------------------------

const NEVER_CLOSED: u64 = u64::MAX;

/// Drive the proxy, then hang up after `read_for`, and report how long the
/// upstream socket stayed open afterwards.
async fn measure_upstream_close_after_disconnect(
    upstream_head: Option<&'static str>,
    read_for: Duration,
) -> u64 {
    let closed_after_ms = Arc::new(AtomicU64::new(NEVER_CLOSED));
    let recorder = Arc::clone(&closed_after_ms);

    let upstream = spawn_raw_upstream(move |_req, mut sock| {
        let recorder = Arc::clone(&recorder);
        Box::pin(async move {
            let started = Instant::now();
            if let Some(head) = upstream_head {
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.flush().await;
            }
            // Never send a body. Wait to see whether the proxy hangs up on us.
            let mut buf = [0u8; 64];
            let closed = tokio::select! {
                read = sock.read(&mut buf) => matches!(read, Ok(0) | Err(_)),
                _ = tokio::time::sleep(Duration::from_secs(10)) => false,
            };
            if closed {
                recorder.store(started.elapsed().as_millis() as u64, Ordering::SeqCst);
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        // Long enough that only the disconnect, not a timeout, can end the attempt.
        config.resume.first_chunk_timeout_secs = 30;
        config.resume.idle_timeout_secs = 30;
    })
    .await;

    let resp = test_client()
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .expect("proxy request failed");
    tokio::time::sleep(read_for).await;
    drop(resp);

    // Allow the abort to propagate.
    tokio::time::sleep(Duration::from_secs(3)).await;
    closed_after_ms.load(Ordering::SeqCst)
}

/// §11: a disconnect while the upstream request is still in flight — before any
/// response head has arrived — must abort that request rather than leaving it
/// running until the first-chunk deadline.
#[tokio::test]
async fn test_client_disconnect_during_dispatch_aborts_upstream() {
    let closed_after_ms =
        measure_upstream_close_after_disconnect(None, Duration::from_millis(300)).await;

    assert_ne!(
        closed_after_ms, NEVER_CLOSED,
        "upstream request was left in flight after the client disconnected"
    );
    assert!(
        closed_after_ms < 3_000,
        "upstream abort took {closed_after_ms}ms; expected it to follow the disconnect promptly"
    );
}

/// §11 again, one stage later: the upstream has answered with a retryable status
/// but is still dribbling out the error body when the client hangs up.
#[tokio::test]
async fn test_client_disconnect_during_error_body_read_aborts_upstream() {
    const ERROR_HEAD: &str = "HTTP/1.1 503 Service Unavailable\r\n\
                              Content-Type: application/json\r\n\
                              Transfer-Encoding: chunked\r\n\r\n";

    let closed_after_ms =
        measure_upstream_close_after_disconnect(Some(ERROR_HEAD), Duration::from_millis(300)).await;

    assert_ne!(
        closed_after_ms, NEVER_CLOSED,
        "upstream error-body read was left in flight after the client disconnected"
    );
    assert!(
        closed_after_ms < 3_000,
        "upstream abort took {closed_after_ms}ms; expected it to follow the disconnect promptly"
    );
}

// ---------------------------------------------------------------------------
// Multi-byte text across a retry boundary
// ---------------------------------------------------------------------------

/// A corpus of multi-byte graphemes: CJK, accented Latin, emoji (including a
/// ZWJ sequence), and RTL text. `default_corpus()` is pure ASCII, so nothing
/// else in the suite drives non-ASCII through the real dedup path.
fn multibyte_corpus() -> String {
    let mut out = String::new();
    for i in 1..=40 {
        out.push_str(&format!("行 {i}: café — naïve — 🇯🇵 — 👩‍👩‍👧‍👦 — مرحبا — Ω≈ç√∫\n"));
    }
    out
}

fn multibyte_lines_from(offset: usize) -> Vec<String> {
    let chars: Vec<char> = multibyte_corpus().chars().collect();
    let remaining: String = chars.get(offset..).unwrap_or_default().iter().collect();
    remaining.split_inclusive('\n').map(String::from).collect()
}

/// §6.3 requires all dedup comparisons and slicing to operate on `char`
/// boundaries. A cut mid-way through multi-byte content must resume without
/// duplicating, dropping, or splitting a character.
#[tokio::test]
async fn test_multibyte_text_survives_a_retry_cut() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            let offset = resume_offset(&req.body);
            let lines = multibyte_lines_from(offset);
            if req.num == 1 {
                for (i, line) in lines.iter().take(9).enumerate() {
                    write_sse(&mut sock, &content_chunk(line, i == 0)).await;
                }
                truncate_mid_chunk(sock).await;
            } else {
                for (i, line) in lines.iter().enumerate() {
                    write_sse(&mut sock, &content_chunk(line, i == 0)).await;
                }
                write_sse(&mut sock, &finish_chunk("stop")).await;
                write_done(&mut sock).await;
                end_stream_cleanly(&mut sock).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(result.text, multibyte_corpus());
    assert_eq!(result.done_count, 1);
    assert_eq!(result.roles.len(), 1);
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert_eq!(upstream.requests(), 2);
}

/// The same, but the upstream replays a chunk of multi-byte text it already
/// sent, so the overlap window has to match on characters rather than bytes.
#[tokio::test]
async fn test_multibyte_overlap_is_deduplicated() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            if req.num == 1 {
                for (i, line) in multibyte_lines_from(0).iter().take(9).enumerate() {
                    write_sse(&mut sock, &content_chunk(line, i == 0)).await;
                }
                truncate_mid_chunk(sock).await;
            } else {
                // Rewind 30 characters: the retry re-sends text already emitted.
                let offset = resume_offset(&req.body).saturating_sub(30);
                for (i, line) in multibyte_lines_from(offset).iter().enumerate() {
                    write_sse(&mut sock, &content_chunk(line, i == 0)).await;
                }
                write_sse(&mut sock, &finish_chunk("stop")).await;
                write_done(&mut sock).await;
                end_stream_cleanly(&mut sock).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(
        result.text,
        multibyte_corpus(),
        "multi-byte overlap was not deduplicated cleanly"
    );
    assert_eq!(result.done_count, 1);
}

// ---------------------------------------------------------------------------
// §5 models endpoint liveness
// ---------------------------------------------------------------------------

/// The shared HTTP client sets only a connect timeout. An upstream that accepts
/// the connection and then stalls must not hang `GET /v1/models` forever.
#[tokio::test]
async fn test_models_endpoint_upstream_stall_times_out() {
    let upstream = spawn_raw_upstream(|_req, sock| {
        Box::pin(async move {
            // Accept, answer nothing, hold the socket open.
            tokio::time::sleep(Duration::from_secs(30)).await;
            drop(sock);
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.resume.first_chunk_timeout_secs = 2;
    })
    .await;

    let started = Instant::now();
    let resp = test_client()
        .get(format!("http://{proxy}/v1/models"))
        .send()
        .await
        .expect("models request should return, not hang");
    let elapsed = started.elapsed();

    assert_eq!(resp.status(), reqwest::StatusCode::GATEWAY_TIMEOUT);
    assert!(
        elapsed < Duration::from_secs(10),
        "models endpoint took {elapsed:?}; the deadline did not bound it"
    );

    let body: Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].is_string());
    assert_eq!(body["error"]["type"], "api_error");
}

/// A models upstream that stalls midway through the body is bounded too.
#[tokio::test]
async fn test_models_endpoint_stalled_body_times_out() {
    let upstream = spawn_raw_upstream(|_req, mut sock| {
        Box::pin(async move {
            let _ = sock
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                      Transfer-Encoding: chunked\r\n\r\n",
                )
                .await;
            let _ = sock.flush().await;
            tokio::time::sleep(Duration::from_secs(30)).await;
            drop(sock);
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.resume.first_chunk_timeout_secs = 2;
    })
    .await;

    let started = Instant::now();
    let resp = test_client()
        .get(format!("http://{proxy}/v1/models"))
        .send()
        .await
        .expect("models request should return, not hang");

    assert_eq!(resp.status(), reqwest::StatusCode::GATEWAY_TIMEOUT);
    assert!(started.elapsed() < Duration::from_secs(10));
}

// ---------------------------------------------------------------------------
// Failed generations, mixed deltas, and in-band errors
// ---------------------------------------------------------------------------

/// §9: a failed generation must discard its half-assembled tool call. Flushing
/// it first would hand the client a broken call followed by the retry's
/// replacement — two calls for one intended invocation.
#[tokio::test]
async fn test_malformed_function_call_discards_partial_tool_state() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            if req.num == 1 {
                write_sse(
                    &mut sock,
                    &tool_delta_chunk(
                        json!([{"index": 0, "id": "call_partial", "type": "function",
                                "function": {"name": "get_weather", "arguments": "{\"ci"}}]),
                        true,
                    ),
                )
                .await;
                write_sse(
                    &mut sock,
                    &finish_chunk("function_call_filter: MALFORMED_FUNCTION_CALL"),
                )
                .await;
                write_done(&mut sock).await;
                end_stream_cleanly(&mut sock).await;
            } else {
                write_sse(
                    &mut sock,
                    &tool_delta_chunk(
                        json!([{"index": 0, "id": "call_good", "type": "function",
                                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]),
                        true,
                    ),
                )
                .await;
                write_sse(&mut sock, &finish_chunk("tool_calls")).await;
                write_done(&mut sock).await;
                end_stream_cleanly(&mut sock).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(
        upstream.requests(),
        2,
        "the failed generation should be retried"
    );
    assert_eq!(
        result.tool_call_chunks.len(),
        1,
        "the discarded attempt must not emit its partial call"
    );
    let calls = result.tool_call_chunks[0].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["id"], "call_good");
    assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(result.finish_reasons, vec!["tool_calls".to_string()]);
    assert_eq!(result.roles.len(), 1);
    assert_eq!(result.done_count, 1);
}

/// A single upstream event may carry content, reasoning and tool calls at once.
/// Passthrough mode must emit each field exactly once — forwarding the mixed
/// source object from the tool branch would re-send the other two.
#[tokio::test]
async fn test_passthrough_does_not_duplicate_mixed_deltas() {
    let upstream = spawn_raw_upstream(|_req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            write_sse(
                &mut sock,
                &json!({
                    "id": "upstream-raw",
                    "object": "chat.completion.chunk",
                    "created": 1690000000,
                    "model": "raw-mock-model",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "VISIBLE",
                            "reasoning_content": "THINKING",
                            "tool_calls": [{"index": 0, "id": "call_m", "type": "function",
                                            "function": {"name": "f", "arguments": "{}"}}]
                        },
                        "finish_reason": Value::Null
                    }]
                }),
            )
            .await;
            write_sse(&mut sock, &finish_chunk("tool_calls")).await;
            write_done(&mut sock).await;
            end_stream_cleanly(&mut sock).await;
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.tools.buffer_tool_calls = false;
    })
    .await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(result.text, "VISIBLE", "content emitted more than once");
    assert_eq!(
        result.reasoning, "THINKING",
        "reasoning emitted more than once"
    );
    assert_eq!(
        result.tool_call_chunks.len(),
        1,
        "tool calls emitted more than once"
    );
    assert_eq!(result.roles.len(), 1);
    assert_eq!(result.done_count, 1);
}

/// §7: an upstream can report failure inside a 200 stream. Such an event has no
/// delta, so ignoring it and honouring the following `[DONE]` would produce a
/// successful, empty completion.
#[tokio::test]
async fn test_in_band_error_event_is_retried() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            if req.num == 1 {
                write_sse(&mut sock, &json!({"error": {"message": "quota exhausted"}})).await;
                write_done(&mut sock).await;
                end_stream_cleanly(&mut sock).await;
            } else {
                serve_rest_of_corpus(&mut sock, resume_offset(&req.body)).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(upstream.requests(), 2, "an in-band error should be retried");
    assert_eq!(result.text, default_corpus());
    assert!(
        result.errors.is_empty(),
        "the recovered request must look clean"
    );
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert_eq!(result.done_count, 1);
}

/// ...but an in-band refusal is fatal, exactly as the same message would be over
/// a status code: retrying resends the content that was refused.
#[tokio::test]
async fn test_in_band_policy_error_is_fatal() {
    let upstream = spawn_raw_upstream(|_req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            write_sse(
                &mut sock,
                &json!({"error": {"message": "Request blocked by safety filter"}}),
            )
            .await;
            write_done(&mut sock).await;
            end_stream_cleanly(&mut sock).await;
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |_| {}).await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(upstream.requests(), 1, "a refusal must not be retried");
    assert_eq!(result.errors.len(), 1);
    assert!(result.errors[0]["message"]
        .as_str()
        .unwrap()
        .contains("safety filter"));
    assert_eq!(result.done_count, 1);
}

/// §9: `disable_resume_for_tool_responses` suppresses resumption only once text
/// has begun emitting. The mock upstream cannot express this — a request with
/// `tools` always gets its tool fixture, which emits no text — so it is driven
/// from a raw socket.
#[tokio::test]
async fn test_disable_resume_for_tool_responses_after_text() {
    fn body_with_tools() -> Value {
        json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "Count lines"}],
            "tools": [{"type": "function", "function": {"name": "noop"}}]
        })
    }

    async fn run(disable: bool) -> (StreamResult, u32) {
        let upstream = spawn_raw_upstream(|req, mut sock| {
            Box::pin(async move {
                let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
                if req.num == 1 {
                    for (i, line) in corpus_lines_from(0).iter().take(4).enumerate() {
                        write_sse(&mut sock, &content_chunk(line, i == 0)).await;
                    }
                    truncate_mid_chunk(sock).await;
                } else {
                    serve_rest_of_corpus(&mut sock, resume_offset(&req.body)).await;
                }
            })
        })
        .await;

        let proxy = spawn_proxy(upstream.base_url(), move |config| {
            config.tools.disable_resume_for_tool_responses = disable;
        })
        .await;
        let result = collect(&test_client(), proxy, body_with_tools()).await;
        let requests = upstream.requests();
        (result, requests)
    }

    let (disabled, disabled_requests) = run(true).await;
    assert_eq!(
        disabled_requests, 1,
        "resumption should have been suppressed"
    );
    assert!(
        !disabled.text.is_empty(),
        "the partial answer is still delivered"
    );
    assert_ne!(disabled.text, default_corpus(), "it must remain truncated");
    assert_eq!(disabled.finish_reasons, vec!["stop".to_string()]);
    assert_eq!(disabled.done_count, 1);

    let (enabled, enabled_requests) = run(false).await;
    assert_eq!(enabled_requests, 2, "control: resumption should proceed");
    assert_eq!(enabled.text, default_corpus());
}

// ---------------------------------------------------------------------------
// Reasoning continuity, emission accounting, admission control
// ---------------------------------------------------------------------------

/// §9: reasoning is forwarded but excluded from continuation `messages`, so an
/// attempt that dies after reasoning and before content leaves
/// `accumulated_text` empty — the retry is a plain re-request and the upstream
/// thinks aloud again. Reasoning needs its own overlap window.
#[tokio::test]
async fn test_reasoning_is_not_duplicated_across_a_retry() {
    let upstream = spawn_raw_upstream(|req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            let think = |text: &str, role: bool| {
                let mut delta = serde_json::Map::new();
                if role {
                    delta.insert("role".into(), json!("assistant"));
                }
                delta.insert("reasoning_content".into(), json!(text));
                json!({
                    "id": "upstream-raw",
                    "object": "chat.completion.chunk",
                    "created": 1690000000,
                    "model": "raw-mock-model",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": Value::Null}]
                })
            };

            // Both attempts think the same thoughts; only the second gets to
            // any visible content.
            write_sse(&mut sock, &think("Step one. ", true)).await;
            write_sse(&mut sock, &think("Step two. ", false)).await;

            if req.num == 1 {
                truncate_mid_chunk(sock).await;
            } else {
                write_sse(&mut sock, &content_chunk("Answer.", false)).await;
                write_sse(&mut sock, &finish_chunk("stop")).await;
                write_done(&mut sock).await;
                end_stream_cleanly(&mut sock).await;
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.resume.dedup_window_chars = 8;
    })
    .await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert_eq!(upstream.requests(), 2);
    assert_eq!(
        result.reasoning, "Step one. Step two. ",
        "reasoning was replayed to the client"
    );
    assert_eq!(result.text, "Answer.");
    assert_eq!(result.done_count, 1);
}

/// §7 give-up: reasoning reaches the client without touching `accumulated_text`,
/// so "nothing was emitted" must mean no payload at all. Otherwise an error
/// chunk is appended to a response the user can already see.
#[tokio::test]
async fn test_give_up_after_reasoning_only_does_not_append_an_error() {
    let upstream = spawn_raw_upstream(|_req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            let mut delta = serde_json::Map::new();
            delta.insert("role".into(), json!("assistant"));
            delta.insert("reasoning_content".into(), json!("Thinking hard. "));
            write_sse(
                &mut sock,
                &json!({
                    "id": "upstream-raw",
                    "object": "chat.completion.chunk",
                    "created": 1690000000,
                    "model": "raw-mock-model",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": Value::Null}]
                }),
            )
            .await;
            truncate_mid_chunk(sock).await;
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.resume.max_attempts = 2;
        config.resume.max_no_progress_attempts = 0;
    })
    .await;
    let result = collect(&test_client(), proxy, chat_body()).await;

    assert!(
        !result.reasoning.is_empty(),
        "reasoning did reach the client"
    );
    assert!(
        result.errors.is_empty(),
        "an error must not be appended after real output: {:?}",
        result.errors
    );
    assert_eq!(result.finish_reasons, vec!["stop".to_string()]);
    assert_eq!(result.done_count, 1);
}

/// §4: admission control refuses rather than queues once the limit is reached.
#[tokio::test]
async fn test_concurrency_limit_refuses_excess_requests() {
    let upstream = spawn_raw_upstream(|_req, mut sock| {
        Box::pin(async move {
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;
            // Hold the stream open so the first request keeps its permit.
            tokio::time::sleep(Duration::from_secs(3)).await;
            write_sse(&mut sock, &finish_chunk("stop")).await;
            write_done(&mut sock).await;
            end_stream_cleanly(&mut sock).await;
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.server.max_concurrent_requests = 1;
        config.resume.first_chunk_timeout_secs = 30;
        config.resume.idle_timeout_secs = 30;
    })
    .await;

    let client = test_client();
    let first = client
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .expect("first request should be admitted");
    assert_eq!(first.status(), reqwest::StatusCode::OK);

    let second = client
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .expect("second request should get a response, not hang");

    assert_eq!(
        second.status(),
        reqwest::StatusCode::SERVICE_UNAVAILABLE,
        "the second request should be refused while the first holds the permit"
    );
    let body: Value = second.json().await.unwrap();
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("concurrency limit"));

    // The permit is released when the first request finishes.
    drop(first);
    tokio::time::sleep(Duration::from_secs(4)).await;
    let third = client
        .post(format!("http://{proxy}/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .unwrap();
    assert_eq!(
        third.status(),
        reqwest::StatusCode::OK,
        "the permit should be released once the request completes"
    );
}

/// §11 / M2: a client that stays connected but stops reading fills the socket
/// buffers and then the channel, parking `send` forever. The upstream deadlines
/// wrap the *upstream* waits, so without bounding the send too, the task and its
/// upstream connection are retained indefinitely.
#[tokio::test]
async fn test_client_that_stops_reading_is_abandoned_at_the_deadline() {
    let closed_after_ms = Arc::new(AtomicU64::new(NEVER_CLOSED));
    let recorder = Arc::clone(&closed_after_ms);

    let upstream = spawn_raw_upstream(move |_req, mut sock| {
        let recorder = Arc::clone(&recorder);
        Box::pin(async move {
            let started = Instant::now();
            let _ = sock.write_all(SSE_HEAD.as_bytes()).await;

            // Push hard enough to fill the downstream socket buffers and then
            // the proxy's channel.
            for i in 0..100_000 {
                let payload = format!(
                    "data: {}\n\n",
                    content_chunk(&format!("chunk {i} padding padding padding "), i == 0)
                );
                let framed = format!("{:x}\r\n{}\r\n", payload.len(), payload);
                if sock.write_all(framed.as_bytes()).await.is_err() {
                    recorder.store(started.elapsed().as_millis() as u64, Ordering::SeqCst);
                    return;
                }
                if started.elapsed() > Duration::from_secs(20) {
                    return;
                }
            }
        })
    })
    .await;

    let proxy = spawn_proxy(upstream.base_url(), |config| {
        config.resume.max_wall_clock_secs = 2;
        config.resume.first_chunk_timeout_secs = 20;
        config.resume.idle_timeout_secs = 20;
        config.resume.max_attempts = 1;
        config.resume.max_accumulated_bytes = usize::MAX;
    })
    .await;

    // A raw client: send the request, read the response head, then stop reading
    // while keeping the connection open.
    let mut client = TcpStream::connect(proxy).await.unwrap();
    let body = chat_body().to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {proxy}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    client.write_all(request.as_bytes()).await.unwrap();
    let mut head = [0u8; 256];
    let _ = client.read(&mut head).await.unwrap();

    // Hold the socket open, reading nothing.
    tokio::time::sleep(Duration::from_secs(6)).await;
    drop(client);

    let closed = closed_after_ms.load(Ordering::SeqCst);
    assert_ne!(
        closed, NEVER_CLOSED,
        "the upstream connection was retained by a client that stopped reading"
    );
    assert!(
        closed < 5_000,
        "abandonment took {closed}ms; the send should be bounded by the wall clock"
    );
}
