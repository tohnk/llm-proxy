use std::convert::Infallible;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

#[derive(Debug, Deserialize, Clone, Default)]
pub struct MockParams {
    pub cut_after: Option<usize>,
    pub cut_first_n_requests: Option<u32>,
    pub stall_after: Option<usize>,
    pub stall_first_n_requests: Option<u32>,
    pub status: Option<u16>,
    pub fail_first: Option<u32>,
    pub repeat_overlap: Option<usize>,
    pub restart: Option<bool>,
    pub slow_first_chunk: Option<u64>,
    pub slow_first_n_requests: Option<u32>,
    pub slow_response: Option<u64>,
    pub slow_response_first_n_requests: Option<u32>,
    pub retry_after_secs: Option<u64>,
    pub retry_after_http_date_secs: Option<u64>,
    pub tool_call: Option<bool>,
    pub multi_tool_call: Option<bool>,
    pub reasoning: Option<bool>,
    pub reasoning_field: Option<String>,
    pub role_only: Option<bool>,
    pub finish_reason: Option<String>,
    pub finish_reason_first_n_requests: Option<u32>,
    pub include_usage: Option<bool>,
    pub error_message: Option<String>,
}

#[derive(Clone, Default)]
pub struct MockServerState {
    pub request_counter: Arc<AtomicU32>,
    pub client_disconnect_counter: Arc<AtomicU32>,
    pub last_auth_header: Arc<tokio::sync::Mutex<Option<String>>>,
    pub last_received_body: Arc<tokio::sync::Mutex<Option<Value>>>,
}

pub fn default_corpus() -> String {
    let mut out = String::new();
    for i in 1..=100 {
        out.push_str(&format!("Line {i}\n"));
    }
    out
}

pub fn create_mock_app(state: MockServerState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/chat/completions", post(chat_completions_handler))
        // The mock stands in for a real upstream, which does not impose axum's
        // 2 MiB default; without this it would reject large prompts before the
        // proxy's own limit is ever exercised.
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}

async fn models_handler(
    State(state): State<MockServerState>,
    Query(params): Query<MockParams>,
) -> Response {
    let req_num = state.request_counter.fetch_add(1, Ordering::SeqCst) + 1;
    if let (Some(fail_first), Some(status_code)) = (params.fail_first, params.status) {
        if req_num <= fail_first {
            let status =
                StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let err_body = serde_json::json!({
                "error": {
                    "message": format!("Mock models upstream simulated error {status_code}"),
                    "type": "api_error",
                    "code": status_code,
                    "param": null
                }
            });
            return (status, Json(err_body)).into_response();
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "object": "list",
            "data": [
                {
                    "id": "mock-model",
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "mock"
                }
            ]
        })),
    )
        .into_response()
}

async fn chat_completions_handler(
    State(state): State<MockServerState>,
    Query(params): Query<MockParams>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let req_num = state.request_counter.fetch_add(1, Ordering::SeqCst) + 1;
    info!(req_num, ?params, "Mock upstream received request");

    // Record auth header and body for assertions
    if let Some(auth) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        *state.last_auth_header.lock().await = Some(auth.to_string());
    }
    *state.last_received_body.lock().await = Some(body.clone());

    let should_slow_response = params
        .slow_response_first_n_requests
        .map(|n| req_num <= n)
        .unwrap_or(true);
    if let Some(delay) = params.slow_response {
        if should_slow_response {
            tokio::time::sleep(Duration::from_secs(delay)).await;
        }
    }

    // Check if we should fail this request based on fail_first & status
    if let (Some(fail_first), Some(status_code)) = (params.fail_first, params.status) {
        if req_num <= fail_first {
            let status =
                StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let mut resp_headers = HeaderMap::new();
            if let Some(retry_after) = params.retry_after_secs {
                resp_headers.insert(
                    header::RETRY_AFTER,
                    retry_after.to_string().parse().unwrap(),
                );
            } else if let Some(future_secs) = params.retry_after_http_date_secs {
                let future = std::time::SystemTime::now() + Duration::from_secs(future_secs);
                let date_str = httpdate::fmt_http_date(future);
                resp_headers.insert(header::RETRY_AFTER, date_str.parse().unwrap());
            }
            let msg = params
                .error_message
                .clone()
                .unwrap_or_else(|| format!("Mock upstream simulated error {status_code}"));
            let err_body = serde_json::json!({
                "error": {
                    "message": msg,
                    "type": if status.is_client_error() { "invalid_request_error" } else { "api_error" },
                    "code": status_code,
                    "param": null
                }
            });
            return (status, resp_headers, Json(err_body)).into_response();
        }
    }

    let is_streaming = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    if !is_streaming {
        // Non-streaming response
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1690000000,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Line 1\nLine 2\nLine 3\n"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            })),
        )
            .into_response();
    }

    // Streaming response
    let (tx, rx) = mpsc::channel::<String>(64);
    let disconnect_counter = state.client_disconnect_counter.clone();

    tokio::spawn(async move {
        let is_tool_call = params.tool_call.unwrap_or(false)
            || body
                .get("tools")
                .and_then(|t| t.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false);

        let should_slow = params
            .slow_first_n_requests
            .or(params.cut_first_n_requests)
            .map(|n| req_num <= n)
            .unwrap_or(true);

        if let Some(delay) = params.slow_first_chunk {
            if should_slow {
                tokio::time::sleep(Duration::from_secs(delay)).await;
            }
        }

        if is_tool_call {
            handle_mock_tool_call_stream(tx, params, disconnect_counter, req_num).await;
        } else {
            handle_mock_text_stream(tx, body, params, disconnect_counter, req_num).await;
        }
    });

    let stream = ReceiverStream::new(rx).map(Ok::<_, Infallible>);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}

async fn handle_mock_tool_call_stream(
    tx: mpsc::Sender<String>,
    params: MockParams,
    disconnect_counter: Arc<AtomicU32>,
    req_num: u32,
) {
    let chunk_id = format!("chatcmpl-mock-tc-attempt-{req_num}");
    let created = 1690000000 + req_num as i64;
    let chunks: Vec<Value> = if params.multi_tool_call.unwrap_or(false) {
        vec![
            serde_json::json!({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_stock_0",
                                "type": "function",
                                "function": {
                                    "name": "lookup_stock",
                                    "arguments": "{\"ticker\": \"AAPL\"}"
                                }
                            },
                            {
                                "index": 1,
                                "id": "call_weather_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_weather",
                                    "arguments": "{\"city\": \"Cupertino\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": Value::Null
                }]
            }),
            serde_json::json!({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls"
                }]
            }),
        ]
    } else {
        vec![
            serde_json::json!({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_mock_123",
                            "type": "function",
                            "function": {
                                "name": "lookup_stock",
                                "arguments": "{\"ticker\":"
                            }
                        }]
                    },
                    "finish_reason": Value::Null
                }]
            }),
            serde_json::json!({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": " \"AAPL\"}"
                            }
                        }]
                    },
                    "finish_reason": Value::Null
                }]
            }),
            serde_json::json!({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls"
                }]
            }),
        ]
    };

    let should_cut = params
        .cut_first_n_requests
        .map(|n| req_num <= n)
        .unwrap_or(true);

    for (i, chunk) in chunks.iter().enumerate() {
        if let Some(cut) = params.cut_after {
            if i >= cut && should_cut {
                warn!("Simulating stream cut for tool call chunk {i}");
                return;
            }
        }

        let formatted = format!("data: {}\n\n", serde_json::to_string(chunk).unwrap());
        if tx.send(formatted).await.is_err() {
            disconnect_counter.fetch_add(1, Ordering::SeqCst);
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let _ = tx.send("data: [DONE]\n\n".to_string()).await;
}

async fn handle_mock_text_stream(
    tx: mpsc::Sender<String>,
    body: Value,
    params: MockParams,
    disconnect_counter: Arc<AtomicU32>,
    req_num: u32,
) {
    let corpus = default_corpus();
    let chunk_id = format!("chatcmpl-mock-chunk-attempt-{req_num}");
    let created = 1690000000 + req_num as i64;

    // Determine start offset in corpus
    let mut offset = 0;
    let restart = params.restart.unwrap_or(false);

    if !restart {
        if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
            for msg in messages.iter().rev() {
                if msg.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        offset = content.chars().count();
                        if let Some(repeat) = params.repeat_overlap {
                            offset = offset.saturating_sub(repeat);
                        }
                        break;
                    }
                }
            }
        }
    }

    // If reasoning is requested and this is the first attempt (offset == 0), emit reasoning deltas
    if params.role_only.unwrap_or(false) && offset == 0 {
        let chunk = serde_json::json!({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": Value::Null
            }]
        });
        let formatted = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
        if tx.send(formatted).await.is_err() {
            disconnect_counter.fetch_add(1, Ordering::SeqCst);
            return;
        }
        tokio::time::sleep(Duration::from_millis(15)).await;
    }

    if params.reasoning.unwrap_or(false) && offset == 0 {
        let reasoning_chunks = vec![
            "Thinking step 1: analyze input... ",
            "Thinking step 2: formulate response.\n",
        ];
        let reasoning_field = params
            .reasoning_field
            .as_deref()
            .unwrap_or("reasoning_content");
        for rc in reasoning_chunks {
            let mut delta = serde_json::Map::new();
            delta.insert(reasoning_field.to_string(), Value::String(rc.to_string()));
            let chunk = serde_json::json!({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": Value::Null
                    }
                ]
            });
            let formatted = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
            if tx.send(formatted).await.is_err() {
                disconnect_counter.fetch_add(1, Ordering::SeqCst);
                return;
            }
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
    }

    let corpus_chars: Vec<char> = corpus.chars().collect();
    let remaining_chars = if offset < corpus_chars.len() {
        &corpus_chars[offset..]
    } else {
        &[]
    };

    // Break remaining chars into line or multi-char chunks
    let remaining_str: String = remaining_chars.iter().collect();
    let lines: Vec<&str> = remaining_str.split_inclusive('\n').collect();

    let should_cut = params
        .cut_first_n_requests
        .map(|n| req_num <= n)
        .unwrap_or(true);
    let should_stall = params
        .stall_first_n_requests
        .map(|n| req_num <= n)
        .unwrap_or(true);

    for (chunk_idx, line) in lines.iter().enumerate() {
        if let Some(cut) = params.cut_after {
            if chunk_idx >= cut && should_cut {
                if let Some(ref fr) = params.finish_reason {
                    let finish_chunk = serde_json::json!({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "mock-model",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": fr
                            }
                        ]
                    });
                    let _ = tx
                        .send(format!(
                            "data: {}\n\n",
                            serde_json::to_string(&finish_chunk).unwrap()
                        ))
                        .await;
                    let _ = tx.send("data: [DONE]\n\n".to_string()).await;
                }
                warn!(
                    chunk_idx,
                    cut, req_num, "Simulating connection cut mid-stream"
                );
                return;
            }
        }

        if let Some(stall) = params.stall_after {
            if chunk_idx >= stall && should_stall {
                warn!(chunk_idx, stall, req_num, "Simulating stream stall");
                tokio::time::sleep(Duration::from_secs(3600)).await;
                return;
            }
        }

        let chunk = serde_json::json!({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": line
                    },
                    "finish_reason": Value::Null
                }
            ]
        });

        let formatted = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
        if tx.send(formatted).await.is_err() {
            info!("Client disconnected from mock upstream");
            disconnect_counter.fetch_add(1, Ordering::SeqCst);
            return;
        }

        tokio::time::sleep(Duration::from_millis(15)).await;
    }

    // Terminal chunk
    let terminal_reason = if params
        .finish_reason_first_n_requests
        .map(|n| req_num <= n)
        .unwrap_or(false)
    {
        params.finish_reason.as_deref().unwrap_or("stop")
    } else {
        "stop"
    };
    let finish_chunk = serde_json::json!({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": terminal_reason
            }
        ]
    });
    let _ = tx
        .send(format!(
            "data: {}\n\n",
            serde_json::to_string(&finish_chunk).unwrap()
        ))
        .await;
    if params.include_usage.unwrap_or(false) {
        let usage_chunk = serde_json::json!({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "mock-model",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        });
        let _ = tx
            .send(format!(
                "data: {}\n\n",
                serde_json::to_string(&usage_chunk).unwrap()
            ))
            .await;
    }
    let _ = tx.send("data: [DONE]\n\n".to_string()).await;
}
