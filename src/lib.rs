pub mod config;
pub mod error;
pub mod mock;
pub mod proxy;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use tracing::{error, warn};

use crate::config::{build_upstream_url, Config};
use crate::error::{parse_retry_after, ApiError};
use crate::proxy::nonstream::handle_nonstreaming_request;
use crate::proxy::stream::{calculate_backoff, handle_streaming_request};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub client: Client,
}

pub fn create_app(config: Arc<Config>) -> Router {
    let connect_timeout = Duration::from_secs(config.upstream.connect_timeout_secs);
    let client = Client::builder()
        .connect_timeout(connect_timeout)
        .build()
        .expect("Failed to build HTTP client");

    let max_body_bytes = config.server.max_body_bytes;
    let state = AppState { config, client };

    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .fallback(not_found_handler)
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}

async fn not_found_handler() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": {
                "message": "Not found",
                "type": "invalid_request_error",
                "param": null,
                "code": null
            }
        })),
    )
}

/// §5: transparent proxy to the upstream models endpoint, with a single retry
/// on 5xx / 429 / transport error. The upstream body is passed through verbatim.
async fn models_handler(State(state): State<AppState>) -> Response {
    let upstream_endpoint = build_upstream_url(&state.config.upstream.base_url, "models");
    let api_key = state.config.get_upstream_api_key();

    // The shared client sets only a connect timeout, so an upstream that
    // completes the handshake and then stalls would hang this task forever.
    // Bound the whole handler — both attempts — by the first-chunk budget.
    let deadline =
        Instant::now() + Duration::from_secs(state.config.resume.first_chunk_timeout_secs);
    let remaining = || deadline.saturating_duration_since(Instant::now());

    let mut last_error: Option<(StatusCode, String)> = None;

    for attempt in 1..=2u32 {
        let mut req = state.client.get(&upstream_endpoint);
        if let Some(ref key) = api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let (status, content_type, retry_after, bytes) =
            match tokio::time::timeout(remaining(), req.send()).await {
                Ok(Ok(resp)) => {
                    let status = resp.status();
                    let content_type = resp
                        .headers()
                        .get(header::CONTENT_TYPE)
                        .cloned()
                        .unwrap_or_else(|| HeaderValue::from_static("application/json"));
                    let retry_after = parse_retry_after(resp.headers());
                    match tokio::time::timeout(remaining(), resp.bytes()).await {
                        Ok(Ok(bytes)) => (status, content_type, retry_after, bytes),
                        Ok(Err(e)) => {
                            last_error = Some((StatusCode::BAD_GATEWAY, e.to_string()));
                            warn!(error = %e, "Models endpoint body read failed");
                            if attempt == 1 && !remaining().is_zero() {
                                sleep_before_models_retry(&state, attempt, None, remaining()).await;
                                continue;
                            }
                            break;
                        }
                        Err(_) => {
                            last_error = Some((
                                StatusCode::GATEWAY_TIMEOUT,
                                "Upstream models response timed out".to_string(),
                            ));
                            warn!("Models endpoint body read timed out");
                            break;
                        }
                    }
                }
                Ok(Err(e)) => {
                    last_error = Some((StatusCode::BAD_GATEWAY, e.to_string()));
                    warn!(error = %e, "Models endpoint network error");
                    if attempt == 1 && !remaining().is_zero() {
                        sleep_before_models_retry(&state, attempt, None, remaining()).await;
                        continue;
                    }
                    break;
                }
                Err(_) => {
                    last_error = Some((
                        StatusCode::GATEWAY_TIMEOUT,
                        "Upstream models request timed out".to_string(),
                    ));
                    warn!("Models endpoint request timed out");
                    break;
                }
            };

        let retryable = status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS;
        if retryable && attempt == 1 && !remaining().is_zero() {
            warn!(status = %status, "Models endpoint request failed; retrying once");
            sleep_before_models_retry(&state, attempt, retry_after, remaining()).await;
            continue;
        }

        return Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(bytes))
            .unwrap()
            .into_response();
    }

    let (status, message) = last_error.unwrap_or((
        StatusCode::BAD_GATEWAY,
        "Failed to reach upstream models endpoint".to_string(),
    ));
    error!(status = %status, error = %message, "Models endpoint failed");
    (
        status,
        Json(ApiError::new(status, message).to_openai_json()),
    )
        .into_response()
}

/// Wait before the single models retry, honouring `Retry-After` on the same
/// terms as the chat path (§7.2) instead of a fixed cadence.
async fn sleep_before_models_retry(
    state: &AppState,
    attempt: u32,
    retry_after: Option<Duration>,
    remaining: Duration,
) {
    let backoff = calculate_backoff(&state.config, attempt, retry_after).min(remaining);
    tokio::time::sleep(backoff).await;
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Response {
    let is_streaming = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    if is_streaming {
        let (tx, rx) = mpsc::channel::<Bytes>(64);
        tokio::spawn(handle_streaming_request(
            state.client.clone(),
            state.config.clone(),
            body,
            tx,
        ));

        let stream = ReceiverStream::new(rx).map(Ok::<_, Infallible>);

        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .header("X-Accel-Buffering", "no")
            .body(Body::from_stream(stream))
            .unwrap_or_else(|e| {
                error!(error = %e, "Failed to construct streaming response");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "message": "Internal server error",
                            "type": "api_error",
                            "param": null,
                            "code": null
                        }
                    })),
                )
                    .into_response()
            })
    } else {
        handle_nonstreaming_request(state.client.clone(), state.config.clone(), body).await
    }
}
