use axum::body::Body;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use reqwest::Client;
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{error, info, trace, warn};

use crate::config::{build_upstream_url, Config};
use crate::error::{classify_http_response, ApiError, AttemptOutcome, Cause};
use crate::proxy::stream::calculate_backoff;

pub async fn handle_nonstreaming_request(
    client: Client,
    config: Arc<Config>,
    original_body: Value,
) -> Response {
    let mut attempt = 1;
    let started_at = Instant::now();
    let max_wall_clock = Duration::from_secs(config.resume.max_wall_clock_secs);
    let upstream_endpoint = build_upstream_url(&config.upstream.base_url, "chat/completions");
    let api_key = config.get_upstream_api_key();

    let model_req = original_body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    let msg_count = original_body
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    info!(
        model = %model_req,
        stream = false,
        message_count = msg_count,
        "Non-streaming request started"
    );

    let mut last_error_status = StatusCode::BAD_GATEWAY;
    let mut last_error_body = serde_json::to_vec(&serde_json::json!({
        "error": {
            "message": "Upstream request failed",
            "type": "api_error",
            "param": null,
            "code": null
        }
    }))
    .unwrap_or_default();
    let mut last_error_content_type = HeaderValue::from_static("application/json");

    loop {
        if attempt > config.resume.max_attempts || started_at.elapsed() >= max_wall_clock {
            error!(
                attempts = attempt - 1,
                elapsed_secs = started_at.elapsed().as_secs(),
                "Non-streaming proxy gave up: max attempts or wall clock ceiling exhausted"
            );
            return Response::builder()
                .status(last_error_status)
                .header(header::CONTENT_TYPE, last_error_content_type)
                .body(Body::from(last_error_body))
                .unwrap()
                .into_response();
        }

        info!(attempt = attempt, "Starting non-streaming upstream attempt");

        if config.log.log_bodies {
            trace!(
                body = %serde_json::to_string(&original_body).unwrap_or_default(),
                "Upstream non-streaming request body"
            );
        }

        let mut req_builder = client.post(&upstream_endpoint).json(&original_body);
        if let Some(ref key) = api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {key}"));
        }

        let upstream_res = tokio::time::timeout(
            max_wall_clock.saturating_sub(started_at.elapsed()),
            req_builder.send(),
        )
        .await
        .map_err(|_| "non-streaming upstream request exceeded wall-clock limit".to_string())
        .and_then(|result| result.map_err(|e| e.to_string()));

        let response = match upstream_res {
            Ok(resp) => resp,
            Err(e) => {
                let cause = Cause::Transport(e.clone());
                warn!(attempt = attempt, error = %e, "Failed to connect to upstream for non-streaming request");
                last_error_status = StatusCode::BAD_GATEWAY;
                last_error_body = serde_json::to_vec(
                    &ApiError::new(StatusCode::BAD_GATEWAY, e.to_string()).to_openai_json(),
                )
                .unwrap_or_default();
                last_error_content_type = HeaderValue::from_static("application/json");

                let backoff_dur = calculate_backoff(&config, attempt, cause.retry_after())
                    .min(max_wall_clock.saturating_sub(started_at.elapsed()));
                tokio::time::sleep(backoff_dur).await;
                attempt += 1;
                continue;
            }
        };

        let status = response.status();
        let headers = response.headers().clone();
        let content_type = headers
            .get(header::CONTENT_TYPE)
            .cloned()
            .unwrap_or_else(|| HeaderValue::from_static("application/json"));
        let body_bytes = match tokio::time::timeout(
            max_wall_clock.saturating_sub(started_at.elapsed()),
            response.bytes(),
        )
        .await
        {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(e)) => {
                warn!(attempt = attempt, error = %e, "Failed to read upstream non-streaming response");
                last_error_status = StatusCode::BAD_GATEWAY;
                last_error_body = serde_json::to_vec(
                    &ApiError::new(StatusCode::BAD_GATEWAY, e.to_string()).to_openai_json(),
                )
                .unwrap_or_default();
                last_error_content_type = HeaderValue::from_static("application/json");

                let backoff_dur = calculate_backoff(&config, attempt, None)
                    .min(max_wall_clock.saturating_sub(started_at.elapsed()));
                tokio::time::sleep(backoff_dur).await;
                attempt += 1;
                continue;
            }
            Err(_) => {
                let message = "non-streaming upstream response exceeded wall-clock limit";
                warn!(
                    attempt = attempt,
                    error = message,
                    "Failed to read upstream non-streaming response"
                );
                last_error_status = StatusCode::BAD_GATEWAY;
                last_error_body = serde_json::to_vec(
                    &ApiError::new(StatusCode::BAD_GATEWAY, message).to_openai_json(),
                )
                .unwrap_or_default();
                last_error_content_type = HeaderValue::from_static("application/json");

                let backoff_dur = calculate_backoff(&config, attempt, None)
                    .min(max_wall_clock.saturating_sub(started_at.elapsed()));
                tokio::time::sleep(backoff_dur).await;
                attempt += 1;
                continue;
            }
        };
        let body_text = String::from_utf8_lossy(&body_bytes);

        if config.log.log_bodies {
            trace!(
                status = %status,
                body = %body_text,
                "Upstream non-streaming response body"
            );
        }

        if status.is_success() {
            let content_type = headers
                .get(header::CONTENT_TYPE)
                .cloned()
                .unwrap_or_else(|| HeaderValue::from_static("application/json"));

            info!(
                attempts = attempt,
                elapsed_secs = started_at.elapsed().as_secs(),
                "Non-streaming request completed successfully"
            );

            return Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, content_type)
                .body(Body::from(body_bytes))
                .unwrap()
                .into_response();
        }

        let outcome = classify_http_response(status, &headers, &body_text);
        last_error_status = status;
        last_error_body = body_bytes.to_vec();
        last_error_content_type = content_type;

        match outcome {
            AttemptOutcome::Retryable(cause) => {
                warn!(
                    attempt = attempt,
                    status = %status,
                    "Non-streaming attempt failed with retryable status"
                );
                let backoff_dur = calculate_backoff(&config, attempt, cause.retry_after())
                    .min(max_wall_clock.saturating_sub(started_at.elapsed()));
                tokio::time::sleep(backoff_dur).await;
                attempt += 1;
                continue;
            }
            AttemptOutcome::Fatal(_) | AttemptOutcome::Terminal(_) => {
                error!(
                    attempt = attempt,
                    status = %status,
                    "Non-streaming attempt failed fatally; returning error"
                );
                return Response::builder()
                    .status(status)
                    .header(header::CONTENT_TYPE, last_error_content_type)
                    .body(Body::from(last_error_body))
                    .unwrap()
                    .into_response();
            }
        }
    }
}
