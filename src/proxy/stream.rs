use bytes::Bytes;
use rand::Rng;
use reqwest::{Client, StatusCode};
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::Sender;
use tracing::{error, info, info_span, trace, warn};
use uuid::Uuid;

use crate::config::{build_upstream_url, Config};
use crate::error::{classify_http_response, ApiError, AttemptOutcome, Cause};
use crate::proxy::chunks::{
    build_chunk_with_finish, format_sse_comment, format_sse_data, format_sse_done, UsageAccumulator,
};
use crate::proxy::consume::{consume_upstream_stream, StreamConsumerState, StreamTiming};
use crate::proxy::continuation::build_continuation_request;
use crate::proxy::read_body_limited;

pub struct ResponseState {
    pub downstream_id: String,
    pub created: i64,
    pub model: String,
    pub model_frozen: bool,
    pub accumulated_text: String,
    pub emitted_role: bool,
    pub attempt: u32,
    pub started_at: Instant,
    pub original_body: Value,
    pub finish_reason: Option<String>,
    pub usage: UsageAccumulator,
}

impl ResponseState {
    pub fn new(original_body: Value) -> Self {
        let downstream_id = format!("chatcmpl-{}", Uuid::new_v4());
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let model = original_body
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown")
            .to_string();

        Self {
            downstream_id,
            created,
            model,
            model_frozen: false,
            accumulated_text: String::new(),
            emitted_role: false,
            attempt: 1,
            started_at: Instant::now(),
            original_body,
            finish_reason: None,
            usage: UsageAccumulator::default(),
        }
    }
}

/// Why attempts are failing, tracked across one downstream request.
///
/// Separates an upstream that is *busy* from one that is failing the same way
/// forever. The former deserves patience (§7.3); the latter is not going to fix
/// itself, and retrying it only converts a fast, clear failure into a slow one
/// with the same outcome (§7.5).
#[derive(Default)]
struct FailureTracker {
    congestion: u32,
    other: u32,
    last_label: Option<String>,
    consecutive_stalled: u32,
    last_failure: Option<ApiError>,
}

impl FailureTracker {
    /// Record one failed attempt. `made_progress` is whether this attempt
    /// emitted any new characters downstream.
    fn record(&mut self, label: &str, is_congestion: bool, made_progress: bool, err: ApiError) {
        if is_congestion {
            self.congestion += 1;
        } else {
            self.other += 1;
        }

        // Congestion is transparent to the stall detector: it neither counts nor
        // clears. "Busy, come back later" cannot emit content by definition, so
        // counting it would cut off exactly the case §7.3 exists to be patient
        // with — but it is also no evidence that the *other* failure has changed,
        // so it must not reset a run either. A degraded upstream interleaving
        // 503s with an identical fault would otherwise never reach the limit.
        if !is_congestion {
            if made_progress {
                self.consecutive_stalled = 0;
            } else if self.last_label.as_deref() == Some(label) {
                self.consecutive_stalled += 1;
            } else {
                self.consecutive_stalled = 1;
            }
            self.last_label = Some(label.to_string());
        }

        self.last_failure = Some(err);
    }

    fn congestion_only(&self) -> bool {
        self.congestion > 0 && self.other == 0
    }

    /// True once the same cause has failed `limit` times running without
    /// emitting anything new. `limit == 0` disables the check.
    fn is_stalled(&self, limit: u32) -> bool {
        limit > 0 && self.consecutive_stalled >= limit
    }
}

pub async fn handle_streaming_request(
    client: Client,
    config: Arc<Config>,
    original_body: Value,
    downstream_tx: Sender<Bytes>,
) {
    let mut state = ResponseState::new(original_body);
    let _span = info_span!("chat_completions", downstream_id = %state.downstream_id);

    // Drives both the deadline (§7.3) and the no-progress cutoff (§7.5), and
    // holds the error surfaced at give-up if nothing was ever emitted (§7).
    let mut failures = FailureTracker::default();
    // Assigned at the top of every attempt, before any use.
    let mut max_wall_clock;
    let upstream_endpoint = build_upstream_url(&config.upstream.base_url, "chat/completions");
    let api_key = config.get_upstream_api_key();

    let has_tools = state
        .original_body
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    let disable_resume_for_tools = config.tools.disable_resume_for_tool_responses && has_tools;
    let mut attempt_had_meaningful_chunk: bool;

    let model_req = state
        .original_body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    let msg_count = state
        .original_body
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    info!(
        downstream_id = %state.downstream_id,
        model = %model_req,
        stream = true,
        message_count = msg_count,
        "Streaming request started"
    );

    loop {
        max_wall_clock = effective_wall_clock(config.as_ref(), failures.congestion_only());

        let stalled = failures.is_stalled(config.resume.max_no_progress_attempts);
        let give_up_reason = if stalled {
            Some("the same failure repeated with no progress")
        } else if state.attempt > config.resume.max_attempts {
            Some("max attempts exhausted")
        } else if state.started_at.elapsed() >= max_wall_clock {
            Some("wall clock ceiling exhausted")
        } else {
            None
        };

        if let Some(reason) = give_up_reason {
            let emitted_chars = state.accumulated_text.chars().count();
            error!(
                downstream_id = %state.downstream_id,
                attempts = state.attempt - 1,
                elapsed_secs = state.started_at.elapsed().as_secs(),
                emitted_chars,
                repeated_cause = failures.last_label.as_deref().unwrap_or("none"),
                reason,
                "Proxy gave up"
            );

            // §7: a clean `stop` protects a partial answer from being corrupted
            // by error text. With nothing emitted there is nothing to protect,
            // and a silent empty response tells the user nothing at all — so the
            // real upstream error goes out instead.
            match (emitted_chars, failures.last_failure.take()) {
                (0, Some(api_err)) => {
                    let err_chunk = api_err.to_openai_json();
                    let _ = downstream_tx.send(format_sse_data(&err_chunk)).await;
                }
                _ => {
                    let finish_chunk = build_chunk_with_finish(
                        &state.downstream_id,
                        state.created,
                        &state.model,
                        "stop",
                        Some(&state.usage),
                    );
                    let _ = downstream_tx.send(format_sse_data(&finish_chunk)).await;
                }
            }
            let _ = downstream_tx.send(format_sse_done()).await;
            break;
        }

        info!(
            downstream_id = %state.downstream_id,
            attempt = state.attempt,
            "Starting upstream attempt"
        );

        let chars_before_attempt = state.accumulated_text.chars().count();

        let request_body = build_continuation_request(
            &state.original_body,
            &state.accumulated_text,
            config.resume.mode,
        );

        if config.log.log_bodies {
            trace!(
                downstream_id = %state.downstream_id,
                body = %serde_json::to_string(&request_body).unwrap_or_default(),
                "Upstream request body"
            );
        }

        let mut req_builder = client.post(&upstream_endpoint).json(&request_body);
        if let Some(ref key) = api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {key}"));
        }

        let attempt_started_at = Instant::now();
        let dispatch_timeout = Duration::from_secs(config.resume.first_chunk_timeout_secs)
            .min(max_wall_clock.saturating_sub(state.started_at.elapsed()));
        let upstream_res = tokio::select! {
            result = tokio::time::timeout(dispatch_timeout, req_builder.send()) => {
                match result {
                    Ok(response) => Ok(response),
                    Err(_) => Err(Cause::FirstChunkTimeout(Duration::from_secs(
                        config.resume.first_chunk_timeout_secs,
                    ))),
                }
            }
            _ = downstream_tx.closed() => {
                info!(downstream_id = %state.downstream_id, "Client disconnected while opening upstream request");
                return;
            }
        };

        let response = match upstream_res {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                let cause = Cause::Transport(e.to_string());
                failures.record(
                    &cause.label(),
                    cause.is_congestion(),
                    false,
                    ApiError::new(StatusCode::BAD_GATEWAY, e.to_string()),
                );
                warn!(
                    downstream_id = %state.downstream_id,
                    attempt = state.attempt,
                    error = %e,
                    "Failed to connect to upstream"
                );
                let backoff_dur = calculate_backoff(&config, state.attempt, cause.retry_after())
                    .min(max_wall_clock.saturating_sub(state.started_at.elapsed()));
                info!(
                    downstream_id = %state.downstream_id,
                    attempt = state.attempt,
                    backoff_ms = backoff_dur.as_millis(),
                    "Retrying after backoff"
                );
                if another_attempt_permitted(
                    &config,
                    state.attempt + 1,
                    state.started_at,
                    &failures,
                ) && !sleep_with_keepalive(backoff_dur, &downstream_tx).await
                {
                    info!(downstream_id = %state.downstream_id, "Client disconnected during retry backoff");
                    return;
                }
                state.attempt += 1;
                continue;
            }
            Err(cause) => {
                failures.record(
                    &cause.label(),
                    cause.is_congestion(),
                    false,
                    ApiError::new(StatusCode::GATEWAY_TIMEOUT, cause.to_string()),
                );
                warn!(
                    downstream_id = %state.downstream_id,
                    attempt = state.attempt,
                    cause = %cause,
                    "Upstream request exceeded first-chunk deadline"
                );
                let backoff_dur = calculate_backoff(&config, state.attempt, cause.retry_after())
                    .min(max_wall_clock.saturating_sub(state.started_at.elapsed()));
                if another_attempt_permitted(
                    &config,
                    state.attempt + 1,
                    state.started_at,
                    &failures,
                ) && !sleep_with_keepalive(backoff_dur, &downstream_tx).await
                {
                    info!(downstream_id = %state.downstream_id, "Client disconnected during retry backoff");
                    return;
                }
                state.attempt += 1;
                continue;
            }
        };

        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body_result = tokio::select! {
                result = tokio::time::timeout(
                    max_wall_clock.saturating_sub(state.started_at.elapsed()),
                    read_body_limited(response, config.upstream.max_response_bytes),
                ) => result,
                _ = downstream_tx.closed() => {
                    info!(downstream_id = %state.downstream_id, "Client disconnected while reading upstream error");
                    return;
                }
            };
            let body_text = match body_result {
                Ok(Ok(bytes)) => String::from_utf8_lossy(&bytes).into_owned(),
                Ok(Err(e)) => {
                    failures.record(
                        "error-body-read",
                        false,
                        false,
                        ApiError::new(status, e.to_string()),
                    );
                    warn!(
                        downstream_id = %state.downstream_id,
                        attempt = state.attempt,
                        error = %e,
                        "Failed to read upstream error response"
                    );
                    let backoff_dur = calculate_backoff(&config, state.attempt, None)
                        .min(max_wall_clock.saturating_sub(state.started_at.elapsed()));
                    if another_attempt_permitted(
                        &config,
                        state.attempt + 1,
                        state.started_at,
                        &failures,
                    ) && !sleep_with_keepalive(backoff_dur, &downstream_tx).await
                    {
                        info!(downstream_id = %state.downstream_id, "Client disconnected during retry backoff");
                        return;
                    }
                    state.attempt += 1;
                    continue;
                }
                Err(_) => {
                    failures.record(
                        "error-body-timeout",
                        false,
                        false,
                        ApiError::new(
                            StatusCode::GATEWAY_TIMEOUT,
                            "Upstream error response exceeded wall-clock limit",
                        ),
                    );
                    warn!(
                        downstream_id = %state.downstream_id,
                        attempt = state.attempt,
                        "Upstream error response exceeded wall-clock limit"
                    );
                    let backoff_dur = calculate_backoff(&config, state.attempt, None)
                        .min(max_wall_clock.saturating_sub(state.started_at.elapsed()));
                    if another_attempt_permitted(
                        &config,
                        state.attempt + 1,
                        state.started_at,
                        &failures,
                    ) && !sleep_with_keepalive(backoff_dur, &downstream_tx).await
                    {
                        info!(downstream_id = %state.downstream_id, "Client disconnected during retry backoff");
                        return;
                    }
                    state.attempt += 1;
                    continue;
                }
            };
            if config.log.log_bodies {
                trace!(
                    downstream_id = %state.downstream_id,
                    status = %status,
                    body = %body_text,
                    "Upstream error response body"
                );
            }
            let outcome = classify_http_response(status, &headers, &body_text);

            match outcome {
                AttemptOutcome::Retryable(cause) => {
                    failures.record(
                        &cause.label(),
                        cause.is_congestion(),
                        false,
                        ApiError::from_status_and_body(status, &body_text),
                    );
                    warn!(
                        downstream_id = %state.downstream_id,
                        attempt = state.attempt,
                        status = %status,
                        congestion = cause.is_congestion(),
                        "Upstream returned retryable HTTP status"
                    );
                    let backoff_dur =
                        calculate_backoff(&config, state.attempt, cause.retry_after())
                            .min(max_wall_clock.saturating_sub(state.started_at.elapsed()));
                    info!(
                        downstream_id = %state.downstream_id,
                        attempt = state.attempt,
                        backoff_ms = backoff_dur.as_millis(),
                        "Retrying after backoff"
                    );
                    if another_attempt_permitted(
                        &config,
                        state.attempt + 1,
                        state.started_at,
                        &failures,
                    ) && !sleep_with_keepalive(backoff_dur, &downstream_tx).await
                    {
                        info!(downstream_id = %state.downstream_id, "Client disconnected during retry backoff");
                        return;
                    }
                    state.attempt += 1;
                    continue;
                }
                AttemptOutcome::Fatal(api_err) => {
                    error!(
                        downstream_id = %state.downstream_id,
                        status = %status,
                        error = %api_err.message,
                        "Upstream returned fatal error; aborting request"
                    );
                    let err_chunk = api_err.to_openai_json();
                    let _ = downstream_tx.send(format_sse_data(&err_chunk)).await;
                    let _ = downstream_tx.send(format_sse_done()).await;
                    break;
                }
                AttemptOutcome::Terminal(finish_reason) => {
                    let finish_chunk = build_chunk_with_finish(
                        &state.downstream_id,
                        state.created,
                        &state.model,
                        &finish_reason,
                        Some(&state.usage),
                    );
                    let _ = downstream_tx.send(format_sse_data(&finish_chunk)).await;
                    let _ = downstream_tx.send(format_sse_done()).await;
                    break;
                }
            }
        }

        // Successfully opened 200 OK upstream stream
        attempt_had_meaningful_chunk = false;
        let mut consumer_state = StreamConsumerState {
            downstream_id: &state.downstream_id,
            created: state.created,
            model: &mut state.model,
            model_frozen: &mut state.model_frozen,
            accumulated_text: &mut state.accumulated_text,
            emitted_role: &mut state.emitted_role,
            usage: &mut state.usage,
            finish_reason: &mut state.finish_reason,
            attempt_had_meaningful_chunk: &mut attempt_had_meaningful_chunk,
        };

        let outcome = match consume_upstream_stream(
            response,
            &config,
            &mut consumer_state,
            state.attempt,
            StreamTiming {
                attempt_started_at,
                request_started_at: state.started_at,
                max_wall_clock,
            },
            &downstream_tx,
        )
        .await
        {
            Ok(out) => out,
            Err(_disconnected) => {
                // Downstream client disconnected
                info!(downstream_id = %state.downstream_id, "Client disconnected; aborted upstream request");
                return;
            }
        };

        match outcome {
            AttemptOutcome::Terminal(finish_reason) => {
                let finish_chunk = build_chunk_with_finish(
                    &state.downstream_id,
                    state.created,
                    &state.model,
                    &finish_reason,
                    Some(&state.usage),
                );
                let _ = downstream_tx.send(format_sse_data(&finish_chunk)).await;
                let _ = downstream_tx.send(format_sse_done()).await;

                info!(
                    downstream_id = %state.downstream_id,
                    attempts = state.attempt,
                    total_duration_secs = state.started_at.elapsed().as_secs(),
                    emitted_chars = state.accumulated_text.chars().count(),
                    finish_reason = %finish_reason,
                    "Streaming request completed successfully"
                );
                break;
            }
            AttemptOutcome::Retryable(cause) => {
                failures.record(
                    &cause.label(),
                    cause.is_congestion(),
                    state.accumulated_text.chars().count() > chars_before_attempt,
                    ApiError::new(StatusCode::BAD_GATEWAY, cause.to_string()),
                );
                if ((!config.resume.enabled && attempt_had_meaningful_chunk)
                    || (disable_resume_for_tools && !state.accumulated_text.is_empty()))
                    || state.started_at.elapsed() >= max_wall_clock
                {
                    info!(
                        downstream_id = %state.downstream_id,
                        resume_enabled = config.resume.enabled,
                        disable_resume_for_tools,
                        "Resumption disabled for this request; finishing stream"
                    );
                    let finish_chunk = build_chunk_with_finish(
                        &state.downstream_id,
                        state.created,
                        &state.model,
                        "stop",
                        Some(&state.usage),
                    );
                    let _ = downstream_tx.send(format_sse_data(&finish_chunk)).await;
                    let _ = downstream_tx.send(format_sse_done()).await;
                    break;
                }

                warn!(
                    downstream_id = %state.downstream_id,
                    attempt = state.attempt,
                    cause = %cause,
                    "Attempt failed with retryable cause; preparing to resume"
                );

                let backoff_dur = calculate_backoff(&config, state.attempt, cause.retry_after())
                    .min(max_wall_clock.saturating_sub(state.started_at.elapsed()));
                info!(
                    downstream_id = %state.downstream_id,
                    attempt = state.attempt,
                    backoff_ms = backoff_dur.as_millis(),
                    "Backing off before resumption"
                );

                if another_attempt_permitted(
                    &config,
                    state.attempt + 1,
                    state.started_at,
                    &failures,
                ) && !sleep_with_keepalive(backoff_dur, &downstream_tx).await
                {
                    info!(downstream_id = %state.downstream_id, "Client disconnected during retry backoff");
                    return;
                }
                state.attempt += 1;
                continue;
            }
            AttemptOutcome::Fatal(api_err) => {
                error!(
                    downstream_id = %state.downstream_id,
                    error = %api_err.message,
                    "Fatal error during stream; terminating"
                );
                let err_chunk = api_err.to_openai_json();
                let _ = downstream_tx.send(format_sse_data(&err_chunk)).await;
                let _ = downstream_tx.send(format_sse_done()).await;
                break;
            }
        }
    }
}

/// Delay before the next upstream attempt.
///
/// `Retry-After` is treated as a *floor*, not as a replacement for the
/// exponential curve. The proxy never comes back sooner than the upstream asked
/// (§7), but it does not hold a constant cadence either: an upstream still
/// answering `Retry-After: 1` on the sixth attempt has plainly not recovered,
/// and obeying it literally would spend the whole `max_attempts` budget in a
/// handful of seconds while `max_wall_clock_secs` — the ceiling that actually
/// governs how long a response may take — sits almost entirely unused.
/// §7.3: how long the proxy may keep trying.
///
/// While every failure so far has been the upstream reporting congestion, the
/// longer `overloaded_max_wall_clock_secs` applies — that condition clears on
/// its own, and giving up early wastes a budget that was meant to absorb it.
/// The moment any other kind of failure appears, the normal ceiling returns.
/// Takes the larger of the two so the setting can only ever extend.
/// Whether another attempt is actually permitted after the current failure.
///
/// The give-up checks live at the top of the loop, so without this the last
/// failed attempt sleeps out a full backoff — up to `backoff_max_ms` — before
/// discovering it had no retry left. Observed costing 20s of dead waiting at the
/// end of an already-doomed request.
fn another_attempt_permitted(
    config: &Config,
    next_attempt: u32,
    started_at: Instant,
    failures: &FailureTracker,
) -> bool {
    next_attempt <= config.resume.max_attempts
        && !failures.is_stalled(config.resume.max_no_progress_attempts)
        && started_at.elapsed() < effective_wall_clock(config, failures.congestion_only())
}

pub fn effective_wall_clock(config: &Config, congestion_only: bool) -> Duration {
    let normal = Duration::from_secs(config.resume.max_wall_clock_secs);
    if congestion_only {
        normal.max(Duration::from_secs(
            config.resume.overloaded_max_wall_clock_secs,
        ))
    } else {
        normal
    }
}

pub fn calculate_backoff(config: &Config, attempt: u32, retry_after: Option<Duration>) -> Duration {
    let max_dur = Duration::from_millis(config.resume.backoff_max_ms);

    let initial = config.resume.backoff_initial_ms as f64;
    let multiplier = config.resume.backoff_multiplier;
    let max_ms = config.resume.backoff_max_ms as f64;

    let exponent = (attempt.saturating_sub(1)) as i32;
    let base_ms = (initial * multiplier.powi(exponent)).min(max_ms);

    let jitter = config.resume.backoff_jitter;
    let factor = if jitter > 0.0 {
        let mut rng = rand::thread_rng();
        1.0 + rng.gen_range(-jitter..=jitter)
    } else {
        1.0
    };

    let exponential = Duration::from_millis((base_ms * factor).clamp(0.0, max_ms) as u64);

    match retry_after {
        Some(hint) => exponential.max(hint).min(max_dur),
        None => exponential,
    }
}

async fn sleep_with_keepalive(duration: Duration, downstream_tx: &Sender<Bytes>) -> bool {
    let sleep_start = Instant::now();
    let mut interval = tokio::time::interval(Duration::from_secs(10));
    interval.tick().await; // consume first immediate tick

    loop {
        let elapsed = sleep_start.elapsed();
        if elapsed >= duration {
            break;
        }
        let remaining = duration - elapsed;
        tokio::select! {
            _ = tokio::time::sleep(remaining) => {
                break;
            }
            _ = interval.tick() => {
                if downstream_tx.send(format_sse_comment("retrying")).await.is_err() {
                    return false;
                }
            }
            _ = downstream_tx.closed() => {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err() -> ApiError {
        ApiError::new(StatusCode::BAD_GATEWAY, "boom")
    }

    #[test]
    fn test_failure_tracker_counts_repeats_without_progress() {
        let mut t = FailureTracker::default();
        for expected in 1..=3 {
            t.record("malformed-function-call", false, false, err());
            assert_eq!(t.consecutive_stalled, expected);
        }
        assert!(t.is_stalled(3));
        assert!(!t.is_stalled(0), "limit 0 disables the check");
    }

    #[test]
    fn test_failure_tracker_resets_on_progress_or_a_different_cause() {
        let mut t = FailureTracker::default();
        t.record("stream-ended", false, false, err());
        t.record("stream-ended", false, false, err());
        assert_eq!(t.consecutive_stalled, 2);

        // An attempt that emitted new text is progress, however it ended.
        t.record("stream-ended", false, true, err());
        assert_eq!(t.consecutive_stalled, 0);
        assert!(!t.is_stalled(3));

        // A different failure starts a new run rather than continuing the old.
        t.record("idle-timeout", false, false, err());
        t.record("stream-ended", false, false, err());
        assert_eq!(t.consecutive_stalled, 1);
    }

    /// Congestion cannot make content progress by definition, so counting it
    /// would cut off exactly the case §7.3 exists to be patient with.
    #[test]
    fn test_failure_tracker_never_stalls_on_congestion() {
        let mut t = FailureTracker::default();
        for _ in 0..20 {
            t.record("http-503", true, false, err());
        }
        assert_eq!(t.consecutive_stalled, 0);
        assert!(!t.is_stalled(3));
        assert!(t.congestion_only());

        // ...but congestion must not *clear* a run either: it says nothing about
        // whether the other failure has changed. This is the production sequence
        // observed against Gemini — 503s interleaved with an identical fault.
        let mut t = FailureTracker::default();
        for (label, congestion) in [
            ("http-503", true),
            ("malformed-function-call", false),
            ("malformed-function-call", false),
            ("http-503", true),
            ("http-503", true),
            ("http-503", true),
        ] {
            t.record(label, congestion, false, err());
            assert!(!t.is_stalled(3));
        }
        t.record("malformed-function-call", false, false, err());
        assert!(
            t.is_stalled(3),
            "interleaved congestion must not hide a repeating fault"
        );

        // One non-congestion failure ends the extended deadline.
        t.record("stream-ended", false, false, err());
        assert!(!t.congestion_only());
    }

    /// §7.3: the extended deadline applies only while congestion is the *only*
    /// thing that has gone wrong, and it can never shorten the normal ceiling.
    #[test]
    fn test_effective_wall_clock_extends_only_for_congestion() {
        let mut config = Config::default();
        config.resume.max_wall_clock_secs = 240;
        config.resume.overloaded_max_wall_clock_secs = 300;

        assert_eq!(
            effective_wall_clock(&config, true),
            Duration::from_secs(300),
            "congestion-only should use the extended ceiling"
        );
        assert_eq!(
            effective_wall_clock(&config, false),
            Duration::from_secs(240),
            "any other failure returns the normal ceiling"
        );

        // Never shortens: a lower overloaded value is simply ignored.
        config.resume.overloaded_max_wall_clock_secs = 60;
        assert_eq!(
            effective_wall_clock(&config, true),
            Duration::from_secs(240)
        );

        // Shipped defaults: 900 normal already exceeds 300, so nothing changes.
        let defaults = Config::default();
        assert_eq!(
            effective_wall_clock(&defaults, true),
            effective_wall_clock(&defaults, false)
        );
    }

    #[test]
    fn test_calculate_backoff_no_jitter() {
        let mut config = Config::default();
        config.resume.backoff_initial_ms = 500;
        config.resume.backoff_multiplier = 2.0;
        config.resume.backoff_max_ms = 5000;
        config.resume.backoff_jitter = 0.0;

        assert_eq!(
            calculate_backoff(&config, 1, None),
            Duration::from_millis(500)
        );
        assert_eq!(
            calculate_backoff(&config, 2, None),
            Duration::from_millis(1000)
        );
        assert_eq!(
            calculate_backoff(&config, 3, None),
            Duration::from_millis(2000)
        );
        assert_eq!(
            calculate_backoff(&config, 4, None),
            Duration::from_millis(4000)
        );
        // Capped at max 5000ms
        assert_eq!(
            calculate_backoff(&config, 5, None),
            Duration::from_millis(5000)
        );
    }

    /// `Retry-After` raises the delay when it is longer than the exponential
    /// value, and is superseded once the curve overtakes it — so an upstream
    /// repeating `Retry-After: 1` gets progressively more breathing room instead
    /// of six rapid-fire attempts inside six seconds.
    #[test]
    fn test_retry_after_is_a_floor_not_a_replacement() {
        let mut config = Config::default();
        config.resume.backoff_initial_ms = 500;
        config.resume.backoff_multiplier = 2.0;
        config.resume.backoff_max_ms = 30_000;
        config.resume.backoff_jitter = 0.0;

        // The hint dominates a shorter exponential value.
        assert_eq!(
            calculate_backoff(&config, 1, Some(Duration::from_secs(3))),
            Duration::from_secs(3)
        );

        // A constant one-second hint across six attempts still escalates.
        for (i, want_ms) in [1_000u64, 1_000, 2_000, 4_000, 8_000, 16_000]
            .into_iter()
            .enumerate()
        {
            let attempt = i as u32 + 1;
            assert_eq!(
                calculate_backoff(&config, attempt, Some(Duration::from_secs(1))),
                Duration::from_millis(want_ms),
                "attempt {attempt}"
            );
        }
    }

    #[test]
    fn test_retry_after_is_clamped_to_backoff_max() {
        let mut config = Config::default();
        config.resume.backoff_max_ms = 10_000;

        assert_eq!(
            calculate_backoff(&config, 1, Some(Duration::from_secs(20))),
            Duration::from_millis(10_000)
        );
    }

    /// An HTTP-date already in the past parses as zero, which must not
    /// short-circuit the backoff into an immediate retry.
    #[test]
    fn test_retry_after_in_the_past_falls_back_to_exponential() {
        let mut config = Config::default();
        config.resume.backoff_initial_ms = 500;
        config.resume.backoff_multiplier = 2.0;
        config.resume.backoff_max_ms = 30_000;
        config.resume.backoff_jitter = 0.0;

        assert_eq!(
            calculate_backoff(&config, 3, Some(Duration::ZERO)),
            Duration::from_millis(2_000)
        );
    }

    #[test]
    fn test_calculate_backoff_with_jitter_bounds() {
        let mut config = Config::default();
        config.resume.backoff_initial_ms = 1000;
        config.resume.backoff_multiplier = 2.0;
        config.resume.backoff_jitter = 0.25;

        for _ in 0..50 {
            let dur = calculate_backoff(&config, 1, None);
            let ms = dur.as_millis();
            assert!(
                (750..=1250).contains(&ms),
                "Expected backoff with jitter in [750, 1250], got {ms}"
            );
        }
    }

    #[test]
    fn test_calculate_backoff_jitter_does_not_exceed_maximum() {
        let mut config = Config::default();
        config.resume.backoff_initial_ms = 1000;
        config.resume.backoff_max_ms = 1000;
        config.resume.backoff_jitter = 0.25;

        for _ in 0..50 {
            assert!(calculate_backoff(&config, 1, None) <= Duration::from_millis(1000));
        }
    }
}
