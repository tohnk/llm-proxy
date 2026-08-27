use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::Response;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::Sender;
use tracing::{debug, trace, warn};

use crate::config::Config;
use crate::error::{
    is_context_or_policy_failure, is_failed_generation_finish_reason, normalize_finish_reason,
    ApiError, AttemptOutcome, Cause,
};
use crate::proxy::chunks::{
    build_chunk_with_content, build_chunk_with_reasoning, build_chunk_with_tool_calls,
    format_sse_data, parse_chunk_details, rewrite_upstream_chunk, UsageAccumulator,
};
use crate::proxy::dedup::DedupBuffer;
use crate::proxy::tools::ToolCallsBuffer;

/// Records when the upstream connection last delivered *bytes*, as opposed to
/// when it last produced a parsed SSE event.
///
/// SSE comment heartbeats (`: keep-alive`) are discarded by the event decoder
/// without dispatching an event, so an event-driven idle watchdog cannot tell a
/// deliberately kept-alive stream from a stalled one and would tear down a
/// healthy connection. §8 resets the idle timer on upstream activity; this is
/// how that activity is observed.
#[derive(Clone)]
struct ActivityClock {
    base: Instant,
    last_millis: Arc<AtomicU64>,
}

impl ActivityClock {
    fn new(base: Instant) -> Self {
        Self {
            base,
            last_millis: Arc::new(AtomicU64::new(base.elapsed().as_millis() as u64)),
        }
    }

    /// Note that bytes just arrived from upstream.
    fn mark(&self) {
        self.last_millis
            .store(self.base.elapsed().as_millis() as u64, Ordering::Relaxed);
    }

    fn last_activity(&self) -> Instant {
        self.base + Duration::from_millis(self.last_millis.load(Ordering::Relaxed))
    }
}

/// The downstream client went away while an attempt was in flight.
///
/// A named type rather than `()` so the one failure mode this function has is
/// legible at the call site — it is not an error to report upstream, it means
/// abandon the request entirely (§11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientDisconnected;

impl std::fmt::Display for ClientDisconnected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "downstream client disconnected")
    }
}

impl std::error::Error for ClientDisconnected {}

pub struct StreamConsumerState<'a> {
    pub downstream_id: &'a str,
    pub created: i64,
    pub model: &'a mut String,
    pub model_frozen: &'a mut bool,
    pub accumulated_text: &'a mut String,
    /// Reasoning emitted downstream so far. Deduplicated like content (§9) but
    /// deliberately excluded from continuation `messages`.
    pub accumulated_reasoning: &'a mut String,
    /// Count of payload frames sent downstream across all attempts. Content is
    /// not the only thing a client can have received — reasoning and passthrough
    /// tool deltas go out without touching `accumulated_text` — so decisions
    /// about whether anything reached the client key off this instead.
    pub emitted_payloads: &'a mut u64,
    pub emitted_role: &'a mut bool,
    pub usage: &'a mut UsageAccumulator,
    pub finish_reason: &'a mut Option<String>,
    pub attempt_had_meaningful_chunk: &'a mut bool,
}

/// Send one frame downstream, bounded by the request deadline.
///
/// A bounded channel plus a connected-but-unreading client is enough to defeat
/// every other deadline: the transport buffer and the channel fill, `send`
/// parks waiting for capacity, and the task holds its upstream connection open
/// indefinitely. The deadlines wrap the *upstream* waits, not this one.
async fn send_downstream(
    downstream_tx: &Sender<Bytes>,
    frame: Bytes,
    deadline: Instant,
) -> Result<(), ClientDisconnected> {
    tokio::select! {
        result = downstream_tx.send(frame) => result.map_err(|_| ClientDisconnected),
        _ = downstream_tx.closed() => Err(ClientDisconnected),
        _ = tokio::time::sleep_until(deadline.into()) => {
            warn!("Downstream client stopped reading; abandoning the request");
            Err(ClientDisconnected)
        }
    }
}

pub struct StreamTiming {
    pub attempt_started_at: std::time::Instant,
    pub request_started_at: std::time::Instant,
    pub max_wall_clock: Duration,
}

pub async fn consume_upstream_stream(
    response: Response,
    config: &Config,
    state: &mut StreamConsumerState<'_>,
    attempt: u32,
    timing: StreamTiming,
    downstream_tx: &Sender<Bytes>,
) -> Result<AttemptOutcome, ClientDisconnected> {
    // Every byte the upstream sends — heartbeat comments included — refreshes the
    // idle watchdog, even when it never becomes a dispatchable SSE event.
    let activity = ActivityClock::new(timing.attempt_started_at);
    let mut event_stream = {
        let activity = activity.clone();
        response
            .bytes_stream()
            .inspect(move |_| activity.mark())
            .eventsource()
    };
    let mut received_first_meaningful_chunk = false;
    let mut dedup_buffer = DedupBuffer::new(
        state.accumulated_text,
        config.resume.dedup_window_chars,
        attempt == 1,
    );
    let mut reasoning_dedup = DedupBuffer::new(
        state.accumulated_reasoning,
        config.resume.dedup_window_chars,
        attempt == 1,
    );
    let mut tool_buffer = ToolCallsBuffer::new(config.tools.max_tool_call_bytes);
    let mut last_finish_reason: Option<String> = None;
    let mut seen_done = false;

    let wall_clock_deadline = timing.request_started_at + timing.max_wall_clock;
    let first_timeout = Duration::from_secs(config.resume.first_chunk_timeout_secs);
    let idle_timeout = Duration::from_secs(config.resume.idle_timeout_secs);

    loop {
        // Wait for the next SSE event under whichever watchdog currently applies.
        // The first-chunk deadline is fixed from dispatch (§8) and heartbeats do
        // not extend it; the idle deadline tracks the last upstream activity, so
        // a heartbeat pushes it out and the sleep is simply re-armed.
        let next_item = 'watchdog: loop {
            let in_first_chunk_phase =
                !received_first_meaningful_chunk && last_finish_reason.is_none();
            let watchdog_deadline = if in_first_chunk_phase {
                timing.attempt_started_at + first_timeout
            } else {
                activity.last_activity() + idle_timeout
            };
            let deadline = watchdog_deadline.min(wall_clock_deadline);

            tokio::select! {
                item = event_stream.next() => break 'watchdog Ok(item),
                _ = tokio::time::sleep_until(deadline.into()) => {
                    let now = Instant::now();
                    if now >= wall_clock_deadline {
                        break 'watchdog Err(());
                    }
                    // Recompute: activity during the sleep moves the idle
                    // deadline forward, in which case the stream is alive.
                    let refreshed = if in_first_chunk_phase {
                        timing.attempt_started_at + first_timeout
                    } else {
                        activity.last_activity() + idle_timeout
                    };
                    if refreshed > now {
                        trace!("Upstream keep-alive observed; idle watchdog re-armed");
                        continue 'watchdog;
                    }
                    break 'watchdog Err(());
                }
                _ = downstream_tx.closed() => return Err(ClientDisconnected),
            }
        };

        let event_res = match next_item {
            Ok(Some(res)) => res,
            Ok(None) => {
                // Stream ended (EOF)
                break;
            }
            Err(()) => {
                // Timeout expired
                if last_finish_reason.is_some() {
                    break;
                } else if !received_first_meaningful_chunk {
                    warn!("First chunk timeout expired after {:?}", first_timeout);
                    return Ok(AttemptOutcome::Retryable(Cause::FirstChunkTimeout(
                        first_timeout,
                    )));
                } else {
                    warn!("Idle timeout expired after {:?}", idle_timeout);
                    return Ok(AttemptOutcome::Retryable(Cause::IdleTimeout(idle_timeout)));
                }
            }
        };

        let event = match event_res {
            Ok(ev) => ev,
            Err(e) => {
                warn!("Transport error reading upstream SSE stream: {}", e);
                return Ok(AttemptOutcome::Retryable(Cause::Transport(e.to_string())));
            }
        };

        let data = event.data.trim();
        if data.is_empty() {
            continue;
        }

        if config.log.log_bodies {
            trace!(
                downstream_id = %state.downstream_id,
                chunk = %data,
                "Upstream response chunk"
            );
        }

        if data == "[DONE]" {
            seen_done = true;
            break;
        }

        let mut chunk_json: serde_json::Value = match serde_json::from_str(data) {
            Ok(val) => val,
            Err(e) => {
                debug!("Skipping non-JSON SSE event data ({}): {}", e, data);
                continue;
            }
        };

        // §7: an upstream may report failure *inside* a 200 stream rather than
        // by status code. Such an event carries no delta, so it would otherwise
        // be ignored — and a following `[DONE]` would turn it into a successful
        // empty completion, the exact silent failure §7's give-up rules exist to
        // prevent.
        if let Some(error) = chunk_json.get("error").filter(|e| !e.is_null()) {
            let body = serde_json::json!({ "error": error }).to_string();
            warn!(error = %body, "Upstream reported an error inside the SSE stream");

            if is_context_or_policy_failure(&body) {
                return Ok(AttemptOutcome::Fatal(ApiError::from_status_and_body(
                    reqwest::StatusCode::BAD_GATEWAY,
                    &body,
                )));
            }
            // Anything else is treated as retryable: an in-band error is often a
            // transient quota or capacity problem. A genuinely permanent one
            // repeats identically and is caught by the no-progress cutoff (§7.5).
            let message = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("upstream stream error")
                .to_string();
            return Ok(AttemptOutcome::Retryable(Cause::UpstreamStreamError(
                message,
            )));
        }

        let details = parse_chunk_details(&chunk_json);

        // Update model name if this is the first time we see it from upstream chunks
        if let Some(m) = details.model {
            if !*state.model_frozen {
                *state.model = m;
                *state.model_frozen = true;
            }
        }

        // Accumulate usage if present
        if let Some(u) = details.usage {
            state.usage.update_from_json(&u);
        }

        let event_finish_reason = details.finish_reason.clone();
        if let Some(fr) = event_finish_reason.clone() {
            last_finish_reason = Some(fr);
        }

        let has_meaningful_delta = details.reasoning.is_some()
            || details.tool_calls.is_some()
            || details.content.is_some();
        if details.has_role && !has_meaningful_delta && !*state.emitted_role {
            *state.attempt_had_meaningful_chunk = true;
            rewrite_upstream_chunk(
                &mut chunk_json,
                state.downstream_id,
                state.created,
                state.model,
                state.emitted_role,
                false,
            );
            if send_downstream(
                downstream_tx,
                format_sse_data(&chunk_json),
                wall_clock_deadline,
            )
            .await
            .is_err()
            {
                debug!("Downstream client disconnected while sending role chunk");
                return Err(ClientDisconnected);
            }
        }

        // Reasoning is forwarded downstream but never appended to
        // `accumulated_text`, so it never enters continuation `messages` (§9).
        // It still needs its own overlap window: an attempt that fails after
        // reasoning but before any content leaves `accumulated_text` empty, so
        // the retry is a plain re-request and the upstream thinks aloud again.
        if let Some(reasoning) = details.reasoning {
            received_first_meaningful_chunk = true;
            *state.attempt_had_meaningful_chunk = true;
            let reasoning_field = details
                .reasoning_field
                .as_deref()
                .unwrap_or("reasoning_content");

            if let Some(text) = reasoning_dedup.push_delta(&reasoning) {
                if !text.is_empty() {
                    if state.accumulated_reasoning.len().saturating_add(text.len())
                        > config.resume.max_accumulated_bytes
                    {
                        warn!(
                            limit = config.resume.max_accumulated_bytes,
                            "Accumulated reasoning hit the configured size limit; ending the stream"
                        );
                        return Ok(AttemptOutcome::Terminal("length".to_string()));
                    }
                    state.accumulated_reasoning.push_str(&text);
                    let chunk = build_chunk_with_reasoning(
                        state.downstream_id,
                        state.created,
                        state.model,
                        &text,
                        reasoning_field,
                        !*state.emitted_role,
                    );
                    *state.emitted_role = true;
                    if send_downstream(downstream_tx, format_sse_data(&chunk), wall_clock_deadline)
                        .await
                        .is_err()
                    {
                        debug!("Downstream client disconnected while sending reasoning");
                        return Err(ClientDisconnected);
                    }
                    *state.emitted_payloads += 1;
                }
            }
        }

        // Handle tool calls
        if let Some(tool_calls) = details.tool_calls {
            received_first_meaningful_chunk = true;
            *state.attempt_had_meaningful_chunk = true;
            if config.tools.buffer_tool_calls {
                tool_buffer.process_tool_call_deltas(&tool_calls);
            } else {
                let chunk = build_chunk_with_tool_calls(
                    state.downstream_id,
                    state.created,
                    state.model,
                    &tool_calls,
                    !*state.emitted_role,
                );
                *state.emitted_role = true;
                if send_downstream(downstream_tx, format_sse_data(&chunk), wall_clock_deadline)
                    .await
                    .is_err()
                {
                    debug!("Downstream client disconnected while sending tool call chunk");
                    return Err(ClientDisconnected);
                }
                *state.emitted_payloads += 1;
            }
        }

        // Handle content deltas
        if let Some(content) = details.content {
            received_first_meaningful_chunk = true;
            *state.attempt_had_meaningful_chunk = true;
            if let Some(text_to_forward) = dedup_buffer.push_delta(&content) {
                if !text_to_forward.is_empty() {
                    if state
                        .accumulated_text
                        .len()
                        .saturating_add(text_to_forward.len())
                        > config.resume.max_accumulated_bytes
                    {
                        warn!(
                            limit = config.resume.max_accumulated_bytes,
                            "Accumulated response hit the configured size limit; ending the stream"
                        );
                        return Ok(AttemptOutcome::Terminal("length".to_string()));
                    }
                    state.accumulated_text.push_str(&text_to_forward);
                    let chunk = build_chunk_with_content(
                        state.downstream_id,
                        state.created,
                        state.model,
                        &text_to_forward,
                        !*state.emitted_role,
                    );
                    *state.emitted_role = true;

                    if send_downstream(downstream_tx, format_sse_data(&chunk), wall_clock_deadline)
                        .await
                        .is_err()
                    {
                        debug!("Downstream client disconnected while sending content chunk");
                        return Err(ClientDisconnected);
                    }
                    *state.emitted_payloads += 1;
                }
            }
        }
    }

    // Upstream stream finished. Flush any buffered dedup text.
    if let Some(text_to_forward) = dedup_buffer.finish() {
        state.accumulated_text.push_str(&text_to_forward);
        let chunk = build_chunk_with_content(
            state.downstream_id,
            state.created,
            state.model,
            &text_to_forward,
            !*state.emitted_role,
        );
        *state.emitted_role = true;

        if send_downstream(downstream_tx, format_sse_data(&chunk), wall_clock_deadline)
            .await
            .is_err()
        {
            debug!("Downstream client disconnected during dedup flush");
            return Err(ClientDisconnected);
        }
        *state.emitted_payloads += 1;
    }

    // Flush any reasoning still held behind its overlap window.
    if let Some(text) = reasoning_dedup.finish() {
        state.accumulated_reasoning.push_str(&text);
        let chunk = build_chunk_with_reasoning(
            state.downstream_id,
            state.created,
            state.model,
            &text,
            "reasoning_content",
            !*state.emitted_role,
        );
        *state.emitted_role = true;
        if send_downstream(downstream_tx, format_sse_data(&chunk), wall_clock_deadline)
            .await
            .is_err()
        {
            debug!("Downstream client disconnected during reasoning flush");
            return Err(ClientDisconnected);
        }
        *state.emitted_payloads += 1;
    }

    // Check finish reason
    let finish_reason = last_finish_reason.or_else(|| {
        if seen_done {
            Some("stop".to_string())
        } else {
            None
        }
    });

    match finish_reason {
        Some(fr) => {
            // §6.5: only the OpenAI-defined set reaches the client. Log whatever
            // the upstream actually said, then forward the normalised value.
            let normalized = normalize_finish_reason(&fr);
            if normalized != fr {
                debug!(
                    upstream_finish_reason = %fr,
                    normalized_finish_reason = %normalized,
                    "Normalised a non-OpenAI finish reason"
                );
            }

            // Decided BEFORE any flush: a failed generation must not push its
            // half-assembled call downstream, or the client sees a broken call
            // followed by the retry's replacement.
            if is_failed_generation_finish_reason(&fr) {
                warn!(
                    upstream_finish_reason = %fr,
                    "Upstream reported a failed generation; discarding partial tool state and retrying"
                );
                return Ok(AttemptOutcome::Retryable(Cause::MalformedFunctionCall));
            }

            if tool_buffer.is_truncated() {
                warn!(
                    limit = config.tools.max_tool_call_bytes,
                    "Tool call arguments exceeded the configured limit; discarding and retrying"
                );
                return Ok(AttemptOutcome::Retryable(Cause::MalformedFunctionCall));
            }

            if normalized == "tool_calls" && config.tools.buffer_tool_calls {
                // Flush assembled tool calls chunk downstream before finish chunk
                if let Some(tc_chunk) = tool_buffer.build_assembled_tool_calls_chunk(
                    state.downstream_id,
                    state.created,
                    state.model,
                    !*state.emitted_role,
                ) {
                    *state.emitted_role = true;
                    if send_downstream(
                        downstream_tx,
                        format_sse_data(&tc_chunk),
                        wall_clock_deadline,
                    )
                    .await
                    .is_err()
                    {
                        debug!("Downstream client disconnected during tool calls flush");
                        return Err(ClientDisconnected);
                    }
                    *state.emitted_payloads += 1;
                }
            }

            if normalized == "length" && config.resume.resume_on_length_finish {
                *state.finish_reason = Some(normalized.to_string());
                Ok(AttemptOutcome::Retryable(Cause::LengthFinishResumed))
            } else {
                *state.finish_reason = Some(normalized.to_string());
                Ok(AttemptOutcome::Terminal(normalized.to_string()))
            }
        }
        None => {
            if seen_done {
                *state.finish_reason = Some("stop".to_string());
                Ok(AttemptOutcome::Terminal("stop".to_string()))
            } else {
                warn!("Stream ended abruptly with no finish_reason and no [DONE]");
                Ok(AttemptOutcome::Retryable(Cause::StreamEndedUnexpectedly))
            }
        }
    }
}
