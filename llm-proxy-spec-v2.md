# llm-api-proxy — implementation spec (v2 / Revised)

A local HTTP proxy in Rust that sits between an OpenAI-compatible client (Jan.ai) and an
OpenAI-compatible upstream (initially Google's Gemini compatibility endpoint), and makes
streaming responses survive transient upstream failures.

> **Note on this revision (v2):** This document incorporates verified design choices and
> operational refinements established during implementation and testing, while preserving
> all core architectural invariants from v1. The original spec remains unchanged in `llm-proxy-spec.md`.
>
> **Amendment, 2026-08-20.** Three clauses were tightened after a coverage audit found the
> original wording either under-specified or actively harmful in practice. Each is marked
> **Amended (2026-08-20)** at the point of change, with the reasoning inline, and each is
> pinned by a named test so the behaviour cannot silently regress:
>
> 1. `Retry-After` is a **floor** on the backoff, not a replacement for it, and
>    `retry-after-ms` is read in preference to it (§7.2).
> 2. The fatal-body scan is **not applied to 429** except for unambiguous context-length
>    messages (§7.1).
> 3. The **idle** timer resets on upstream *bytes* — comment heartbeats included — while the
>    **first-chunk** timer remains an absolute budget from dispatch (§8.1).
> 4. `GET /v1/models` is **deadline-bounded** and no longer able to hang on a stalled
>    upstream (§5).
> 5. The downstream body limit is **raised and configurable**, because axum's 2 MiB default
>    rejects legitimate long conversations (§4, §5).
> 6. Re-emitted upstream `code`/`param` keep their **original JSON types** (§5).
>
> §3.1 records the downstream-channel properties that any change to the streaming path must
> preserve, independent of the payload type.

---

## 1. Problem statement

Jan.ai consumes streaming chat completions over SSE. When the upstream connection breaks
mid-response — connection reset, 429, 5xx, or a stream that simply goes silent — Jan treats
the truncated stream as a completed response. There is no retry and no resumption; the user
sees a message that stops mid-sentence and is recorded in history as final.

The proxy fixes this by owning the downstream SSE connection for the entire logical response.
Upstream requests become disposable attempts behind that connection. When an attempt fails,
the proxy backs off, re-issues the request with the partial output as context, deduplicates
the overlap, and continues emitting chunks on the same downstream stream. Jan sees one
uninterrupted response.

**Non-goal:** this is not a load balancer, router, cache, or key manager. One upstream,
configured at startup.

---

## 2. Pre-implementation verification & Continuation Modes

Two facts govern upstream compatibility:

1. **Wire Failure Types:** TCP reset mid-stream, HTTP 429/503 status codes, and idle silent stalls. All three are handled by dedicated transport recovery, backoff, and watchdog timers (§8).
2. **Continuation Behavior:**
   - **`ContinuationMode::Prefix` (default):** Sends partial output in a trailing `assistant` message. If the model continues smoothly from that prefix, this mode gives zero-seam resumption.
   - **`ContinuationMode::Instructed`:** When an upstream model does not support trailing assistant turns or restarts from the beginning, appends the assistant turn plus an explicit instruction turn to continue without repetition. Handled via the overlap deduplication window (§6.3).

Both modes are supported behind the `resume.mode` configuration option.

---

## 3. Architecture

```
Jan.ai ──SSE──> [ proxy: downstream stream owner ]
                        │
                        ├── attempt 1 ──> upstream  (fails at N tokens)
                        ├── backoff (with SSE keepalive comments)
                        ├── attempt 2 ──> upstream  (resumes, dedup, continues)
                        └── attempt 3 ──> upstream  (completes, [DONE])
```

Single binary. `axum` server, `reqwest` client, `tokio` runtime. One `tokio::spawn`ed task
per downstream request driving an outer retry loop around an inner
consume-upstream-stream function; chunks flow to the response body via an
`mpsc::channel` bridged into an axum `Body::from_stream`.

**Invariant:** the downstream stream emits `data: [DONE]` exactly once, and only when the
response is genuinely terminal (§7). No upstream failure ever closes it prematurely.

### 3.1 Downstream channel contract

The channel payload type is an implementation detail (`String`, `Bytes`, or anything else
axum's `Body::from_stream` accepts). These properties are not, and any change to the
streaming path must preserve all of them:

1. **Framing is byte-identical** to §6.6: `data: {compact-json}\n\n` per chunk, `: {text}\n\n`
   for keepalive comments, `data: [DONE]\n\n` to terminate. No pretty-printing, and no
   newline may appear inside a serialised chunk, which would split the frame.
2. **A send failure means the client is gone.** Every `send` on the channel is checked; a
   failure aborts the attempt and unwinds the task so the in-flight upstream request is
   dropped (§11). Errors must never be swallowed to "keep the stream tidy".
3. **`downstream_tx.closed()` remains selectable** at every await point that can block for a
   long time — dispatch, error-body read, backoff sleep, and the event wait (§8.1).
4. **Backpressure is preserved.** The channel is bounded; `send` must stay `await`ed rather
   than becoming a `try_send` that drops frames under load.
5. **UTF-8 boundaries are never split.** Content deltas are assembled and deduplicated as
   `char`s (§6.3); a byte-oriented payload type must still frame whole chunks, never partial
   multi-byte sequences.

---

## 4. Configuration

TOML file, path from `--config` or `$LLM_PROXY_CONFIG`, defaulting to
`~/.config/llm-proxy/llm-proxy.toml`. Every field is overridable by environment variable (`LLM_PROXY_<SECTION>_<KEY>`).

```toml
[server]
bind = "127.0.0.1:8787"
max_body_bytes = 33554432           # 32 MiB; see §5 — axum's 2 MiB default rejects long chats

[upstream]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key_env = "GEMINI_API_KEY"     # read the key from this env var (recommended)
api_key = ""                        # optional: direct key override in config / env (LLM_PROXY_UPSTREAM_API_KEY)
connect_timeout_secs = 10

[resume]
enabled = true
mode = "prefix"                     # "prefix" | "instructed"
max_attempts = 6                    # total upstream attempts per downstream request
max_wall_clock_secs = 900           # hard ceiling across all attempts
backoff_initial_ms = 500
backoff_max_ms = 30000
backoff_multiplier = 2.0
backoff_jitter = 0.25               # ±25% randomisation
dedup_window_chars = 240            # see §6.3
first_chunk_timeout_secs = 120      # timeout until first content, tool-call, or reasoning chunk
idle_timeout_secs = 45              # no chunk for this long => treat as break
resume_on_length_finish = false     # whether finish_reason "length" is treated as resumable

[tools]
buffer_tool_calls = true            # assemble tool call fragments before emitting (§9)
disable_resume_for_tool_responses = false

[log]
level = "info"
log_bodies = false                  # request/response bodies at trace level; off by default
```

`base_url` is fully configurable, and any query string on it is preserved when the endpoint
path is appended. If `base_url` already ends with the endpoint path *as a whole path segment*
— `.../v1/chat/completions` for the chat endpoint — the path is not appended again, which
lets a `base_url` point directly at a single endpoint. The comparison is made on `/`
boundaries, so a base merely *ending with* those characters (`.../v1/mock_models` for the
`models` path) still has the path appended.

Permissive CORS headers are enabled to seamlessly support local desktop clients (like Jan.ai).

---

## 5. Endpoints

| Route | Behaviour |
|---|---|
| `POST /v1/chat/completions` | Core path. §6 if `stream: true`, §10 if not. |
| `GET /v1/models` | Transparent proxy to upstream, single retry on 5xx/429/transport error, honouring `Retry-After`. Passes raw upstream body through. Both attempts together are bounded by `first_chunk_timeout_secs`; on expiry the proxy returns `504` with an OpenAI-shaped body rather than hanging. |
| `GET /health` | `200 {"status":"ok"}`. No upstream call. |
| anything else | `404` with an OpenAI-shaped error body. |

**Auth:** Client's `Authorization` header is discarded; the proxy always injects its own configured upstream API key if available.

**Passthrough:** Request body is handled as untyped JSON (`serde_json::Value`). All client parameters (temperature, top_p, custom flags) survive untouched; only `messages` and `max_tokens` are rewritten for continuation attempts.

**Request size — *Amended (2026-08-20)*:** the downstream router sets an explicit
`server.max_body_bytes` (32 MiB default). axum's built-in 2 MiB default is well below a
realistic long conversation or a pasted document, and silently rejecting a legitimate prompt
with `413` is the failure mode that actually occurs here — the proxy binds to loopback by
default, so remote resource exhaustion is not the threat being defended against. The limit
remains configurable for deployments that do expose the port.

**Error bodies — *Amended (2026-08-20)*:** when an upstream error is re-emitted (as a fatal
error chunk on the streaming path, or by `/v1/models`), `code` and `param` are forwarded with
their original JSON types. OpenAI's `code` may be a string or a number, and re-encoding it —
as a quoted string, or coercing a number to a string — changes what a client can match on.

---

## 6. Streaming path

### 6.1 State

Held across attempts for one downstream request:

```rust
struct ResponseState {
    downstream_id: String,        // "chatcmpl-" + uuid, generated once
    created: i64,                 // unix ts, generated once
    model: String,                // from original request, frozen/overridden on first upstream chunk
    accumulated_text: String,     // all content deltas emitted downstream so far
    emitted_role: bool,           // whether the role delta has gone out
    attempt: u32,
    started_at: Instant,
    original_body: serde_json::Value,
    usage: UsageAccumulator,
}
```

### 6.2 Outer loop

```
loop {
    if attempt > max_attempts || started_at.elapsed() > max_wall_clock => log give-up; emit finish chunk + [DONE]; break
    build request body (§6.4)
    open upstream stream
    match consume_stream(&mut state) {
        Terminal(reason)  => emit finish chunk + [DONE]; break
        Retryable(err)    => log; sleep(backoff(attempt)); attempt += 1; continue
        Fatal(err)        => emit error chunk + [DONE]; break
    }
}
```

**During backoff, the downstream stream must stay alive.** Emit an SSE comment line
(`: retrying\n\n`) every 10 seconds while sleeping to prevent socket timeouts.

### 6.3 Overlap deduplication

On every attempt after the first:

1. Do not forward content deltas immediately. Accumulate resumed text into a `pending` buffer.
2. Once `pending.chars().count() >= dedup_window_chars` — or the upstream stream ends, whichever is first — run the match.
3. Let `tail` be the last `dedup_window_chars` of `accumulated_text`. For `k` from `min(tail.len(), pending.len())` down to 1, if `pending` starts with the final `k` characters of `tail`, take the largest `k`, drop the first `k` characters of `pending`, and forward the rest.
4. If no `k` matches, forward `pending` unchanged.
5. From that point on in this attempt, forward deltas as they arrive with no buffering.
6. All comparisons and slicing operate strictly on `char` boundaries. Anything forwarded downstream is appended to `accumulated_text`.

### 6.4 Building the continuation request

If `accumulated_text` is empty, send `original_body` unmodified (plain retry).

- **`Prefix` mode:** Append `{"role": "assistant", "content": "<accumulated_text>"}` to `messages`.
- **`Instructed` mode:** Append the assistant turn followed by `{"role": "user", "content": "Continue your previous response from exactly where it stopped. Do not repeat any text, do not restate or summarise, do not acknowledge this instruction — output only the continuation."}`.
- In both modes, decrement `max_tokens` or `max_completion_tokens` by estimated tokens emitted (`accumulated_text.chars().count() / 4`, floored at 1).

### 6.5 Chunk rewriting

Every chunk forwarded downstream is rewritten:
- `id` → `state.downstream_id`
- `created` → `state.created`
- `model` → `state.model`
- `choices[0].delta.role` → emitted once, on the first chunk of the whole response only.
- `choices[0].finish_reason` → suppressed during stream; forwarded only on terminal outcome.

### 6.6 SSE framing

Downstream: `data: {json}\n\n` per chunk, terminated by `data: [DONE]\n\n`.
Headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`.

---

## 7. Error classification

```rust
enum AttemptOutcome {
    Terminal(String),   // finish_reason; response complete
    Retryable(Cause),   // back off and resume
    Fatal(ApiError),    // propagate to client, do not retry
}
```

**Terminal:** `finish_reason` of `stop`, `tool_calls`, `content_filter`, or `length` when `resume_on_length_finish = false`.

**Retryable:**
- HTTP 408, 409, 425, 429, 500, 502, 503, 504
- Transport errors: connection reset, broken pipe, incomplete message, TLS error, connect timeout, read timeout
- `first_chunk_timeout` or `idle_timeout` expiry (§8)
- Upstream stream ending with no `finish_reason` and no `[DONE]`
- `finish_reason: "length"` when `resume_on_length_finish = true`

**Fatal:**
- HTTP 400, 401, 403, 404, 413, 422
- Any error response whose body or error message indicates a context-length exceeded or content-policy/safety violation (regardless of HTTP status code) — **except on 429, see below**
- Unrecoverable client errors

**`resume.enabled = false` Semantics:** If `resume.enabled = false`, mid-stream failures do not attempt continuation; the stream terminates cleanly with a `stop` chunk. Initial pre-stream transient errors (e.g. initial 429/503 before any chunk) still follow normal retry up to `max_attempts`.

**Give-up behaviour:** When attempts or wall clock are exhausted, log at error level and emit a clean finish chunk with `finish_reason: "stop"` followed by `[DONE]`. Do not inject synthetic error text into the user's chat stream.

### 7.1 Fatal-body scanning and the 429 exception — *Amended (2026-08-20)*

The body scan is split into three vocabularies:

| Predicate | Example phrases | Fatal on |
|---|---|---|
| Unambiguous context length | `context_length_exceeded`, `maximum context length`, `context window`, `context limit`, `prompt is too long`, `string_too_long`, `reduce your prompt`, `reduce the length of the messages` | every status, 429 included |
| Ambiguous size | `token limit`, `too many tokens` | every status **except** 429 |
| Content policy / safety | `content_policy`, `content filter`, `safety policy`, `blocked by safety`, `prohibited content`, `harm_category`, … | every status **except** 429 |

**Why 429 is excluded.** A rate limit is the one error whose own vocabulary collides with this
heuristic: quotas are metered in tokens per minute, so throttling messages routinely say
"token limit" or "too many tokens". Scanning a 429 with the general predicate therefore
converts the single most recoverable upstream condition into a hard failure. Only a message
that can *only* mean an oversized context may condemn a 429 — there, retrying is genuinely
futile, because each continuation attempt resends a prompt that is already too long.

**Accepted cost:** a genuine safety refusal delivered *as* a 429 is now retried up to
`max_attempts` rather than failing immediately. That pairing is rare — refusals arrive as
400 — and the retries are bounded, so the cost is latency, not correctness.

### 7.2 Backoff and `Retry-After` — *Amended (2026-08-20)*

The retry hint is read from `retry-after-ms` when present, falling back to `Retry-After`
(seconds or HTTP-date). RFC 9110 defines `Retry-After` delay-seconds as `1*DIGIT`, so a
provider wanting sub-second precision cannot express it in the standard header and sends the
millisecond variant instead; OpenAI and Azure OpenAI both do.

Whichever header supplies it, the hint is honoured as a **floor** on the backoff, not as a
replacement for it:

```
delay = min( max( exponential_with_jitter(attempt), retry_after ), backoff_max_ms )
```

The proxy therefore never returns sooner than the upstream asked, but it does not hold a
constant cadence either.

**Why not obey the header verbatim.** `max_attempts` and `max_wall_clock_secs` are meant to
be independent ceilings. An upstream answering `Retry-After: 1` to every attempt collapses
them into one: the entire six-attempt budget is spent in roughly six seconds, and the proxy
gives up with almost all of its 900-second wall clock unused — on precisely the condition
most likely to clear if it simply waited longer. A hint that stops varying while the failures
keep arriving also carries no information: many gateways emit a fixed constant regardless of
real load, and at that point the exponential curve is the better estimator.

Consequences of the floor form:
- A longer hint still wins outright: `Retry-After: 30` yields a 30s wait, clamped only by `backoff_max_ms`.
- A repeated `Retry-After: 1` escalates roughly 1s, 1s, 2s, 4s, 8s, 16s under default settings — the exponential component still carries `backoff_jitter` (±25% by default), so only the floored early attempts are exact.
- An HTTP-date already in the past parses as zero and falls through to the exponential value, rather than producing an immediate zero-delay retry.
- This applies to every retryable status carrying the header, not only 429.


---

## 8. Timeouts

- **`first_chunk_timeout_secs`** — from request dispatch until the first chunk containing non-empty content, a tool-call delta, or a reasoning/thinking delta. This is an **absolute budget measured from dispatch**, and it includes the time spent establishing the connection and awaiting response headers.
- **`idle_timeout_secs`** — between consecutive chunks once streaming has begun.

On expiry: abort the upstream request and return `Retryable(FirstChunkTimeout | IdleTimeout)`.

### 8.1 What resets each timer — *Amended (2026-08-20)*

The two timers observe different signals, and the distinction is load-bearing:

- The **idle** timer resets on any **bytes** arriving from upstream, whether or not they
  parse into a dispatchable SSE event. SSE comment heartbeats (`: keep-alive`) are discarded
  by the event decoder without producing an event, so an event-driven watchdog cannot
  distinguish a deliberately kept-alive stream from a stalled one and will tear down a
  healthy connection. Upstreams and the reverse proxies in front of them send these
  heartbeats for exactly this purpose — the proxy emits `: retrying` downstream for the same
  reason (§6.2) — so they must be honoured on the way in as well.
- The **first-chunk** timer does **not** reset, on heartbeats or on anything else. It is a
  budget, not a watchdog: §8 defines it as the time until the first *meaningful* delta, and
  heartbeats are evidence of liveness, not of progress. A connection that stays open while
  producing nothing still fails at `first_chunk_timeout_secs`.

A stream that delivers bytes continuously without ever completing an event would hold the
idle timer open indefinitely; `max_wall_clock_secs` remains the backstop, and every watchdog
deadline is clamped to it.

---

## 9. Tool calls and reasoning content

- **Tool Call Buffering (`buffer_tool_calls = true`, default):** Accumulate `delta.tool_calls[*]` keyed by index. When `finish_reason: "tool_calls"` is reached, emit the complete assembled tool call(s) in a single consolidated delta followed by the finish chunk. If an attempt fails mid-tool-call, discard the uncommitted tool call state and retry from the last committed text state.
- **Tool Call Passthrough (`buffer_tool_calls = false`):** Forward tool call deltas as they arrive, rewriting IDs and suppressing intermediate finish reasons. Note that passthrough has **no equivalent of the §6.3 overlap window**: fragments already forwarded by an attempt that later fails cannot be recalled, so a cut mid-tool-call leaves the client holding the abandoned fragment as well as the complete retry. Buffering (the default) does not have this problem, because nothing is emitted until the call is whole.
- **Reasoning / Thinking content:** Forward `delta.reasoning_content` / `delta.reasoning` downstream immediately, but **exclude from `accumulated_text` and continuation `messages`**.
- **`disable_resume_for_tool_responses = true`:** Requests with a `tools` array skip resumption if a failure occurs after text/content has begun emitting.

---

## 10. Non-streaming requests

`stream: false` issues the request, classifies errors with §7 rules, and retries on `Retryable`. On success, passes the raw upstream response status and body through verbatim (preserving original key order and types). On give-up or fatal error, returns the upstream error status and body.

---

## 11. Client disconnect

When the downstream client disconnects (detected via mpsc channel send failure), immediately abort in-flight upstream requests and cleanly terminate the spawned task.

---

## 12. Logging

- `tracing` + `tracing-subscriber`, `EnvFilter` with `RUST_LOG` or configured log level.
- Info: request start, attempt boundaries, retries with cause/backoff, completion stats.
- Debug: dedup results (overlap character count).
- Trace: request and response bodies (gated by `log_bodies = true`).
- Redact Authorization headers; never log secret API keys.

---

## 13. Out of scope

Explicitly out of scope: `n > 1` choices, non-JSON multimodal handling, legacy `/v1/completions`, embeddings, multi-upstream load balancing, caching, rate limit enforcement, and native non-OpenAI protocols.

---

## 14. Acceptance & Testing

Two upstream harnesses are used, because they can express different failures.

**1. Configurable mock upstream** (`src/mock.rs`, `src/bin/mock_upstream.rs`), driven by
`tests/integration_tests.rs`. Covers clean streams, single and multiple cuts, overlap dedup,
stalls, `Retry-After` in both seconds and HTTP-date form, fatal error propagation, give-up,
chunk identity, tool calls across cuts, client disconnect, non-streaming retry, timeout
recovery, the complete retryable and fatal status matrices from §7 on both the streaming and
non-streaming paths, the §6.6 response headers, and CORS.

**2. Raw TCP upstream**, driven by `tests/raw_upstream_tests.rs`. The mock speaks through
axum, so it always terminates its response body cleanly and the proxy only ever observes a
well-formed EOF from it — the wire failures §2 and §7 actually care about are unreachable
through it. The raw harness hand-rolls HTTP/1.1 on a socket and covers:

- TCP reset mid-stream (`SO_LINGER 0`), incomplete chunked message, connection closed before the response head, and TLS handshake failure.
- Malformed and multi-line SSE framing, unknown SSE fields, and comment heartbeats (§8.1).
- Interleaved fragmented tool calls across indices, discard of uncommitted tool-call state across a failed attempt, and passthrough behaviour across a cut (§9).
- Abort-on-client-disconnect (§11), asserted by measuring when the *upstream* socket actually closes rather than by inspecting proxy state.
- Multi-byte text (CJK, accents, emoji ZWJ sequences, RTL) across a retry cut and across a replayed overlap, since the mock corpus is pure ASCII and cannot exercise the `char`-boundary requirement of §6.3.
- `GET /v1/models` against an upstream that accepts the connection and then stalls, on both the header and body reads (§5).

The amended clauses of §7.1, §7.2 and §8.1 are each pinned by a named test; the heartbeat
clause in particular is guarded from both sides, by one test asserting that heartbeats keep a
stream alive and another asserting that genuine silence still trips the idle timer.
