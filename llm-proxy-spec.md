# llm-api-proxy — implementation spec

A local HTTP proxy in Rust that sits between an OpenAI-compatible client (Jan.ai) and an
OpenAI-compatible upstream (initially Google's Gemini compatibility endpoint), and makes
streaming responses survive transient upstream failures.

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

## 2. Pre-implementation verification

Two facts must be established with `curl` before writing the resumption logic, because the
architecture depends on the answers. Do these first and record the results.

### 2.1 What does the failure actually look like on the wire?

Run the proxy in passthrough mode (or `mitmproxy`) and capture a failing session. Classify it:

| Observed | Mechanism needed |
|---|---|
| TCP reset / `IncompleteMessage` mid-stream | Transport error handling |
| HTTP 429 / 503 before or during stream | Status-based retry |
| Stream stops sending chunks, socket stays open | **Idle watchdog** (§8) |
| Clean `[DONE]` with `finish_reason: "STOP"` but truncated text | Not recoverable by this proxy — different problem |

Implement all of the first three regardless; the capture tells you which to prioritise in
testing.

### 2.2 Does the upstream honour a trailing assistant turn?

The entire resumption design rests on being able to send the conversation with the partial
output appended as a final `assistant` message, and having the model *continue* that text
rather than start a fresh reply.

Test against `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`:

```json
{
  "model": "<model>",
  "messages": [
    {"role": "user", "content": "Count from 1 to 20, one number per line."},
    {"role": "assistant", "content": "1\n2\n3\n4\n5"}
  ],
  "stream": false
}
```

- **Continues from 6** → prefix continuation works. Use `ContinuationMode::Prefix`.
- **Restarts from 1, or errors on a trailing assistant turn** → fall back to
  `ContinuationMode::Instructed` (§6.4), which needs more aggressive dedup and produces
  visible seams. Note this in the README.

Implement both modes behind an enum; make the default configurable. Do not assume Prefix
works.

---

## 3. Architecture

```
Jan.ai ──SSE──> [ proxy: downstream stream owner ]
                        │
                        ├── attempt 1 ──> upstream  (fails at N tokens)
                        ├── backoff
                        ├── attempt 2 ──> upstream  (resumes, dedup, continues)
                        └── attempt 3 ──> upstream  (completes, [DONE])
```

Single binary. `axum` server, `reqwest` client, `tokio` runtime. One `tokio::spawn`ed task
per downstream request driving an outer retry loop around an inner
consume-upstream-stream function; chunks flow to the response body via an
`mpsc::channel` bridged into an axum `Body::from_stream`.

**Invariant:** the downstream stream emits `data: [DONE]` exactly once, and only when the
response is genuinely terminal (§7). No upstream failure ever closes it.

---

## 4. Configuration

TOML file, path from `--config` or `$LLM_PROXY_CONFIG`, defaulting to
`~/.config/llm-proxy/llm-proxy.toml`. Every field overridable by env var (`LLM_PROXY_<SECTION>_<KEY>`).

```toml
[server]
bind = "127.0.0.1:8787"

[upstream]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key_env = "GEMINI_API_KEY"     # read the key from this env var, never store it in the file
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
first_chunk_timeout_secs = 120      # generous: thinking models pause before first token
idle_timeout_secs = 45              # no chunk for this long => treat as break
resume_on_length_finish = false     # whether finish_reason "length" is treated as resumable

[tools]
buffer_tool_calls = true            # see §9
disable_resume_for_tool_responses = false

[log]
level = "info"
log_bodies = false                  # request/response bodies at trace level; off by default
```

`base_url` being configurable is deliberate — nothing in the implementation should be
Google-specific.

---

## 5. Endpoints

| Route | Behaviour |
|---|---|
| `POST /v1/chat/completions` | Core path. §6 if `stream: true`, §10 if not. |
| `GET /v1/models` | Transparent proxy to upstream, single retry on 5xx/429. Jan's model picker breaks without it. |
| `GET /health` | `200 {"status":"ok"}`. No upstream call. |
| anything else | `404` with an OpenAI-shaped error body. |

Auth: read the client's `Authorization` header and **discard it**; always inject the
proxy's own configured key. Jan can be pointed at the proxy with a dummy key.

Request body is parsed as `serde_json::Value`, not a typed struct. Unknown fields must
survive the round trip — providers add parameters faster than you'll update a struct.
Only specific fields are read (`stream`, `model`, `messages`, `tools`) and only `messages`
is rewritten.

---

## 6. Streaming path

### 6.1 State

Held across attempts for one downstream request:

```rust
struct ResponseState {
    downstream_id: String,        // "chatcmpl-" + uuid, generated once
    created: i64,                 // unix ts, generated once
    model: String,                // from the first upstream chunk, then frozen
    accumulated_text: String,     // all content deltas emitted downstream so far
    emitted_role: bool,           // whether the role delta has gone out
    attempt: u32,
    started_at: Instant,
    original_body: serde_json::Value,
    finish_reason: Option<String>,
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
(`: retrying\n\n`) every 10 seconds while sleeping. Comment lines are ignored by conformant
SSE parsers but keep the socket warm and prevent intermediaries from timing out. Verify Jan
tolerates them; if it doesn't, cap backoff below its client timeout instead.

### 6.3 Overlap deduplication

Continuations frequently repeat the tail of what was already produced, and `Instructed`
mode makes this near-certain.

On every attempt after the first:

1. Do not forward content deltas immediately. Accumulate resumed text into a `pending`
   buffer.
2. Once `pending.len() >= dedup_window_chars` — or the upstream stream ends, whichever is
   first — run the match.
3. Let `tail` be the last `dedup_window_chars` of `accumulated_text`. For `k` from
   `min(tail.len(), pending.len())` down to 1, if `pending` starts with the final `k`
   characters of `tail`, that's the overlap. Take the largest such `k`, drop the first `k`
   characters of `pending`, forward the rest.
4. If no `k` matches, forward `pending` unchanged.
5. From that point on in this attempt, forward deltas as they arrive with no buffering.

Operate on `char` boundaries, not bytes — slicing a multi-byte UTF-8 sequence will panic or
corrupt output. Consider comparing on whitespace-normalised copies while slicing the
originals, since a continuation may differ from the original only by a leading space.

Anything forwarded downstream is appended to `accumulated_text`.

### 6.4 Building the continuation request

Start from `original_body`. If `accumulated_text` is empty (failure before any content),
send it unmodified — this is a plain retry.

**`ContinuationMode::Prefix`** — append to `messages`:

```json
{"role": "assistant", "content": "<accumulated_text>"}
```

**`ContinuationMode::Instructed`** — append two messages:

```json
{"role": "assistant", "content": "<accumulated_text>"},
{"role": "user", "content": "Continue your previous response from exactly where it stopped. Do not repeat any text, do not restate or summarise, do not acknowledge this instruction — output only the continuation."}
```

In both modes, if `max_tokens` is set, decrement it by an estimate of tokens already
emitted (`accumulated_text.chars().count() / 4`, floored at 1) so the total stays roughly
within what the client asked for.

### 6.5 Chunk rewriting

Every chunk forwarded downstream is rewritten so the response looks like one continuous
generation:

- `id` → `state.downstream_id` (upstream assigns a new id per attempt; Jan may treat a
  changed id as a new message).
- `created` → `state.created`.
- `model` → `state.model`.
- `choices[0].delta.role` → emitted once, on the first chunk of the whole response only.
  Strip it from resumed chunks.
- `choices[0].finish_reason` → forwarded only when terminal (§7). A `finish_reason` on a
  chunk from a failed attempt must be suppressed.

Only ever handle `choices[0]`; `n > 1` is out of scope (§13).

### 6.6 SSE framing

Downstream: `data: {json}\n\n` per chunk, terminated by `data: [DONE]\n\n`.
Response headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`,
`Connection: keep-alive`, `X-Accel-Buffering: no`.

Parse the upstream stream with `eventsource-stream` over `reqwest`'s byte stream. Do not
hand-roll SSE parsing — chunk boundaries split events at arbitrary byte offsets. Ignore
upstream comment lines and events whose data isn't valid JSON (log at debug, don't fail).

---

## 7. Error classification

```rust
enum AttemptOutcome {
    Terminal(String),   // real finish_reason; response is complete
    Retryable(Cause),   // back off and resume
    Fatal(ApiError),    // propagate to client, do not retry
}
```

**Terminal:** `finish_reason` of `stop`, `tool_calls`, `content_filter`; or `length` when
`resume_on_length_finish = false`.

**Retryable:**
- HTTP 408, 409, 425, 429, 500, 502, 503, 504
- Transport errors: connection reset, broken pipe, incomplete message, TLS error,
  connect timeout, read timeout
- `first_chunk_timeout` or `idle_timeout` expiry (§8)
- Upstream stream ending with no `finish_reason` and no `[DONE]`
- `finish_reason: "length"` when `resume_on_length_finish = true`

**Fatal:** HTTP 400, 401, 403, 404, 413, 422; malformed upstream JSON that persists; any
error body whose message indicates a context-length or content-policy failure. These will
never succeed on retry and looping on them wastes quota.

On 429, honour `Retry-After` (seconds or HTTP-date) if present, in preference to computed
backoff, clamped to `backoff_max_ms`.

**Give-up behaviour.** When attempts or wall clock are exhausted, log the give-up in
the proxy output/logs (at error/warn level), but do not inject any synthetic failure or
proxy error message into the chat text stream sent to Jan.ai. Simply emit a chunk with
`finish_reason: "stop"` followed by `[DONE]`, leaving whatever text was accumulated so far
cut off naturally as-is in Jan.ai's chat interface.

---

## 8. Timeouts

Two distinct timers, both reset-on-activity:

- **`first_chunk_timeout_secs`** — from request dispatch until the first chunk containing
  non-empty content or a tool-call delta. Must be generous; reasoning models can be silent
  for a long time before any visible output.
- **`idle_timeout_secs`** — between consecutive chunks once streaming has begun.

Implement with `tokio::time::timeout` wrapped around each `stream.next()` await, switching
the duration after the first content chunk. On expiry: abort the upstream request (drop the
response, which cancels it) and return `Retryable(IdleTimeout)`.

No overall per-attempt timeout beyond these — a legitimately long generation should not be
killed for being long.

---

## 9. Tool calls and reasoning content

Partial tool-call arguments cannot be resumed by text continuation. Half a JSON object
appended as an assistant turn produces garbage.

**Default (`buffer_tool_calls = true`):** accumulate `delta.tool_calls[*]` fragments in the
proxy, keyed by index, without forwarding. Only when the upstream reports
`finish_reason: "tool_calls"` do you emit the assembled tool calls downstream — as a
correctly-shaped sequence of deltas — followed by the finish chunk. If an attempt fails
mid-tool-call, discard the partial accumulation entirely and retry the request from the
last committed text state. This costs streaming granularity on tool calls, which is
invisible to the user, and buys correctness.

**Reasoning/thinking content** (`delta.reasoning_content`, `delta.reasoning`, or provider
equivalents): forward it downstream if present, but **exclude it from
`accumulated_text`** and from the continuation `messages`. It is not part of the assistant's
visible output and re-submitting it will confuse the model. Reasoning produced before a
failure is simply lost; that's acceptable.

If `disable_resume_for_tool_responses = true`, a request carrying a `tools` array skips
resumption entirely and falls back to whole-request retry (only valid before any content has
been emitted). Provided as an escape hatch if buffering proves problematic.

---

## 10. Non-streaming requests

`stream: false` is a plain retry loop: issue the request, classify the outcome with the same
rules (§7), back off and re-issue on `Retryable`, return the upstream body verbatim on
success. No resumption, no continuation, no chunk rewriting — a non-streaming failure
returns nothing partial, so there is nothing to resume from. On give-up, return the last
upstream error status and body.

---

## 11. Client disconnect

If Jan closes the connection, the `mpsc` sender's receiver drops and sends start failing.
Detect this and abort the in-flight upstream request immediately rather than continuing to
generate tokens nobody will read. Dropping the `reqwest` response future is sufficient to
cancel it. Log at info and exit the task cleanly.

---

## 12. Logging

`tracing` + `tracing-subscriber`, `EnvFilter` from `RUST_LOG` with the config value as
default.

Per downstream request, one span carrying `downstream_id`. Log at **info**: request start
(model, stream, message count), each attempt boundary, each retry with cause and backoff
duration, completion with attempt count and total duration. At **debug**: dedup results
(overlap length found, or none). At **trace**, gated behind `log_bodies`: full request and
response bodies.

Never log the API key. Redact `Authorization` headers.

---

## 13. Out of scope

State plainly in the README so nobody expects them: multiple choices (`n > 1`), multimodal
inputs beyond passthrough, the legacy `/v1/completions` endpoint, embeddings, multiple
upstreams or failover, request caching, rate-limit enforcement, and any Anthropic- or
Gemini-native (non-OpenAI-shaped) API surface.

---

## 14. Testing

The resume path cannot be tested reliably against a real provider. Build a mock upstream.

### 14.1 Mock upstream binary

A second binary target (`src/bin/mock_upstream.rs`), axum, serving
`POST /v1/chat/completions`. Behaviour driven by query parameters so tests can script
failures:

| Param | Effect |
|---|---|
| `cut_after=<n>` | Emit `n` chunks, then drop the connection without `[DONE]` |
| `stall_after=<n>` | Emit `n` chunks, then hold the socket open forever |
| `status=<code>&fail_first=<n>` | Return `code` for the first `n` requests, then succeed |
| `repeat_overlap=<n>` | On a continuation, repeat the last `n` chars of the prior output |
| `restart` | Ignore the trailing assistant turn; always emit from the beginning |
| `slow_first_chunk=<secs>` | Delay before the first chunk |

It should read the trailing assistant message and continue a deterministic corpus (e.g.
numbers 1–200, or a fixed lorem passage) from the correct offset, so tests can assert the
reassembled output equals the corpus exactly.

### 14.2 Required integration tests

Each spins up mock + proxy on ephemeral ports, drives a real SSE client, and asserts on the
reassembled text and chunk metadata.

1. **Clean passthrough** — no failure; output matches corpus, one `[DONE]`.
2. **Single mid-stream cut** — output matches corpus exactly, no duplicated or missing text.
3. **Three consecutive cuts** — same assertion; verifies repeated resumption.
4. **Overlap dedup** — `repeat_overlap=80`; output matches corpus with no repetition.
5. **Stall** — `stall_after=5` with `idle_timeout_secs=2`; proxy recovers and completes.
6. **429 then success** — retried, `Retry-After` honoured, output complete.
7. **400 fatal** — no retry loop, error propagated, exactly one `[DONE]`.
8. **Give-up** — mock always cuts; proxy stops at `max_attempts`, logs the give-up in
   proxy logs without injecting error text into the stream, and terminates with a clean
   finish chunk and `[DONE]`.
9. **Chunk identity** — across a resumed response, every chunk carries the same `id`,
   `created`, and `model`, and `delta.role` appears exactly once.
10. **Tool call across a cut** — cut mid-arguments; assert the client receives one complete,
    valid tool call and never a partial one.
11. **Client disconnect** — drop the client mid-stream; assert the mock observes its
    connection closing rather than the generation running to completion.
12. **Non-streaming retry** — `stream: false` with `fail_first=2`; correct body returned.

Unit tests for the dedup function specifically: no overlap, full overlap, partial overlap,
overlap at a multi-byte character boundary, overlap longer than the window, empty pending
buffer.

---

## 15. Dependencies

```toml
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.5", features = ["trace"] }
reqwest = { version = "0.12", features = ["stream", "json", "rustls-tls"], default-features = false }
eventsource-stream = "0.2"
futures-util = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
thiserror = "1"
uuid = { version = "1", features = ["v4"] }
rand = "0.8"
```

Suggested layout:

```
src/
  main.rs           entry, config load, server bind
  config.rs         config structs + env overrides
  error.rs          AttemptOutcome, Cause, ApiError, classification
  proxy/
    mod.rs
    stream.rs       outer retry loop, ResponseState
    consume.rs      inner upstream stream consumption, timeouts
    chunks.rs       chunk parse + rewrite
    dedup.rs        overlap matching (unit-tested in isolation)
    continuation.rs continuation request construction
    tools.rs        tool-call buffering
    nonstream.rs    stream:false path
  bin/
    mock_upstream.rs
```

---

## 16. Acceptance

Done when: pointing Jan.ai at the proxy produces responses that survive upstream cuts,
stalls, and 429s with no visible break in the text and no duplicated fragments; all
integration tests in §14.2 pass; and the give-up path logs clearly in proxy output without
annotating the chat message.
