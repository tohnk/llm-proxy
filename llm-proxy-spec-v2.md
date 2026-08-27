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
> 7. `ContinuationMode::Instructed` is now the **default**, because the default upstream
>    rejects the trailing assistant turn that `Prefix` builds (§2.1).
> 8. Give-up **surfaces the upstream error** when nothing was ever emitted, instead of a
>    silent empty response (§7).
> 9. Downstream `finish_reason` is **normalised** to the OpenAI set, and a finish reason
>    reporting a failed generation is **retried** (§7.4).
> 10. The wall clock **extends** while the upstream reports only congestion (§7.3).
> 11. Retrying **stops early** when the same cause repeats without making progress (§7.5).
> 12. An error reported **inside** a 200 SSE stream is classified rather than ignored (§7.6).
> 13. A failed generation **discards** its partial tool state before any flush, and passthrough
>     never forwards a mixed source chunk (§9).
> 14. Configuration is **strict**: unknown keys are rejected and an env-selected path must
>     exist (§4).
> 15. The backoff before a retry that **cannot happen** is skipped (§6.2).
> 16. CORS is **closed by default**; browser origins must be listed explicitly (§4).
> 17. Dedup detects a **full upstream restart** via a head window, not only a tail one (§6.3).
> 18. Upstream output and accumulated state are **byte-bounded** (§4).
> 19. Reasoning is **deduplicated** across attempts, and "nothing emitted" counts every
>     payload, not only content (§9, §7).
> 20. Downstream sends are **deadline-bounded**, and admission is **capped** (§4, §11).
> 21. `delta.refusal` is **forwarded** and counts as output (§9).
> 22. Untrimmed single-field chunks are **passed through**, keeping upstream metadata (§6.5).
> 23. Configuration is **validated** at startup, and body-extraction failures are
>     **OpenAI-shaped** (§4, §5).
> 24. Reported usage is one attempt's **coherent snapshot**, not a blend (§6.1).
> 25. An assembled tool call that is **unusable** is a failed generation, not a result (§9).
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
   - **`ContinuationMode::Prefix`:** Sends partial output in a trailing `assistant` message. If the model continues smoothly from that prefix, this mode gives zero-seam resumption.
   - **`ContinuationMode::Instructed` (default):** For upstreams that do not support trailing assistant turns, or that restart from the beginning, appends the assistant turn followed by an explicit instruction turn to continue without repetition. Any seam is absorbed by the overlap deduplication window (§6.3).

Both modes are supported behind the `resume.mode` configuration option.

### 2.1 Verified upstream behaviour — *Amended (2026-08-20)*

The verification this section calls for has now been performed against the default
upstream, and it settles the default.

**Google's OpenAI-compatibility endpoint rejects `Prefix` continuations outright.** A
request whose `messages` end with an assistant turn returns:

```
400 INVALID_ARGUMENT — "Requests ending with a model turn are not supported."
```

Confirmed 2026-08-20 against `models/gemini-3.7-flash` by sending both shapes directly:
`[user, assistant]` fails with the above; `[user, assistant, user("Continue…")]` succeeds
and continues cleanly — given `"1, 2, 3,"` it returned `" 4, 5, 6"`, with no repetition,
restatement, or acknowledgement of the instruction.

Because the endpoint accepts the first attempt and only rejects the *continuation*, the
failure surfaces exactly when the proxy matters most: the first dropped stream, on the
retry that was supposed to rescue it. `Prefix` is therefore not merely suboptimal here —
it silently disables the proxy's entire purpose against its own default upstream.

**`Instructed` is consequently the default.** A default `resume.mode` that cannot resume
against the default `upstream.base_url` is not a usable default. Deployments pointed at an
upstream that does support trailing assistant turns may set `mode = "prefix"` for zero-seam
resumption.

The trailing role is a compatibility contract rather than an internal detail, so it is
pinned by test: `test_instructed_continuation_never_ends_with_an_assistant_turn` and
`test_default_continuation_mode_is_compatible_with_default_upstream` assert the exact
`messages` array each mode sends upstream.

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
allowed_origins = []                # browser origins; empty = no CORS headers at all
max_concurrent_requests = 64        # chat completions in flight; beyond this, 503 (§4)

[upstream]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key_env = "GEMINI_API_KEY"     # read the key from this env var (recommended)
api_key = ""                        # optional: direct key override in config / env (LLM_PROXY_UPSTREAM_API_KEY)
connect_timeout_secs = 10
max_response_bytes = 33554432       # 32 MiB cap on a non-streaming / models / error body

[resume]
enabled = true
mode = "instructed"                 # "prefix" | "instructed" — see §2.1
max_attempts = 6                    # total upstream attempts per downstream request
max_wall_clock_secs = 900           # hard ceiling across all attempts
overloaded_max_wall_clock_secs = 300 # extended ceiling while only congestion is failing (§7.3)
backoff_initial_ms = 500
backoff_max_ms = 30000
backoff_multiplier = 2.0
backoff_jitter = 0.25               # ±25% randomisation
dedup_window_chars = 240            # see §6.3
first_chunk_timeout_secs = 120      # timeout until first content, tool-call, or reasoning chunk
idle_timeout_secs = 45              # no chunk for this long => treat as break
resume_on_length_finish = false     # whether finish_reason "length" is treated as resumable
max_no_progress_attempts = 3        # stop after this many identical, fruitless failures (§7.5)
max_accumulated_bytes = 33554432    # 32 MiB cap on accumulated response text

[tools]
buffer_tool_calls = true            # assemble tool call fragments before emitting (§9)
max_tool_call_bytes = 4194304       # 4 MiB cap on one call's assembled arguments
disable_resume_for_tool_responses = false

[log]
level = "info"
log_bodies = false                  # request/response bodies at trace level; off by default
```

**Validation — *Amended (2026-08-24)*.** Values that parse but cannot work are refused at
startup: a zero `max_attempts` (which would emit an empty successful stream without ever
contacting the upstream), zero sizes or timeouts, a `backoff_multiplier` below 1 or
non-finite, a `backoff_jitter` outside `0..=1`, a `backoff_max_ms` below the initial backoff,
a `base_url` that is not an http/https URL, and an API key that cannot be encoded as a header.
Durations are capped at one day, which keeps deadline arithmetic clear of `Instant` overflow.

**Strictness — *Amended (2026-08-24)*.** Unknown keys are rejected in every section: a
typo such as `max_attemps` would otherwise silently use the default for a security, timeout
or endpoint setting. A config path named through `$LLM_PROXY_CONFIG` is treated as explicit,
so a path that does not exist is an error rather than a silent fall back to defaults; only
the implicit `~/.config/...` path may be absent.

`base_url` is fully configurable, and any query string on it is preserved when the endpoint
path is appended. If `base_url` already ends with the endpoint path *as a whole path segment*
— `.../v1/chat/completions` for the chat endpoint — the path is not appended again, which
lets a `base_url` point directly at a single endpoint. The comparison is made on `/`
boundaries, so a base merely *ending with* those characters (`.../v1/mock_models` for the
`models` path) still has the path appended.

**CORS — *Amended (2026-08-24)*.** No CORS headers are emitted unless
`server.allowed_origins` lists an origin. Jan.ai and comparable desktop clients issue native
HTTP requests and send no `Origin`, so they never needed them; permissive headers only
widened the surface, letting any page a browser happens to load reach the loopback listener
and spend the configured account's quota. Browser-based clients opt in by listing their
origin, which also grants `GET`/`POST`/`OPTIONS` and the `content-type`/`authorization`
headers.

**Resource bounds — *Amended (2026-08-24)*.** Timeouts bound how *long* an upstream may
take, not how much it may send, so a fast malfunctioning or hostile upstream could exhaust
memory well inside every deadline. Three independent caps close that, all set far above any
legitimate workload so they never truncate real output:

| Setting | Default | Bounds |
|---|---|---|
| `upstream.max_response_bytes` | 32 MiB | a non-streaming body, a `/v1/models` body, or an error body read before classification |
| `resume.max_accumulated_bytes` | 32 MiB | text accumulated across all attempts of one response — roughly eight million tokens |
| `tools.max_tool_call_bytes` | 4 MiB | assembled arguments of a single buffered tool call, which never reach `accumulated_text` |

Exceeding the accumulation cap ends the stream with `finish_reason: "length"`, which reports
the truncation honestly rather than presenting it as a clean stop. Exceeding the tool-call
cap discards the call and retries it as a failed generation (§7.4), since truncated arguments
cannot be valid JSON.

---

## 5. Endpoints

| Route | Behaviour |
|---|---|
| `POST /v1/chat/completions` | Core path. §6 if `stream: true`, §10 if not. |
| `GET /v1/models` | Transparent proxy to upstream, single retry on 5xx/429/transport error, honouring `Retry-After`. Passes raw upstream body through. Both attempts together are bounded by `first_chunk_timeout_secs`; on expiry the proxy returns `504` with an OpenAI-shaped body rather than hanging. |
| `GET /health` | `200 {"status":"ok"}`. No upstream call. |
| anything else | `404` with an OpenAI-shaped error body. |

**Request rejection — *Amended (2026-08-24)*:** malformed JSON, a missing or wrong content
type, and an oversized body are all rejected by the extractor before the handler runs. Those
rejections are mapped onto the OpenAI error schema rather than returned as axum's plain-text
defaults, so a client parses a failure the same way whatever produced it.

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

**Usage accounting — *Amended (2026-08-24)*.** Upstream usage is cumulative *within* an
attempt, so merging fields individually across attempts produced objects that never existed —
a `prompt_tokens` from one attempt beside a `completion_tokens` from another. Whole snapshots
are stored instead, which also preserves detail fields (`prompt_tokens_details`, cached and
reasoning token counts) that a field-by-field copy discarded.

What is reported downstream is the **final attempt's** snapshot: the numbers an unproxied
request would have returned, which is what a client showing context usage expects. Summing
across attempts would report several times the prompt for a retried request and read as a
bug. The cross-attempt total is tracked separately and logged as `billed_prompt_tokens` /
`billed_completion_tokens`, since only the proxy knows what was actually spent.

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

**No backoff before a retry that cannot happen — *Amended (2026-08-24)*.** The limits are
tested at the top of the loop, so a naive implementation sleeps a full backoff after the
final failure and only then discovers it had no attempt left. With `backoff_max_ms = 30000`
that is up to 30 s of dead waiting at the end of an already-doomed request. Check whether
another attempt is permitted *before* sleeping.

### 6.3 Overlap deduplication

On every attempt after the first:

1. Do not forward content deltas immediately. Accumulate resumed text into a `pending` buffer.
2. Once `pending.chars().count() >= dedup_window_chars` — or the upstream stream ends, whichever is first — run the match.
3. Let `tail` be the last `dedup_window_chars` of `accumulated_text`. For `k` from `min(tail.len(), pending.len())` down to 1, if `pending` starts with the final `k` characters of `tail`, take the largest `k`, drop the first `k` characters of `pending`, and forward the rest.
4. If no `k` matches, forward `pending` unchanged.
5. From that point on in this attempt, forward deltas as they arrive with no buffering.
6. All comparisons and slicing operate strictly on `char` boundaries. Anything forwarded downstream is appended to `accumulated_text`.

**Full restarts — *Amended (2026-08-24)*.** The procedure above matches the *end* of what was
emitted against the *start* of the retry, which finds the overlap when an upstream continues.
It cannot see a **full restart**: an upstream that ignores the prefix and regenerates the
whole response. Once more than `dedup_window_chars` have been emitted, the retry's opening no
longer touches the tail at all, no `k` matches, and every character emitted so far is
delivered a second time. §2 names this as the very behaviour `Instructed` mode exists to
absorb, so it must actually be absorbed.

A **head** window separates the two cases before step 3:

1. Let `head` be the first `min(dedup_window_chars, accumulated_text.len())` characters of `accumulated_text`.
2. If `pending` begins with `head`, the retry is replaying from the start. Continue buffering until `pending` reaches `accumulated_text.len()`, then drop that whole prefix and forward the remainder.
3. Otherwise apply the tail procedure above, unchanged.

A replay that diverges after matching the head falls back to tail matching rather than
dropping the divergent text, and a replay that ends before completing forwards nothing, since
all of it was duplicate. Continuation — the common case — is unaffected and still resolves
after one window; only a detected restart buffers further, and only up to what was already
emitted.

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

***Amended (2026-08-24)*:** where a delta carries exactly one payload field and dedup did not
shorten it, the upstream chunk is forwarded with the fields above rewritten in place, rather
than being rebuilt from scratch. Reconstruction silently discarded everything the proxy does
not model — `logprobs`, `system_fingerprint`, `service_tier`, and provider extensions such as
Gemini's `extra_content` — on every chunk of a clean first attempt. A chunk that *was*
rewritten (dedup trimmed it, or the delta mixes fields that are emitted separately) is still
rebuilt, since the surrounding metadata no longer describes what is being sent.

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

**Terminal:** `finish_reason` of `stop`, `tool_calls`, `content_filter`, or `length` when `resume_on_length_finish = false`. Upstream values outside this set are normalised before being forwarded (§7.4).

**Retryable:**
- HTTP 408, 409, 425, 429, 500, 502, 503, 504
- Transport errors: connection reset, broken pipe, incomplete message, TLS error, connect timeout, read timeout
- `first_chunk_timeout` or `idle_timeout` expiry (§8)
- Upstream stream ending with no `finish_reason` and no `[DONE]`
- `finish_reason: "length"` when `resume_on_length_finish = true`
- A `finish_reason` reporting a *failed* generation rather than a finished one (§7.4)

**Fatal:**
- HTTP 400, 401, 403, 404, 413, 422
- Any error response whose body or error message indicates a context-length exceeded or content-policy/safety violation (regardless of HTTP status code) — **except on 429, see below**
- Unrecoverable client errors

**`resume.enabled = false` Semantics:** If `resume.enabled = false`, mid-stream failures do not attempt continuation; the stream terminates cleanly with a `stop` chunk. Initial pre-stream transient errors (e.g. initial 429/503 before any chunk) still follow normal retry up to `max_attempts`.

**Emission accounting — *Amended (2026-08-24)*.** "Nothing was emitted" means *no payload
frame reached the client*, counted across content, reasoning, and passthrough tool deltas —
not merely that `accumulated_text` is empty. Reasoning and unbuffered tool output reach the
client without touching `accumulated_text`, so keying off content alone would append an error
chunk to a response the user can already see, and would let the no-progress cutoff (§7.5)
stop a request that is visibly producing output.

**Give-up behaviour — *Amended (2026-08-21)*:** When attempts or wall clock are exhausted,
log at error level, then terminate the downstream stream in one of two ways:

- **Some content was emitted.** Emit a clean finish chunk with `finish_reason: "stop"`
  followed by `[DONE]`. Never append error text to a partial answer: the user can read what
  arrived, and corrupting it with a proxy message is worse than a quiet truncation.
- **Nothing was emitted.** Emit the last upstream error as an error chunk, followed by
  `[DONE]`, and no finish chunk. A clean `stop` here would hand the client an empty but
  apparently successful response — indistinguishable from a model that chose to say nothing,
  and strictly less informative than the bare upstream error the client would have seen
  without a proxy in the path. The rule against *synthetic* error text still holds: what is
  forwarded is the upstream's own error, at protocol level, not fabricated assistant content.

### 7.3 Congestion extends the deadline — *Amended (2026-08-21)*

`max_attempts` and `max_wall_clock_secs` are meant to be independent ceilings, but when an
upstream is degraded each failed attempt can cost 15–30 s, so a low attempt count exhausts
the budget while most of the wall clock goes unused. Observed in production against Gemini:
give-up after 6 attempts at 131 s and 120 s, against a 900 s ceiling.

While **every** failure so far has been the upstream reporting congestion, the deadline
becomes `overloaded_max_wall_clock_secs` instead:

```
deadline = if congestion_failures > 0 && other_failures == 0 {
    max(max_wall_clock_secs, overloaded_max_wall_clock_secs)
} else {
    max_wall_clock_secs
}
```

**Congestion** means HTTP **429** or **503** — the two statuses whose defined semantics are
exactly "busy, come back later", and which describe a condition that clears on its own.
Transport errors, timeouts and truncated streams report a fault rather than a queue, so they
return the normal ceiling immediately.

Taking the larger of the two means the setting can only ever extend, never shorten; at the
shipped defaults (900 normal, 300 overloaded) it is inert, and it takes effect once
`max_wall_clock_secs` is tightened to an interactive value.

Recommended pairing: set `max_attempts` high enough that it never binds and let the wall
clock be the single governor.

### 7.6 Errors inside a 200 stream — *Amended (2026-08-24)*

An upstream may report failure as an SSE event rather than a status code:

```
data: {"error":{"message":"quota exhausted"}}
data: [DONE]
```

Such an event carries no `choices` and no delta, so a chunk parser ignores it — and the
`[DONE]` that follows then satisfies §7's terminal rule, turning an upstream failure into a
successful *empty* completion. This is the same silent failure the give-up rules exist to
prevent, arriving by a different route.

Any event with a non-null root `error` is therefore classified before delta extraction:

- **Fatal** if the message indicates a context-length or content-policy failure (§7.1) —
  retrying resends exactly what was refused.
- **Retryable** otherwise, since in-band errors are commonly transient quota or capacity
  problems. A permanent one repeats identically and is caught by the no-progress cutoff
  (§7.5), so this cannot loop.

### 7.5 No-progress cutoff — *Amended (2026-08-24)*

Retrying assumes the next attempt might differ. When it demonstrably does not, retrying
converts a fast, legible failure into a slow one with the same outcome, and burns the
user's time and the account's quota on the way.

An attempt is **stalled** when it fails with the same `Cause` label as the previous attempt
*and* emitted no new characters downstream. After `max_no_progress_attempts` consecutive
stalled attempts, the proxy gives up and surfaces the error under the §7 give-up rules.

Two exemptions keep this from firing on failures that are genuinely worth retrying:

- **Congestion is transparent.** A 429 or 503 neither counts toward the limit nor clears a
  run. It cannot emit content by definition, so counting it would cut off exactly the case
  §7.3 exists to be patient with — but it is equally no evidence that some *other* failure
  has changed, so it must not reset the count either. A degraded upstream interleaving 503s
  with an identical fault would otherwise never reach the limit at all. Congestion remains
  bounded by the wall clock alone.
- **Progress resets the counter.** Repeated mid-stream cuts each advance `accumulated_text`,
  so resumption is working and continues; only a run of attempts that achieve literally
  nothing is stopped.

A different cause also resets the run, since alternating failures indicate a genuinely
unstable upstream rather than a deterministic one.

**Motivating case.** Observed against Gemini with tools enabled: twelve attempts over 240 s,
alternating 503s with `MALFORMED_FUNCTION_CALL`, `emitted_chars=0`. The malformed calls were
identical every time — the failure was in the tool schema being sent, which no retry can
change. Because the malformed failures were interleaved with 503s rather than consecutive, this
only works if congestion is transparent rather than resetting: replaying that exact sequence,
a resetting rule stops at attempt 12 (no better than the wall clock), while a transparent one
stops at attempt 7 — the same error reaching the client either way, roughly three minutes
sooner.

### 7.4 Finish-reason handling — *Amended (2026-08-21)*

Gemini leaks native finish reasons through its compatibility layer — `SAFETY`,
`RECITATION`, `MAX_TOKENS`, and composite forms such as
`function_call_filter: MALFORMED_FUNCTION_CALL`. A client switching on the OpenAI enum has
no case for these, so the value is normalised before it goes downstream (§6.5), while the
upstream original is logged:

| Upstream contains | Downstream value |
|---|---|
| `safety`, `recitation`, `blocklist`, `prohibited`, `spii`, `content_filter` | `content_filter` |
| `tool_call`, `function_call` | `tool_calls` |
| `max_tokens`, `length` | `length` |
| anything else | `stop` |

Separately, `MALFORMED_FUNCTION_CALL` is **retryable**. It reports that the model attempted a
tool call, emitted syntactically invalid output, and had it discarded — so the response is
empty or truncated even though the stream ended without error. Generation is
non-deterministic, so another attempt has a real chance of succeeding; a safety refusal, by
contrast, would fail identically every time and stays terminal.

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
  - ***Amended (2026-08-24)*:** a finish reason reporting a *failed* generation (§7.4) must be
    recognised **before** the flush, not after. `MALFORMED_FUNCTION_CALL` normalises to
    `tool_calls`, so a flush keyed on the normalised value would push the half-assembled call
    downstream and then retry — leaving the client with a broken call followed by its
    replacement, two calls for one intended invocation.
- **Tool Call Passthrough (`buffer_tool_calls = false`):** Emit tool call deltas as they arrive, rewriting IDs and suppressing intermediate finish reasons. ***Amended (2026-08-24)*:** emit a chunk containing *only* the tool calls rather than forwarding the upstream object verbatim. A single upstream event may carry `content` and `reasoning` alongside `tool_calls`, and those are emitted by their own branches; forwarding the mixed source object sends them twice. Note that passthrough has **no equivalent of the §6.3 overlap window**: fragments already forwarded by an attempt that later fails cannot be recalled, so a cut mid-tool-call leaves the client holding the abandoned fragment as well as the complete retry. Buffering (the default) does not have this problem, because nothing is emitted until the call is whole.
- **Refusals — *Amended (2026-08-24)*:** `delta.refusal` is a standard OpenAI field carrying a
  model refusal in place of content. It is forwarded, deduplicated on its own window like
  reasoning, excluded from continuation `messages`, and **counts as meaningful output** —
  without which a refusal-only stream is invisible to the first-chunk watchdog (§8) and is
  retried until give-up despite the upstream having answered.
- **Unusable assembled calls — *Amended (2026-08-24)*:** before flushing, a buffered call must
  have a non-empty `id`, a non-empty function name, and arguments that parse as JSON if
  present. Inventing a placeholder id or an empty name produces a call the client cannot
  execute — and duplicate placeholders collide when several calls are incomplete. An unusable
  call is treated as a failed generation (§7.4) rather than forwarded.
- **Reasoning / Thinking content:** Forward `delta.reasoning_content` / `delta.reasoning` downstream immediately, but **exclude from `accumulated_text` and continuation `messages`**.
  - ***Amended (2026-08-24)*:** reasoning gets its **own overlap window**, held in
    `accumulated_reasoning`. Excluding it from `accumulated_text` means an attempt that dies
    after reasoning but before any content leaves the continuation empty, so the retry is a
    plain re-request and the upstream thinks aloud again — the client sees the reasoning
    twice. Deduplicating it separately preserves the §9 rule (it still never enters
    continuation `messages`) while keeping the visible stream continuous.
- **`disable_resume_for_tool_responses = true`:** Requests with a `tools` array skip resumption if a failure occurs after text/content has begun emitting.

---

## 10. Non-streaming requests

`stream: false` issues the request, classifies errors with §7 rules, and retries on `Retryable`. On success, passes the raw upstream response status and body through verbatim (preserving original key order and types). On give-up or fatal error, returns the upstream error status and body.

---

## 11. Client disconnect

When the downstream client disconnects (detected via mpsc channel send failure), immediately abort in-flight upstream requests and cleanly terminate the spawned task.

**Clients that stop reading — *Amended (2026-08-24)*.** Disconnection is not the only way a
client stops consuming. One that stays connected but never reads fills the socket buffers and
then the bounded channel, at which point `send` parks waiting for capacity — and every other
deadline wraps an *upstream* wait, not that one, so the task and its upstream connection are
retained indefinitely. Every downstream send is therefore bounded by the request deadline as
well as by channel closure; exceeding it abandons the request exactly as a disconnect does.

**Admission control — *Amended (2026-08-24)*.** At most
`server.max_concurrent_requests` chat completions run at once. Excess requests are refused
with `503` rather than queued: a queued request still holds its parsed body and waits behind
work that may run for the full wall clock. Each admitted request holds its permit for the
whole of its life, including the spawned streaming task.

---

## 12. Logging

- `tracing` + `tracing-subscriber`, `EnvFilter` with `RUST_LOG` or configured log level.
- Info: request start, attempt boundaries, retries with cause/backoff, completion stats including `emitted_chars`, so a give-up can be told apart from a partial answer without guesswork.
- Debug: dedup results (overlap character count).
- Trace: request and response bodies (gated by `log_bodies = true`).
- Redact Authorization headers; never log secret API keys.
- ***Amended (2026-08-24)*:** the upstream URL is redacted before logging — query strings and
  userinfo both carry credentials for some providers.
- `log_bodies = true` only gates the `trace!` calls; the filter must also admit them. Warn at
  startup when the configured level would discard the traces it was just asked to emit.

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
