# llm-proxy

A local resilient HTTP proxy in Rust that sits between an OpenAI-compatible client (such as [Jan.ai](https://jan.ai)) and an OpenAI-compatible upstream (e.g. Google's Gemini OpenAI compatibility endpoint or local models), ensuring streaming SSE responses survive transient upstream failures (TCP resets, mid-stream disconnects, HTTP 429/5xx, and stalls).

---

## Features

- **Downstream Stream Ownership:** Maintains the downstream SSE connection uninterrupted while upstream attempts fail and retry behind the scenes.
- **Transparent Resumption & Deduplication:** Re-issues requests with partial output context, deduplicates overlapping tokens using character-boundary matching, and continues streaming to the client.
- **Tool-Call Buffering:** Safely buffers partial tool call argument fragments and emits only fully-assembled tool calls upon `finish_reason: "tool_calls"`.
- **Reasoning Content Support:** Forwards reasoning/thinking deltas downstream while excluding them from continuation context.
- **Configurable Timeouts & Backoff:** Distinguishes first-chunk vs idle timeouts, respects `Retry-After` headers, and applies exponential backoff with jitter and SSE keepalive comments.
- **Transparent Endpoints:** Supports `POST /v1/chat/completions`, `GET /v1/models`, and `GET /health`.

---

## Continuation Modes

The proxy supports two continuation modes for resuming broken streams:

- **`prefix` (`ContinuationMode::Prefix`, default):** Appends the accumulated partial response as a trailing assistant message in the conversation context. Upstream models that support prefix continuation continue generating smoothly from the cutoff point.
- **`instructed` (`ContinuationMode::Instructed`):** Appends the partial assistant turn followed by an explicit instruction turn requesting continuation without repetition. Used as a fallback when an upstream model restarts or errors on trailing assistant turns. May produce visible seams and relies on the overlap deduplication window.

---

## Out of Scope

The following capabilities are explicitly out of scope for this proxy:

- Multiple completion choices (`n > 1`)
- Multimodal inputs beyond transparent JSON passthrough
- Legacy `/v1/completions` endpoint (only `/v1/chat/completions` is supported)
- Embeddings endpoints (`/v1/embeddings`)
- Multiple upstreams, routing, or upstream failover
- Request caching
- Rate-limit enforcement
- Native Anthropic or Gemini APIs (only OpenAI-compatible surfaces are supported)

---

## Configuration

Configuration is loaded from `--config <path>`, `$LLM_PROXY_CONFIG`, or defaults to `~/.config/llm-proxy/llm-proxy.toml`. Every option is overridable via environment variables (`LLM_PROXY_<SECTION>_<KEY>`).

```toml
[server]
bind = "127.0.0.1:8787"
max_body_bytes = 33554432           # 32 MiB (axum's 2 MiB default rejects long conversations)

[upstream]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key_env = "GEMINI_API_KEY"     # Read key from this environment variable
connect_timeout_secs = 10

[resume]
enabled = true
mode = "prefix"                     # "prefix" | "instructed"
max_attempts = 6
max_wall_clock_secs = 900
backoff_initial_ms = 500
backoff_max_ms = 30000
backoff_multiplier = 2.0
backoff_jitter = 0.25
dedup_window_chars = 240
first_chunk_timeout_secs = 120
idle_timeout_secs = 45
resume_on_length_finish = false

[tools]
buffer_tool_calls = true
disable_resume_for_tool_responses = false

[log]
level = "info"
log_bodies = false
```

---

## Building & Testing

```bash
# Build the proxy and mock upstream binary
cargo build

# Run all unit and integration tests
cargo test
```

### Test layout

| Suite | Upstream | Covers |
|---|---|---|
| `src/**` unit tests | none | Config parsing and the full env-override matrix, backoff, dedup, chunk rewriting, error classification, tool-call assembly |
| `tests/integration_tests.rs` | `src/mock.rs` (axum) | End-to-end acceptance behaviour: cuts, dedup, stalls, `Retry-After`, give-up, chunk identity, tool calls, non-streaming, the complete retryable/fatal status matrices, SSE response headers, and CORS |
| `tests/raw_upstream_tests.rs` | raw TCP socket | Behaviour the axum mock cannot express: real TCP resets, incomplete chunked messages, connections closed before the response head, TLS failures, malformed/heartbeat SSE framing, interleaved tool-call fragments, and abort-on-client-disconnect |

The axum mock always terminates its response body cleanly, so the proxy only ever
sees a well-formed EOF from it. `tests/raw_upstream_tests.rs` drives a hand-rolled
HTTP/1.1 server so the wire failures in §2/§7 of the spec (RST, broken pipe,
incomplete message, TLS error) are exercised as they actually occur.
