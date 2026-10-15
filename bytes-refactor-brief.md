# Task brief: convert the llm-proxy downstream channel from `String` to `Bytes`

You are making a **type-only refactor** to a Rust SSE proxy. Read
`llm-proxy-spec-v2.md` first — especially **§3.1 Downstream channel contract**,
which lists the properties this change must preserve, and §6.6, which defines the
exact wire framing.

This is a tidying change with no expected performance impact. It is being made on
the most safety-critical path in the codebase, so **conservatism matters far more
than elegance**. A minimal, boring diff is the goal.

---

## The single most important rule

**Every existing test must pass without being modified**, with exactly one
permitted exception named in §5 below.

A correct version of this refactor produces **byte-identical output on the wire**.
The test suite asserts on that output. Therefore:

- If a test fails, **your code is wrong** — the test is not stale.
- If you find yourself editing a test to make it pass, **stop and report instead**.
- Do not add, delete, rename, or re-scope any test.

---

## 1. What to change

Replace the downstream SSE channel payload type `String` with `bytes::Bytes` in
the **proxy** (not the mock — see §2).

Add to `Cargo.toml` under `[dependencies]`:

```toml
bytes = "1"
```

It is already in `Cargo.lock` at 1.12.1 as a transitive dependency, so this
compiles nothing new.

### `src/proxy/chunks.rs` — the three formatting helpers

```rust
// before
pub fn format_sse_data(json: &Value) -> String
pub fn format_sse_done() -> &'static str
pub fn format_sse_comment(comment: &str) -> String

// after
pub fn format_sse_data(json: &Value) -> Bytes
pub fn format_sse_done() -> Bytes          // Bytes::from_static(b"data: [DONE]\n\n")
pub fn format_sse_comment(comment: &str) -> Bytes
```

`format_sse_data` currently writes compact JSON through `Value`'s `Display` impl
into a `String`, in one allocation. Preserve that property: serialise into a
`BytesMut` with `serde_json::to_writer` (which takes any `io::Write`, and
`&mut BytesMut` qualifies via `bytes::BufMut`), then `.freeze()`. Keep the
`with_capacity(64)` hint.

### `src/lib.rs` — one site

```rust
let (tx, rx) = mpsc::channel::<String>(64);   // -> mpsc::channel::<Bytes>(64)
```

The `ReceiverStream::new(rx).map(Ok::<_, Infallible>)` bridge and
`Body::from_stream` call need no change — `Body::from_stream` accepts any item
type implementing `Into<Bytes>`, which now holds trivially.

### `src/proxy/stream.rs` — two signatures, thirteen call sites

```rust
downstream_tx: Sender<String>          // -> Sender<Bytes>   (line ~65)
downstream_tx: &Sender<String>         // -> &Sender<Bytes>  (sleep_with_keepalive, ~line 473)
```

Six call sites read `downstream_tx.send(format_sse_done().to_string())`. Drop the
`.to_string()` — that allocation disappearing is the entire point of the change.

### `src/proxy/consume.rs` — one signature, seven call sites

```rust
downstream_tx: &Sender<String>         // -> &Sender<Bytes>  (line ~76)
```

---

## 2. What NOT to change — read this carefully

**`src/mock.rs` uses `mpsc::Sender<String>` at three places (lines ~217, ~259,
~394). Leave every one of them alone.** That is the *mock upstream* — a test
double that plays the role of the remote server. Its channel is a completely
separate thing that happens to share a type. A find-and-replace across the
codebase will convert it, the tests will still pass, and the diff will be twice
the size it should be for no reason. Only the four proxy files listed in §1 change.

Also leave entirely alone, because none of them need to change for a payload-type
swap:

- The watchdog / deadline loop in `consume.rs` (the `'watchdog: loop` block and its `tokio::select!`). This implements §8.1 and is subtle.
- `DedupBuffer` and everything in `src/proxy/dedup.rs`.
- Any other `tokio::select!` arm, in particular every `downstream_tx.closed()` branch.
- Backoff logic, error classification, continuation building, config.
- `build_chunk_with_*` and `rewrite_upstream_chunk` — these return `serde_json::Value` and stay that way. Only the *framing* helpers change type.

**Preserve the existing error-handling shape at each call site exactly.** Some
sends are deliberately `let _ = downstream_tx.send(...).await;` because the code
is about to `break` out of a terminal path anyway; others check `.is_err()` and
return. Do not normalise these to match each other — the asymmetry is intentional.

Keep every `send` **awaited**. Do not convert any of them to `try_send`; the
channel is bounded and that backpressure is load-bearing (§3.1 point 4).

---

## 3. No `unsafe`

There is no `unsafe` in this codebase and this change does not need any. In
particular, do not build the frame by writing bytes through
`String::as_mut_vec()` or `from_utf8_unchecked` — a partially-written buffer
after a serialisation error is undefined behaviour. `BytesMut` has no UTF-8
invariant, so writing into it is safe by construction. That is the whole reason
`Bytes` is a comfortable target here.

---

## 4. Framing must stay byte-identical

Per §6.6, and asserted throughout the suite:

- data frames: `data: {compact-json}\n\n`
- comments:    `: {text}\n\n`
- terminator:  `data: [DONE]\n\n`

Compact JSON only. A newline inside a serialised chunk would split the SSE frame
and corrupt the stream, so do not switch to any pretty-printing writer.

---

## 5. The one permitted test change

`src/proxy/chunks.rs` contains `test_format_sse_data_frames_compact_json`, which
compares the helper's return value against a `String`. It cannot compile once the
return type changes. Adapt it **mechanically and minimally** — compare bytes
instead of a string, keep every assertion it already makes (starts with `data: `,
ends with `\n\n`, byte-equal to the separately-serialised form, exactly two
newlines total).

This is the only test that may be touched. If any other test needs modification,
that is a signal your change altered behaviour: stop and report it.

---

## 6. Verification

```bash
cargo test --all && cargo clippy --all-targets -- -D warnings && cargo fmt --all -- --check
```

All three must be clean. Expected: **119 tests** — 47 unit, 55 in
`tests/integration_tests.rs`, 17 in `tests/raw_upstream_tests.rs`.
`cargo test --all 2>&1 | grep -cE "^test result: ok"` should print `6`.

Several tests are timing-sensitive (backoff escalation, idle and first-chunk
watchdogs, client-disconnect propagation). **Run the suite three times.** A single
green run is not evidence of stability; intermittent failure means the change
affected timing and must be investigated, not re-run until green.

---

## 7. If something looks wrong

If you notice a bug, a redundancy, or a stylistic improvement outside the scope
above: **report it in your summary and do not fix it.** Unrelated changes bundled
into this diff are strictly unwelcome — they make the refactor impossible to
review against the "byte-identical output" property that makes it safe.

---

## 8. Deliverable

- The diff, limited to `Cargo.toml` and the four proxy files in §1.
- Confirmation that all three verification commands passed, and that the suite was run three times.
- A list of any `.to_string()` / allocation sites removed.
- Anything you noticed but deliberately did not change, per §7.
