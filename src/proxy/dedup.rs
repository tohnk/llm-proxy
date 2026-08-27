use tracing::debug;

/// Finds the length of the longest overlapping suffix of `tail` that matches a prefix of `pending`.
/// Returns the number of characters in `pending` that overlap with `tail`.
pub fn find_overlap_chars(tail: &[char], pending: &[char]) -> usize {
    if tail.is_empty() || pending.is_empty() {
        return 0;
    }

    let max_k = tail.len().min(pending.len());
    for k in (1..=max_k).rev() {
        let tail_slice = &tail[tail.len() - k..];
        let pending_slice = &pending[..k];
        if tail_slice == pending_slice {
            return k;
        }
    }

    0
}

/// Deduplicates `pending` against `accumulated_text`.
/// Returns `(overlap_chars_count, remaining_string)`.
pub fn deduplicate(accumulated_text: &str, pending: &str, window_chars: usize) -> (usize, String) {
    if pending.is_empty() {
        return (0, String::new());
    }

    let mut tail: Vec<char> = accumulated_text.chars().rev().take(window_chars).collect();
    tail.reverse();
    let pending_chars: Vec<char> = pending.chars().collect();

    let overlap = find_overlap_chars(&tail, &pending_chars);

    let remaining_chars: String = pending_chars[overlap..].iter().collect();
    (overlap, remaining_chars)
}

/// Which strategy the current retry attempt is resolving under.
#[derive(Debug, PartialEq, Eq)]
enum Phase {
    /// First attempt, or dedup already resolved: forward everything.
    Passthrough,
    /// Collecting the first window to decide restart vs. continuation.
    Deciding,
    /// A full restart was detected; consuming the replayed prefix.
    SkippingReplay,
}

/// Buffers and deduplicates the opening of a resumed attempt.
///
/// Two distinct upstream behaviours have to be handled, and §6.3's tail window
/// only covers one of them:
///
/// - **Continuation** — the upstream carries on from where it stopped, possibly
///   repeating a little. The overlap sits between the *end* of what was already
///   emitted and the *start* of the retry, so a tail window finds it.
/// - **Full restart** — the upstream ignores the prefix and generates the whole
///   response again. If more than `window_chars` had been emitted, the retry's
///   opening no longer overlaps the tail at all and every previously emitted
///   character would be sent a second time.
///
/// A *head* window separates the two cheaply: if the retry's first
/// `window_chars` match the first `window_chars` already emitted, it is a
/// restart, and the replayed prefix is consumed before forwarding resumes.
/// Continuation is unaffected and still emits after one window.
pub struct DedupBuffer {
    window_chars: usize,
    accumulated: Vec<char>,
    pending: String,
    phase: Phase,
}

impl DedupBuffer {
    pub fn new(accumulated_text: &str, window_chars: usize, is_first_attempt: bool) -> Self {
        if is_first_attempt {
            Self {
                window_chars,
                accumulated: Vec::new(),
                pending: String::new(),
                phase: Phase::Passthrough,
            }
        } else {
            Self {
                window_chars,
                accumulated: accumulated_text.chars().collect(),
                pending: String::new(),
                phase: Phase::Deciding,
            }
        }
    }

    /// Process an incoming content delta.
    /// Returns `Some(text_to_forward)` when text is ready, or `None` while buffering.
    pub fn push_delta(&mut self, delta: &str) -> Option<String> {
        match self.phase {
            Phase::Passthrough => {
                if delta.is_empty() {
                    None
                } else {
                    Some(delta.to_string())
                }
            }
            Phase::Deciding => {
                self.pending.push_str(delta);
                let pending: Vec<char> = self.pending.chars().collect();

                if self.accumulated.is_empty() {
                    return Some(self.resolve(pending, false));
                }

                let probe = self.window_chars.min(self.accumulated.len());
                if pending.len() < probe {
                    return None;
                }

                if pending[..probe] == self.accumulated[..probe] {
                    // The retry is replaying from the beginning.
                    if pending.len() < self.accumulated.len() {
                        self.phase = Phase::SkippingReplay;
                        return None;
                    }
                    return Some(self.resolve(pending, true));
                }

                Some(self.resolve(pending, false))
            }
            Phase::SkippingReplay => {
                self.pending.push_str(delta);
                let pending: Vec<char> = self.pending.chars().collect();
                if pending.len() < self.accumulated.len() {
                    return None;
                }
                Some(self.resolve(pending, true))
            }
        }
    }

    /// Flush whatever is buffered when the upstream stream ends.
    pub fn finish(&mut self) -> Option<String> {
        if self.phase == Phase::Passthrough {
            return None;
        }
        let pending: Vec<char> = self.pending.chars().collect();

        // A restart that died partway through its replay produced nothing new.
        if !pending.is_empty()
            && pending.len() < self.accumulated.len()
            && self.accumulated.starts_with(&pending)
        {
            self.phase = Phase::Passthrough;
            self.pending.clear();
            debug!(
                replayed_chars = pending.len(),
                "Retry replayed a prefix and ended; nothing new to forward"
            );
            return None;
        }

        let replayed =
            pending.len() >= self.accumulated.len() && pending.starts_with(&self.accumulated[..]);
        let out = self.resolve(pending, replayed);
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Settle this attempt's overlap and return the text to forward.
    fn resolve(&mut self, pending: Vec<char>, full_replay: bool) -> String {
        self.phase = Phase::Passthrough;
        self.pending.clear();

        if full_replay && pending.starts_with(&self.accumulated[..]) {
            debug!(
                replayed_chars = self.accumulated.len(),
                "Upstream restarted from the beginning; dropped the replayed prefix"
            );
            return pending[self.accumulated.len()..].iter().collect();
        }

        let tail_start = self.accumulated.len().saturating_sub(self.window_chars);
        let overlap = find_overlap_chars(&self.accumulated[tail_start..], &pending);
        debug!(
            overlap_chars = overlap,
            "Dedup evaluated on retry continuation"
        );
        pending[overlap..].iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_overlap() {
        let (overlap, out) = deduplicate("Hello world!", "Goodbye world!", 240);
        assert_eq!(overlap, 0);
        assert_eq!(out, "Goodbye world!");
    }

    #[test]
    fn test_full_overlap() {
        let (overlap, out) = deduplicate("Hello world!", "Hello world!", 240);
        assert_eq!(overlap, 12);
        assert_eq!(out, "");
    }

    #[test]
    fn test_partial_overlap() {
        let (overlap, out) =
            deduplicate("The quick brown fox ", "fox jumps over the lazy dog", 240);
        assert_eq!(overlap, 4); // "fox "
        assert_eq!(out, "jumps over the lazy dog");
    }

    #[test]
    fn test_multibyte_character_boundary() {
        let tail = "🦀 Rust is fast! 🚀🌟";
        let pending = "🚀🌟 And reliable! 🦀";
        let (overlap, out) = deduplicate(tail, pending, 240);
        assert_eq!(overlap, 2); // "🚀🌟"
        assert_eq!(out, " And reliable! 🦀");
    }

    #[test]
    fn test_overlap_longer_than_window() {
        // If window is 5 chars, tail is only last 5 chars "67890"
        let tail = "1234567890";
        let pending = "67890abcdef";
        let (overlap, out) = deduplicate(tail, pending, 5);
        assert_eq!(overlap, 5); // "67890"
        assert_eq!(out, "abcdef");
    }

    #[test]
    fn test_empty_pending() {
        let (overlap, out) = deduplicate("Some text", "", 240);
        assert_eq!(overlap, 0);
        assert_eq!(out, "");
    }

    #[test]
    fn test_empty_accumulated() {
        let (overlap, out) = deduplicate("", "Some pending text", 240);
        assert_eq!(overlap, 0);
        assert_eq!(out, "Some pending text");
    }

    #[test]
    fn test_dedup_buffer_first_attempt() {
        let mut buf = DedupBuffer::new("", 10, true);
        assert_eq!(buf.push_delta("hello"), Some("hello".to_string()));
        assert_eq!(buf.push_delta(" world"), Some(" world".to_string()));
        assert_eq!(buf.finish(), None);
    }

    #[test]
    fn test_dedup_buffer_retry_attempt_buffered() {
        let accumulated = "Counting: 1, 2, 3, 4, 5, ";
        let mut buf = DedupBuffer::new(accumulated, 10, false);
        // "4, 5, " overlaps with end of accumulated
        assert_eq!(buf.push_delta("4, "), None);
        assert_eq!(buf.push_delta("5, 6, "), None);
        assert_eq!(buf.push_delta("7, 8"), Some("6, 7, 8".to_string()));
        assert_eq!(buf.push_delta(", 9"), Some(", 9".to_string()));
    }

    /// The case the tail window alone cannot see: more than `window_chars` were
    /// emitted, and the retry starts the whole response again.
    #[test]
    fn test_dedup_buffer_detects_restart_beyond_the_window() {
        let accumulated: String = (1..=60).map(|i| format!("Line {i}\n")).collect();
        assert!(accumulated.chars().count() > 240);

        let mut buf = DedupBuffer::new(&accumulated, 240, false);

        // The upstream replays everything, then continues.
        let mut emitted = String::new();
        for i in 1..=60 {
            if let Some(out) = buf.push_delta(&format!("Line {i}\n")) {
                emitted.push_str(&out);
            }
        }
        assert!(
            emitted.is_empty(),
            "the replayed prefix must not be forwarded, got {emitted:?}"
        );

        for i in 61..=62 {
            if let Some(out) = buf.push_delta(&format!("Line {i}\n")) {
                emitted.push_str(&out);
            }
        }
        assert_eq!(emitted, "Line 61\nLine 62\n");
        assert_eq!(buf.finish(), None);
    }

    /// A retry that matches the head window but then diverges must not have its
    /// new text dropped. Divergence is only knowable once enough characters have
    /// arrived to rule the replay out, so this resolves late — either when the
    /// count is reached, or at end of stream.
    #[test]
    fn test_dedup_buffer_diverging_replay_falls_back_to_tail_matching() {
        let accumulated = "aaaabbbbccccdddd"; // 16 chars

        // Resolved once pending reaches the accumulated length.
        let mut buf = DedupBuffer::new(accumulated, 4, false);
        assert_eq!(
            buf.push_delta("aaaaZZZZ"),
            None,
            "still indistinguishable from a replay in progress"
        );
        assert_eq!(
            buf.push_delta("ZZZZZZZZ"),
            Some("aaaaZZZZZZZZZZZZ".to_string()),
            "divergent text must be forwarded, not dropped"
        );

        // Resolved at end of stream, before the count is reached.
        let mut buf = DedupBuffer::new(accumulated, 4, false);
        assert_eq!(buf.push_delta("aaaaZZZZ"), None);
        assert_eq!(buf.finish(), Some("aaaaZZZZ".to_string()));
    }

    #[test]
    fn test_dedup_buffer_retry_replaying_short_accumulated() {
        // The probe is capped at what was actually emitted, so a six-character
        // prefix settles this immediately instead of waiting for the window.
        let mut buf = DedupBuffer::new("Hello ", 50, false);
        assert_eq!(buf.push_delta("Hello world"), Some("world".to_string()));
        assert_eq!(buf.finish(), None);
    }

    #[test]
    fn test_dedup_buffer_retry_replay_that_dies_early_emits_nothing() {
        let mut buf = DedupBuffer::new("Hello world, and more", 4, false);
        assert_eq!(buf.push_delta("Hello"), None);
        assert_eq!(buf.finish(), None, "a truncated replay is all duplicate");
    }
}
