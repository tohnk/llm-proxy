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

/// State helper for buffering and deduplicating chunks during a retry attempt.
pub struct DedupBuffer {
    window_chars: usize,
    accumulated_tail: Vec<char>,
    pending: String,
    dedup_done: bool,
}

impl DedupBuffer {
    pub fn new(accumulated_text: &str, window_chars: usize, is_first_attempt: bool) -> Self {
        if is_first_attempt {
            Self {
                window_chars,
                accumulated_tail: Vec::new(),
                pending: String::new(),
                dedup_done: true,
            }
        } else {
            // Walk back from the end rather than materialising the whole
            // accumulated response: only the last `window_chars` are ever read,
            // and by a late retry the accumulated text can be very large.
            let mut tail: Vec<char> = accumulated_text.chars().rev().take(window_chars).collect();
            tail.reverse();

            Self {
                window_chars,
                accumulated_tail: tail,
                pending: String::new(),
                dedup_done: false,
            }
        }
    }

    /// Process an incoming content delta.
    /// Returns `Some(text_to_forward)` if text is ready to be emitted downstream, or `None` if buffering.
    pub fn push_delta(&mut self, delta: &str) -> Option<String> {
        if self.dedup_done {
            if delta.is_empty() {
                None
            } else {
                Some(delta.to_string())
            }
        } else {
            self.pending.push_str(delta);
            if self.pending.chars().count() >= self.window_chars {
                Some(self.perform_dedup())
            } else {
                None
            }
        }
    }

    /// Flush any pending text when the upstream stream ends.
    pub fn finish(&mut self) -> Option<String> {
        if !self.dedup_done {
            let out = self.perform_dedup();
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        } else {
            None
        }
    }

    fn perform_dedup(&mut self) -> String {
        self.dedup_done = true;
        let pending_chars: Vec<char> = self.pending.chars().collect();
        let overlap = find_overlap_chars(&self.accumulated_tail, &pending_chars);
        debug!(
            overlap_chars = overlap,
            "Dedup evaluated on retry continuation"
        );
        let remaining: String = pending_chars[overlap..].iter().collect();
        self.pending.clear();
        remaining
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

    #[test]
    fn test_dedup_buffer_retry_attempt_finish_before_window() {
        let accumulated = "Hello ";
        let mut buf = DedupBuffer::new(accumulated, 50, false);
        assert_eq!(buf.push_delta("Hello world"), None);
        assert_eq!(buf.finish(), Some("world".to_string()));
    }
}
