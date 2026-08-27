use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;

pub mod chunks;
pub mod consume;
pub mod continuation;
pub mod dedup;
pub mod nonstream;
pub mod stream;
pub mod tools;

/// Read an upstream response body, refusing to buffer more than `limit` bytes.
///
/// Timeouts bound how *long* an upstream may take, not how much it may send, so
/// without this a fast malfunctioning or hostile upstream can exhaust memory
/// well inside every deadline.
pub async fn read_body_limited(response: reqwest::Response, limit: usize) -> Result<Bytes, String> {
    let mut stream = response.bytes_stream();
    let mut out = BytesMut::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        if out.len().saturating_add(chunk.len()) > limit {
            return Err(format!("upstream response exceeded the {limit}-byte limit"));
        }
        out.extend_from_slice(&chunk);
    }

    Ok(out.freeze())
}
