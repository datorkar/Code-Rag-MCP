//! Dynamic batching for ONNX inference: chunks are sorted by length and grouped
//! so that `batch_len x longest_chunk_tokens` stays under a token budget. Short
//! chunks get large batches, long chunks small ones; padding and peak memory
//! stay bounded.

use std::ops::Range;

/// Rough token estimate for code (~3 chars/token), capped at the model's max length.
pub fn est_tokens(chars: usize, max_len: usize) -> usize {
    (chars / 3 + 8).min(max_len).max(1)
}

/// End (exclusive) of the batch starting at `start`. `tokens` must be sorted
/// ascending, so the last element of a batch is its longest. Always >= start+1.
pub fn next_batch_end(tokens: &[usize], start: usize, budget: usize, max_batch: usize) -> usize {
    let max_batch = if max_batch == 0 { usize::MAX } else { max_batch };
    let mut end = start + 1;
    while end < tokens.len() && end - start < max_batch && (end - start + 1) * tokens[end] <= budget {
        end += 1;
    }
    end.min(tokens.len())
}

/// Whole plan for a fixed budget (used by tests / diagnostics).
pub fn plan_batches(tokens: &[usize], budget: usize, max_batch: usize) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let e = next_batch_end(tokens, i, budget, max_batch);
        out.push(i..e);
        i = e;
    }
    out
}

pub const GPU_TOKEN_BUDGET: usize = 32_768;
pub const CPU_TOKEN_BUDGET: usize = 8_192;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_chunks_get_big_batches_long_ones_small() {
        let mut t = vec![100; 300];
        t.extend(vec![1000; 40]);
        let b = plan_batches(&t, 32_768, 0);
        assert_eq!(b[0], 0..300); // 300 short chunks fit in one batch (300*100 <= 32768)
        // long chunks: 32 per batch
        assert!(b[1..].iter().all(|r| r.len() <= 32));
        assert_eq!(b.iter().map(|r| r.len()).sum::<usize>(), 340);
        // cap respected
        assert!(plan_batches(&t, 32_768, 16).iter().all(|r| r.len() <= 16));
        // a single over-budget chunk still forms its own batch
        assert_eq!(plan_batches(&[50_000], 32_768, 0), vec![0..1]);
        assert!(plan_batches(&[], 10, 0).is_empty());
    }
}
