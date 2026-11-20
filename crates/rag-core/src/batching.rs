//! Dynamic batching for ONNX inference: chunks are sorted by length and grouped
//! so that `batch_len x longest_chunk_tokens` stays under a token budget. Short
//! chunks get large batches, long chunks small ones; padding and peak memory
//! stay bounded.

use std::ops::Range;

/// Rough token estimate for code (~3 chars/token), capped at the model's max length.
/// Concurrent embedding sessions on GPU (overlap CPU tokenization with GPU work).
pub const GPU_WORKERS: usize = 2;

pub fn est_tokens(chars: usize, max_len: usize) -> usize {
    (chars / 3 + 8).min(max_len).max(1)
}

/// End (exclusive) of the batch starting at `start`: grows while
/// `batch_len x longest_chunk` stays within `budget`. Works for any order
/// (the app sorts longest-first). Always >= start+1.
pub fn next_batch_end(tokens: &[usize], start: usize, budget: usize, max_batch: usize) -> usize {
    let max_batch = if max_batch == 0 { usize::MAX } else { max_batch };
    let mut longest = tokens[start];
    let mut end = start + 1;
    while end < tokens.len() && end - start < max_batch {
        let l = longest.max(tokens[end]);
        if (end - start + 1) * l > budget {
            break;
        }
        longest = l;
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

pub const GPU_TOKEN_BUDGET: usize = 65_536;
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
        // longest-first order gives the same batch sizes, mirrored
        let mut d = t.clone();
        d.reverse();
        let bd = plan_batches(&d, 32_768, 0);
        assert!(bd[..bd.len() - 1].iter().all(|r| r.len() <= 32));
        assert_eq!(bd.iter().map(|r| r.len()).sum::<usize>(), 340);
        assert_eq!(bd.last().unwrap().len(), 276); // 8 leftover long chunks share a batch with 24 short ones
        // a single over-budget chunk still forms its own batch
        assert_eq!(plan_batches(&[50_000], 32_768, 0), vec![0..1]);
        assert!(plan_batches(&[], 10, 0).is_empty());
    }
}
