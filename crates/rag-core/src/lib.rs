//! Shared engine for local semantic code search: config, gitignore-aware
//! walking, blake3 manifests, tree-sitter chunking, FastEmbed embeddings and
//! LanceDB hybrid (vector + BM25) search.

pub mod chunker;
pub mod config;
pub mod embedder;
pub mod engine;
pub mod lock;
pub mod manifest;
pub mod paths;
pub mod store;
pub mod sync;
pub mod walker;

pub use config::{AppConfig, ExecutionProvider, IndexConfig};
pub use engine::{Engine, IndexInfo, IndexStatus, SaveResult, SearchOutcome};
pub use paths::Paths;
pub use store::SearchHit;
pub use sync::{Phase, Progress, ProgressFn, SyncOptions, SyncReport};

/// Error marker for user-requested cancellation (`err.is::<Cancelled>()`).
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("operation cancelled")]
pub struct Cancelled;
