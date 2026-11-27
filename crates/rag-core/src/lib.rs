//! Shared engine for local semantic code search: config, gitignore-aware
//! walking, blake3 manifests, tree-sitter chunking, FastEmbed embeddings and
//! LanceDB hybrid (vector + BM25) search.

pub mod agents_md;
pub mod boilerplate;
pub mod batching;
pub mod chunker;
pub mod config;
pub mod embedder;
pub mod hardware;
pub mod engine;
pub mod idents;
pub mod lock;
pub mod manifest;
pub mod paths;
pub mod runtime;
pub mod share;
pub mod snippet;
pub mod store;
pub mod sync;
pub mod walker;
#[cfg(feature = "winml")]
pub mod winml;

pub use config::{AppConfig, ExecutionProvider, IndexConfig, ModelVariant, Precision, ResourceMode};
pub use engine::{Engine, IndexInfo, IndexStatus, SaveResult, SearchFilter, SearchOutcome};
pub use paths::Paths;
pub use store::SearchHit;
pub use sync::{PhaseTimings, Phase, Progress, ProgressFn, SyncOptions, SyncReport, SyncTooLarge};

/// Error marker for user-requested cancellation (`err.is::<Cancelled>()`).
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("operation cancelled")]
pub struct Cancelled;
