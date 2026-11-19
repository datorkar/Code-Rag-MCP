//! Global settings + per-index configuration persisted in `config.json`.

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::lock::FileLock;
use crate::paths::{write_atomic, Paths};

pub const DEFAULT_EMBED_MODEL: &str = "jinaai/jina-embeddings-v2-base-code";
/// Jina v2 reranker is trained on code retrieval as well as text.
pub const DEFAULT_RERANKER: &str = "jinaai/jina-reranker-v2-base-multilingual";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionProvider {
    /// Best available GPU (Windows ML vendor EP / DirectML / CUDA, per build), else CPU.
    #[default]
    Auto,
    /// Any GPU (same as Auto, but logs a warning when none is usable).
    Gpu,
    /// NPU via Windows ML (e.g. Qualcomm QNN, Intel NPU); experimental for embeddings.
    Npu,
    Cuda,
    Directml,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// Default model for newly created indexes.
    pub active_embed_model: String,
    pub execution_provider: ExecutionProvider,
    /// Number of chunks per ONNX inference batch.
    pub embed_batch_size: usize,
    /// Cross-encoder used to re-score hybrid results (`None` = disabled).
    #[serde(default = "default_reranker")]
    pub reranker_model: Option<String>,
    /// Hybrid candidates fetched per search before reranking.
    pub rerank_candidates: usize,
    /// Searches only auto-sync when at most this many files changed; larger
    /// updates (and full builds) are left to the GUI / `reindex_path`.
    pub auto_sync_max_files: usize,
    /// Results are trimmed to this many lines around the best-matching region.
    pub snippet_max_lines: usize,
    pub indexes: Vec<IndexConfig>,
}

fn default_reranker() -> Option<String> {
    Some(DEFAULT_RERANKER.into())
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            active_embed_model: DEFAULT_EMBED_MODEL.into(),
            execution_provider: ExecutionProvider::Auto,
            embed_batch_size: 32,
            reranker_model: Some(DEFAULT_RERANKER.into()),
            rerank_candidates: 30,
            auto_sync_max_files: 200,
            snippet_max_lines: 40,
            indexes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct IndexConfig {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub root_path: String,
    pub embed_model: String,
    pub allowed_extensions: Vec<String>,
    pub ignore_dirs: Vec<String>,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    /// Files larger than this are skipped (generated code, amalgamations, ...).
    pub max_file_bytes: u64,
    pub auto_update_on_search: bool,
    pub auto_update_cooldown_hours: f64,
    pub last_indexed_at: Option<DateTime<Utc>>,
    pub chunk_count: u64,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            description: None,
            root_path: String::new(),
            embed_model: DEFAULT_EMBED_MODEL.into(),
            allowed_extensions: vec![".cpp".into(), ".h".into()],
            ignore_dirs: vec![],
            chunk_size: 2000,
            chunk_overlap: 200,
            max_file_bytes: 2 * 1024 * 1024,
            auto_update_on_search: true,
            auto_update_cooldown_hours: 24.0,
            last_indexed_at: None,
            chunk_count: 0,
        }
    }
}

impl IndexConfig {
    /// Validate and normalise user input (extension dots, id charset, overlap bounds).
    pub fn normalize(&mut self) -> Result<()> {
        self.id = self.id.trim().to_string();
        if self.id.is_empty() {
            self.id = slugify(&self.name);
        }
        if self.id.is_empty() {
            bail!("index id must not be empty");
        }
        if !self.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            bail!("index id '{}' may only contain [A-Za-z0-9_-]", self.id);
        }
        if self.name.trim().is_empty() {
            self.name = self.id.clone();
        }
        if self.root_path.trim().is_empty() {
            bail!("root_path must not be empty");
        }
        if self.chunk_size < 100 {
            bail!("chunk_size must be >= 100 characters");
        }
        if self.chunk_overlap >= self.chunk_size {
            bail!("chunk_overlap must be smaller than chunk_size");
        }
        self.allowed_extensions = self
            .allowed_extensions
            .iter()
            .map(|e| e.trim().to_ascii_lowercase())
            .filter(|e| !e.is_empty())
            .map(|e| if e.starts_with('.') { e } else { format!(".{e}") })
            .collect();
        self.allowed_extensions.sort();
        self.allowed_extensions.dedup();
        self.ignore_dirs = self
            .ignore_dirs
            .iter()
            .map(|d| d.trim().trim_matches(['/', '\\']).to_string())
            .filter(|d| !d.is_empty())
            .collect();
        self.ignore_dirs.dedup();
        if self.auto_update_cooldown_hours < 0.0 || !self.auto_update_cooldown_hours.is_finite() {
            self.auto_update_cooldown_hours = 0.0;
        }
        Ok(())
    }

    /// True if the index has never been built or is older than the cooldown.
    pub fn is_stale(&self, now: DateTime<Utc>) -> bool {
        match self.last_indexed_at {
            None => true,
            Some(t) => {
                let age_h = (now - t).num_milliseconds() as f64 / 3_600_000.0;
                age_h >= self.auto_update_cooldown_hours
            }
        }
    }

    /// Settings that invalidate every stored chunk when changed.
    pub fn fingerprint(&self) -> String {
        format!("{}|{}|{}", self.embed_model, self.chunk_size, self.chunk_overlap)
    }
}

pub fn slugify(s: &str) -> String {
    let mut out = String::new();
    for c in s.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_string()
}

impl AppConfig {
    /// Load config.json (returns defaults if it does not exist yet).
    pub fn load(paths: &Paths) -> Result<Self> {
        let file = paths.config_file();
        match std::fs::read(&file) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {}", file.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", file.display())),
        }
    }

    pub fn save(&self, paths: &Paths) -> Result<()> {
        let json = serde_json::to_vec_pretty(self)?;
        write_atomic(&paths.config_file(), &json)
    }

    /// Read-modify-write under a cross-process lock so the GUI and the MCP
    /// server never clobber each other's edits.
    pub fn update<T>(paths: &Paths, f: impl FnOnce(&mut AppConfig) -> Result<T>) -> Result<T> {
        let _guard = FileLock::exclusive_blocking(&paths.lock_file("config"))?;
        let mut cfg = Self::load(paths)?;
        let out = f(&mut cfg)?;
        cfg.save(paths)?;
        Ok(out)
    }

    pub fn index(&self, id: &str) -> Option<&IndexConfig> {
        self.indexes.iter().find(|i| i.id == id)
    }

    pub fn index_mut(&mut self, id: &str) -> Option<&mut IndexConfig> {
        self.indexes.iter_mut().find(|i| i.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_fields_get_defaults() {
        let c: AppConfig = serde_json::from_str(r#"{"execution_provider":"cuda","indexes":[]}"#).unwrap();
        assert_eq!(c.reranker_model.as_deref(), Some(DEFAULT_RERANKER));
        assert_eq!(c.auto_sync_max_files, 200);
        let c: AppConfig = serde_json::from_str(r#"{"reranker_model":null}"#).unwrap();
        assert_eq!(c.reranker_model, None, "explicit null disables");
    }
}
