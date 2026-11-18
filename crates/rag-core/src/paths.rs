//! On-disk layout of the application data directory.
//!
//! ```text
//! <root>/config.json
//! <root>/manifests/<index_id>.json
//! <root>/locks/<index_id>.lock
//! <root>/lancedb/idx_<index_id>.lance
//! <root>/models/            (FastEmbed ONNX cache)
//! ```
//!
//! `<root>` is `%APPDATA%\local-code-rag` on Windows, `~/.local/share/local-code-rag`
//! on Linux and `~/Library/Application Support/local-code-rag` on macOS. It can be
//! overridden with the `LOCAL_CODE_RAG_HOME` environment variable.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub const APP_DIR_NAME: &str = "local-code-rag";
pub const HOME_ENV: &str = "LOCAL_CODE_RAG_HOME";

#[derive(Debug, Clone)]
pub struct Paths {
    root: PathBuf,
}

impl Paths {
    /// Resolve the default data root (env override, then OS data dir).
    pub fn resolve() -> Result<Self> {
        if let Some(p) = std::env::var_os(HOME_ENV).filter(|p| !p.is_empty()) {
            return Self::at(PathBuf::from(p));
        }
        let base = dirs::data_dir().context("could not determine the OS data directory")?;
        Self::at(base.join(APP_DIR_NAME))
    }

    /// Use an explicit root directory (created if missing).
    pub fn at(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        for sub in ["manifests", "locks", "lancedb", "models"] {
            std::fs::create_dir_all(root.join(sub))
                .with_context(|| format!("creating {}", root.join(sub).display()))?;
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn config_file(&self) -> PathBuf {
        self.root.join("config.json")
    }
    pub fn manifest_file(&self, index_id: &str) -> PathBuf {
        self.root.join("manifests").join(format!("{index_id}.json"))
    }
    pub fn lock_file(&self, name: &str) -> PathBuf {
        self.root.join("locks").join(format!("{name}.lock"))
    }
    pub fn lancedb_dir(&self) -> PathBuf {
        self.root.join("lancedb")
    }
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }
}

/// Write a file atomically (write to a sibling temp file, then rename over).
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}
