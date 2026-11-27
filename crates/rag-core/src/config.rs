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

/// Per-index model precision (which ONNX variant of the embedding model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// FP16 on GPU (same results as FP32, faster), FP32 on CPU. Older configs'
    /// `auto` / `fp32` / `fp16` map here.
    #[default]
    #[serde(alias = "auto", alias = "fp32", alias = "fp16")]
    Standard,
    /// Dynamically quantized INT8 (`model_quantized.onnx`): ~1.4x faster on
    /// CPU, ~10% of top results differ from FP32. Best on CPU.
    Int8,
}

/// How hard indexing may use the machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ResourceMode {
    /// Below-normal priority; Low on CPU-only machines and integrated GPUs,
    /// full speed on discrete GPUs.
    #[default]
    Auto,
    /// Normal priority, all cores, 2 GPU workers, big batches.
    Max,
    /// Below-normal priority, all logical threads but 2 (one physical core
    /// free), 1 worker, small batches. Keeps laptops responsive.
    Low,
}

/// The ONNX file actually loaded for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelVariant {
    Fp32,
    Fp16,
    Int8,
}

impl ModelVariant {
    /// Repo file for the variant (`None` = the model's default FP32 file).
    pub fn onnx_file(self) -> Option<&'static str> {
        match self {
            ModelVariant::Fp32 => None,
            ModelVariant::Fp16 => Some("onnx/model_fp16.onnx"),
            ModelVariant::Int8 => Some("onnx/model_quantized.onnx"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// Default model for newly created indexes.
    pub active_embed_model: String,
    pub execution_provider: ExecutionProvider,
    /// Upper bound on chunks per ONNX batch (0 = no cap; batches are sized by
    /// `embed_token_budget`).
    pub embed_batch_size: usize,
    /// Approximate tokens per batch (batch size x longest chunk). 0 = auto:
    /// 65536 on GPU, 8192 on CPU. Halved automatically on out-of-memory.
    pub embed_token_budget: usize,
    /// Parallel embedding sessions (0 = auto: 2 on GPU, 1 on CPU). Two sessions
    /// keep the GPU busy while the other one tokenizes on the CPU.
    pub embed_workers: usize,
    /// Machine load while indexing (see [`ResourceMode`]).
    pub resource_mode: ResourceMode,
    /// Troubleshooting: never use the FP16 models on GPU (e.g. GPUs with slow
    /// FP16 such as GTX 10-series, or driver issues). `LOCAL_CODE_RAG_FP16=0|1`
    /// overrides this.
    pub force_fp32: bool,
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
    /// Look for a newer release on GitHub at startup (at most once a day).
    #[serde(default = "default_true")]
    pub check_updates: bool,
    /// Last automatic update check.
    pub last_update_check: Option<DateTime<Utc>>,
    /// Precision preselected for new indexes (set by the first-run setup).
    pub default_precision: Precision,
    /// First-run setup finished or skipped.
    pub setup_completed: bool,
    pub indexes: Vec<IndexConfig>,
}

fn default_true() -> bool {
    true
}

fn default_reranker() -> Option<String> {
    Some(DEFAULT_RERANKER.into())
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            active_embed_model: DEFAULT_EMBED_MODEL.into(),
            execution_provider: ExecutionProvider::Auto,
            embed_batch_size: 0,
            embed_token_budget: 0,
            embed_workers: 0,
            resource_mode: ResourceMode::Auto,
            force_fp32: false,
            reranker_model: Some(DEFAULT_RERANKER.into()),
            rerank_candidates: 30,
            auto_sync_max_files: 200,
            snippet_max_lines: 40,
            check_updates: true,
            last_update_check: None,
            default_precision: Precision::Standard,
            setup_completed: false,
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
    /// Embedding model precision for this index (see [`Precision`]).
    pub precision: Precision,
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
            precision: Precision::Standard,
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

    /// The model file to load for this index under the given app settings.
    pub fn model_variant(&self, app: &AppConfig) -> ModelVariant {
        match self.precision {
            Precision::Int8 => ModelVariant::Int8,
            Precision::Standard if app.fp16_enabled() => ModelVariant::Fp16,
            Precision::Standard => ModelVariant::Fp32,
        }
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
        // `v2`: chunks table gained `text_hash` (embedding cache).
        // `v3`: scope context, split-identifier keyword index, boilerplate
        // header skipping. Older indexes are rebuilt once.
        // FP32 and FP16 produce the same vectors, so only INT8 is a separate
        // class (switching to/from it rebuilds the index).
        let class = if self.precision == Precision::Int8 { "|int8" } else { "" };
        // Models that gained a document prefix rebuild once.
        let dp = if crate::embedder::has_v3_doc_prefix(&self.embed_model) { "|dp" } else { "" };
        format!("v3|{}|{}|{}{class}{dp}", self.embed_model, self.chunk_size, self.chunk_overlap)
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

    /// Whether FP16 models are used: on by default for GPU execution providers
    /// (unless `force_fp32`); `LOCAL_CODE_RAG_FP16=1|0` overrides. Never on CPU.
    pub fn fp16_enabled(&self) -> bool {
        let wanted = match std::env::var("LOCAL_CODE_RAG_FP16").ok().as_deref().map(str::trim) {
            Some("1") | Some("true") | Some("on") => true,
            Some("0") | Some("false") | Some("off") => false,
            _ => !self.force_fp32,
        };
        wanted && self.execution_provider != ExecutionProvider::Cpu && crate::runtime::gpu_available()
    }

    /// Whether indexing should hold back (Low mode, or Auto on CPU-only /
    /// integrated-GPU machines).
    pub fn low_impact(&self) -> bool {
        match self.resource_mode {
            ResourceMode::Max => false,
            ResourceMode::Low => true,
            ResourceMode::Auto => {
                self.execution_provider == ExecutionProvider::Cpu
                    || !crate::runtime::gpu_available()
                    || crate::runtime::integrated_gpu_only()
            }
        }
    }

    /// ONNX Runtime intra-op threads: all logical threads but 2 in low-impact
    /// mode (leaves one physical core free on SMT CPUs); `None` = ORT default.
    pub fn intra_threads(&self) -> Option<usize> {
        self.low_impact().then(|| {
            std::thread::available_parallelism().map(|n| n.get().saturating_sub(2).max(1)).unwrap_or(1)
        })
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

#[cfg(test)]
mod precision_tests {
    use super::*;

    #[test]
    fn fingerprint_only_changes_for_int8() {
        let mut ix = IndexConfig { id: "a".into(), root_path: "x".into(), ..Default::default() };
        let base = ix.fingerprint();
        ix.precision = Precision::Int8;
        assert_ne!(ix.fingerprint(), base, "int8 needs a rebuild");
    }

    #[test]
    fn old_precision_values_map_to_standard() {
        for v in ["auto", "fp32", "fp16", "standard"] {
            let p: Precision = serde_json::from_str(&format!("\"{v}\"")).unwrap();
            assert_eq!(p, Precision::Standard, "{v}");
        }
        let p: Precision = serde_json::from_str("\"int8\"").unwrap();
        assert_eq!(p, Precision::Int8);
    }

    #[test]
    fn variant_selection() {
        // Env override must not leak in from the test environment.
        std::env::remove_var("LOCAL_CODE_RAG_FP16");
        let mut app = AppConfig { execution_provider: ExecutionProvider::Cpu, ..Default::default() };
        let mut ix = IndexConfig::default();
        assert_eq!(ix.model_variant(&app), ModelVariant::Fp32, "FP32 on CPU");
        app.execution_provider = ExecutionProvider::Gpu;
        let gpu_variant = if crate::runtime::gpu_available() { ModelVariant::Fp16 } else { ModelVariant::Fp32 };
        assert_eq!(ix.model_variant(&app), gpu_variant, "FP16 by default on GPU");
        app.force_fp32 = true;
        assert_eq!(ix.model_variant(&app), ModelVariant::Fp32, "force_fp32");
        ix.precision = Precision::Int8;
        assert_eq!(ix.model_variant(&app), ModelVariant::Int8);
    }
}
