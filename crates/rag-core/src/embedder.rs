//! FastEmbed (ONNX Runtime) text embedding with GPU execution providers and
//! a process-wide model cache.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::Serialize;

use crate::config::ExecutionProvider;

/// Model description for the GUI dropdown / MCP output.
#[derive(Debug, Clone, Serialize)]
pub struct ModelSpec {
    /// Canonical id stored in config (HuggingFace repo id).
    pub id: String,
    pub dim: usize,
    pub max_context: usize,
    pub description: String,
    /// Shown first in the GUI.
    pub recommended: bool,
}

/// Aliases for upstream repo names that FastEmbed serves from mirror repos.
const ALIASES: &[(&str, EmbeddingModel)] = &[
    ("BAAI/bge-base-en-v1.5", EmbeddingModel::BGEBaseENV15),
    ("BAAI/bge-small-en-v1.5", EmbeddingModel::BGESmallENV15),
    ("BAAI/bge-large-en-v1.5", EmbeddingModel::BGELargeENV15),
    ("sentence-transformers/all-MiniLM-L6-v2", EmbeddingModel::AllMiniLML6V2),
];

const RECOMMENDED: &[EmbeddingModel] = &[
    EmbeddingModel::JinaEmbeddingsV2BaseCode,
    EmbeddingModel::BGEBaseENV15,
    EmbeddingModel::BGESmallENV15,
    EmbeddingModel::NomicEmbedTextV15,
];

fn max_context(m: &EmbeddingModel) -> usize {
    use EmbeddingModel::*;
    match m {
        JinaEmbeddingsV2BaseCode | JinaEmbeddingsV2BaseEN | NomicEmbedTextV1
        | NomicEmbedTextV15 | NomicEmbedTextV15Q => 8192,
        EmbeddingGemma300M | EmbeddingGemma300MQ | EmbeddingGemma300MQ4 => 2048,
        _ => 512,
    }
}

fn display_id(m: &EmbeddingModel, code: &str) -> String {
    ALIASES
        .iter()
        .find(|(_, model)| model == m)
        .map(|(alias, _)| alias.to_string())
        .unwrap_or_else(|| code.to_string())
}

pub fn supported_models() -> Vec<ModelSpec> {
    let mut v: Vec<ModelSpec> = TextEmbedding::list_supported_models()
        .into_iter()
        .map(|info| ModelSpec {
            id: display_id(&info.model, &info.model_code),
            dim: info.dim,
            max_context: max_context(&info.model),
            description: info.description.clone(),
            recommended: RECOMMENDED.contains(&info.model),
        })
        .collect();
    v.sort_by_key(|s| {
        let rank = RECOMMENDED
            .iter()
            .position(|r| resolve_model(&s.id).ok().as_ref() == Some(r))
            .unwrap_or(usize::MAX);
        (rank, s.id.clone())
    });
    v
}

/// Map a config string (HF id, alias, or FastEmbed enum name) to a model.
pub fn resolve_model(name: &str) -> Result<EmbeddingModel> {
    let n = name.trim();
    if let Some((_, m)) = ALIASES.iter().find(|(a, _)| a.eq_ignore_ascii_case(n)) {
        return Ok(m.clone());
    }
    let all = TextEmbedding::list_supported_models();
    if let Some(info) = all.iter().find(|i| i.model_code.eq_ignore_ascii_case(n)) {
        return Ok(info.model.clone());
    }
    n.parse::<EmbeddingModel>()
        .map_err(|_| anyhow!("unsupported embedding model '{name}' (see supported_models())"))
}

pub fn model_spec(name: &str) -> Result<ModelSpec> {
    let model = resolve_model(name)?;
    let info = TextEmbedding::get_model_info(&model)?;
    Ok(ModelSpec {
        id: display_id(&model, &info.model_code),
        dim: info.dim,
        max_context: max_context(&model),
        description: info.description.clone(),
        recommended: RECOMMENDED.contains(&model),
    })
}

/// A loaded model. `embed` is CPU/GPU-bound and synchronous; call it from
/// `spawn_blocking`.
pub struct Embedder {
    model: TextEmbedding,
    pub spec: ModelSpec,
    query_prefix: &'static str,
}

impl Embedder {
    pub fn load(
        model_name: &str,
        ep: ExecutionProvider,
        cache_dir: &Path,
        max_length: usize,
    ) -> Result<Self> {
        let model = resolve_model(model_name)?;
        let spec = model_spec(model_name)?;
        let max_length = max_length.clamp(128, spec.max_context);
        tracing::info!(
            "loading embedding model {} (dim {}, max_length {max_length}, ep {ep:?})",
            spec.id,
            spec.dim
        );
        let (eps, policy) = crate::runtime::session_plan(ep);
        let mut opts = TextInitOptions::new(model.clone())
            .with_environment(crate::runtime::environment()?)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(false)
            .with_max_length(max_length)
            .with_execution_providers(eps);
        if let Some(p) = policy {
            opts = opts.with_auto_device(p);
        }
        let te = TextEmbedding::try_new(opts)
            .with_context(|| format!("initialising embedding model {}", spec.id))?;
        let query_prefix = if spec.id.to_ascii_lowercase().contains("bge-") {
            "Represent this sentence for searching relevant passages: "
        } else if spec.id.to_ascii_lowercase().contains("nomic") {
            "search_query: "
        } else {
            ""
        };
        Ok(Self { model: te, spec, query_prefix })
    }

    pub fn embed_documents(&mut self, texts: &[String], batch: usize) -> Result<Vec<Vec<f32>>> {
        let docs: Vec<String>;
        let input: &[String] = if self.spec.id.to_ascii_lowercase().contains("nomic") {
            docs = texts.iter().map(|t| format!("search_document: {t}")).collect();
            &docs
        } else {
            texts
        };
        self.model.embed(input, Some(batch.max(1))).context("embedding batch")
    }

    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let q = format!("{}{}", self.query_prefix, query);
        let mut v = self.model.embed(vec![q], Some(1)).context("embedding query")?;
        v.pop().ok_or_else(|| anyhow!("model returned no embedding"))
    }
}

pub type SharedEmbedder = Arc<Mutex<Embedder>>;

static EMBEDDERS: OnceLock<Mutex<HashMap<String, SharedEmbedder>>> = OnceLock::new();
static RERANKERS: OnceLock<Mutex<HashMap<String, SharedReranker>>> = OnceLock::new();

/// Drop all loaded ONNX sessions (e.g. after new execution providers were
/// installed); they are recreated lazily on next use.
pub fn clear_caches() {
    if let Some(c) = EMBEDDERS.get() {
        c.lock().unwrap().clear();
    }
    if let Some(c) = RERANKERS.get() {
        c.lock().unwrap().clear();
    }
}

/// Process-wide cache so the MCP server does not reload ONNX sessions per call.
pub fn cached(
    model_name: &str,
    ep: ExecutionProvider,
    cache_dir: &Path,
    max_length: usize,
) -> Result<SharedEmbedder> {
    let key = format!("{}|{ep:?}|{max_length}", model_name.to_ascii_lowercase());
    let cache = EMBEDDERS.get_or_init(Default::default);
    if let Some(e) = cache.lock().unwrap().get(&key) {
        return Ok(e.clone());
    }
    let e = Arc::new(Mutex::new(Embedder::load(model_name, ep, cache_dir, max_length)?));
    cache.lock().unwrap().insert(key, e.clone());
    Ok(e)
}

/// Token budget derived from chunk size (~3 chars/token for code, plus headroom).
pub fn max_length_for_chunk_size(chunk_size: usize) -> usize {
    (chunk_size / 3 + 96).next_power_of_two().clamp(256, 8192)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_aliases_and_codes() {
        assert_eq!(resolve_model("BAAI/bge-base-en-v1.5").unwrap(), EmbeddingModel::BGEBaseENV15);
        assert_eq!(
            resolve_model("jinaai/jina-embeddings-v2-base-code").unwrap(),
            EmbeddingModel::JinaEmbeddingsV2BaseCode
        );
        assert_eq!(resolve_model("BGESmallENV15").unwrap(), EmbeddingModel::BGESmallENV15);
        assert!(resolve_model("nope/nope").is_err());
        let specs = supported_models();
        assert_eq!(specs[0].id, "jinaai/jina-embeddings-v2-base-code");
        assert_eq!(specs[0].max_context, 8192);
    }
}

// --- reranking ---------------------------------------------------------------

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

pub fn supported_rerankers() -> Vec<String> {
    TextRerank::list_supported_models().into_iter().map(|m| m.model_code).collect()
}

pub fn resolve_reranker(name: &str) -> Result<RerankerModel> {
    name.trim()
        .parse::<RerankerModel>()
        .map_err(|_| anyhow!("unsupported reranker '{name}'; supported: {}", supported_rerankers().join(", ")))
}

pub type SharedReranker = Arc<Mutex<TextRerank>>;

pub fn cached_reranker(name: &str, ep: ExecutionProvider, cache_dir: &Path) -> Result<SharedReranker> {
    let key = format!("{}|{ep:?}", name.to_ascii_lowercase());
    let cache = RERANKERS.get_or_init(Default::default);
    if let Some(r) = cache.lock().unwrap().get(&key) {
        return Ok(r.clone());
    }
    let model = resolve_reranker(name)?;
    tracing::info!("loading reranker {name} (ep {ep:?})");
    let (eps, policy) = crate::runtime::session_plan(ep);
    let mut opts = RerankInitOptions::new(model)
        .with_environment(crate::runtime::environment()?)
        .with_cache_dir(cache_dir.to_path_buf())
        .with_show_download_progress(false)
        .with_max_length(1024)
        .with_execution_providers(eps);
    if let Some(p) = policy {
        opts = opts.with_auto_device(p);
    }
    let r = Arc::new(Mutex::new(
        TextRerank::try_new(opts).with_context(|| format!("initialising reranker {name}"))?,
    ));
    cache.lock().unwrap().insert(key, r.clone());
    Ok(r)
}
