//! High-level facade shared by the MCP server and the Tauri GUI.

use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use chrono::Utc;
use serde::Serialize;

use crate::config::{AppConfig, IndexConfig};
use crate::embedder;
use crate::lock::FileLock;
use crate::manifest::Manifest;
use crate::paths::Paths;
use crate::store::{self, ChunkStore, SearchHit};
use crate::snippet;
use crate::sync::{sync_index, SyncOptions, SyncReport, SyncTooLarge};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexStatus {
    Ready,
    Indexing,
    Stale,
    NeverIndexed,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexInfo {
    #[serde(flatten)]
    pub config: IndexConfig,
    pub status: IndexStatus,
    /// Settings changed since the last build (model / chunking) -> full rebuild needed.
    pub needs_rebuild: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SaveResult {
    pub index: IndexConfig,
    pub created: bool,
    /// Set when settings were adjusted (e.g. chunk size clamped to the model).
    pub note: Option<String>,
    /// Model or chunk settings changed on an existing, built index.
    pub needs_rebuild: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchOutcome {
    pub hits: Vec<SearchHit>,
    pub auto_sync: Option<SyncReport>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Engine {
    pub paths: Paths,
}

impl Engine {
    pub fn new() -> Result<Self> {
        Ok(Self { paths: Paths::resolve()? })
    }

    pub fn with_paths(paths: Paths) -> Self {
        Self { paths }
    }

    pub fn config(&self) -> Result<AppConfig> {
        AppConfig::load(&self.paths)
    }

    pub fn update_settings(&self, f: impl FnOnce(&mut AppConfig)) -> Result<AppConfig> {
        AppConfig::update(&self.paths, |c| {
            f(c);
            Ok(c.clone())
        })
    }

    pub fn status(&self, ix: &IndexConfig) -> (IndexStatus, bool) {
        let lock = self.paths.lock_file(&ix.id);
        let busy = matches!(FileLock::try_exclusive(&lock), Ok(None));
        let manifest_fp = Manifest::load_fingerprint(&self.paths, &ix.id);
        let needs_rebuild = ix.last_indexed_at.is_some() && manifest_fp != ix.fingerprint();
        let status = if busy {
            IndexStatus::Indexing
        } else if ix.last_indexed_at.is_none() {
            IndexStatus::NeverIndexed
        } else if needs_rebuild || (ix.auto_update_cooldown_hours > 0.0 && ix.is_stale(Utc::now())) {
            IndexStatus::Stale
        } else {
            IndexStatus::Ready
        };
        (status, needs_rebuild)
    }

    pub fn list(&self) -> Result<Vec<IndexInfo>> {
        Ok(self
            .config()?
            .indexes
            .into_iter()
            .map(|config| {
                let (status, needs_rebuild) = self.status(&config);
                IndexInfo { config, status, needs_rebuild }
            })
            .collect())
    }

    /// Insert or update an index. Runtime fields (last_indexed_at, chunk_count)
    /// are preserved from the stored copy. `original_id` allows renaming.
    pub fn save_index(&self, mut ix: IndexConfig, original_id: Option<&str>) -> Result<SaveResult> {
        ix.normalize()?;
        if !std::path::Path::new(&ix.root_path).is_dir() {
            bail!("root folder '{}' does not exist or is not a folder", ix.root_path);
        }
        let spec = embedder::model_spec(&ix.embed_model)?;
        let mut note = None;
        if ix.chunk_size > spec.max_chunk_chars {
            note = Some(format!(
                "Chunk size lowered from {} to {} characters: {} reads at most {} tokens.",
                ix.chunk_size, spec.max_chunk_chars, spec.id, spec.max_context
            ));
            ix.chunk_size = spec.max_chunk_chars;
            if ix.chunk_overlap >= ix.chunk_size {
                ix.chunk_overlap = ix.chunk_size / 10;
            }
        }
        AppConfig::update(&self.paths, |cfg| {
            let lookup = original_id.unwrap_or(&ix.id).to_string();
            if lookup != ix.id && cfg.index(&ix.id).is_some() {
                bail!("an index with id '{}' already exists", ix.id);
            }
            match cfg.index_mut(&lookup) {
                Some(existing) => {
                    if lookup != ix.id {
                        bail!("renaming index ids is not supported; create a new index instead");
                    }
                    let needs_rebuild =
                        existing.last_indexed_at.is_some() && existing.fingerprint() != ix.fingerprint();
                    ix.last_indexed_at = existing.last_indexed_at;
                    ix.chunk_count = existing.chunk_count;
                    *existing = ix.clone();
                    Ok(SaveResult { index: ix, created: false, needs_rebuild, note })
                }
                None => {
                    ix.last_indexed_at = None;
                    ix.chunk_count = 0;
                    cfg.indexes.push(ix.clone());
                    Ok(SaveResult { index: ix, created: true, needs_rebuild: false, note })
                }
            }
        })
    }

    pub async fn delete_index(&self, id: &str) -> Result<()> {
        let _lock = FileLock::exclusive(&self.paths.lock_file(id), Some(Duration::from_secs(5)))
            .await?
            .ok_or_else(|| anyhow!("index '{id}' is busy; cancel indexing first"))?;
        let conn = store::connect(&self.paths).await?;
        ChunkStore::drop(&conn, id).await?;
        Manifest::delete(&self.paths, id)?;
        AppConfig::update(&self.paths, |c| {
            c.indexes.retain(|i| i.id != id);
            Ok(())
        })
    }

    pub async fn sync(&self, id: &str, opts: SyncOptions) -> Result<SyncReport> {
        sync_index(&self.paths, id, opts).await
    }

    /// Hybrid search. With `allow_auto_sync`, an incremental sync runs first
    /// when the index has `auto_update_on_search` and is past its cooldown.
    pub async fn search(
        &self,
        id: &str,
        query: &str,
        limit: usize,
        allow_auto_sync: bool,
    ) -> Result<SearchOutcome> {
        let app = self.config()?;
        let ix = app.index(id).cloned().ok_or_else(|| {
            let ids: Vec<_> = app.indexes.iter().map(|i| i.id.as_str()).collect();
            anyhow!("unknown index '{id}'. Available: {}", ids.join(", "))
        })?;
        let mut notes = Vec::new();
        let mut auto_sync = None;

        if allow_auto_sync && ix.auto_update_on_search && ix.is_stale(Utc::now()) {
            let opts = SyncOptions {
                lock_timeout: Some(Duration::ZERO),
                max_changed_files: Some(app.auto_sync_max_files),
                ..Default::default()
            };
            match self.sync(id, opts).await {
                Ok(r) => auto_sync = Some(r),
                Err(e) if e.is::<SyncTooLarge>() => {
                    notes.push(format!(
                        "index not auto-updated: {e}; results may be out of date. \
                         Ask the user to run \"Index now\" in the Local Code RAG app, or call reindex_path."
                    ));
                }
                Err(e) => {
                    tracing::warn!("auto-sync of '{id}' skipped: {e:#}");
                    notes.push(format!("auto-update skipped: {e}"));
                }
            }
        }

        // Shared lock: wait briefly for a running GUI re-index; LanceDB reads are
        // MVCC-safe, so proceed (with a note) rather than block the LLM forever.
        let _guard = FileLock::shared(&self.paths.lock_file(id), Some(Duration::from_secs(3))).await?;
        if _guard.is_none() {
            notes.push("index is currently being rebuilt; results may be incomplete".into());
        }

        let conn = store::connect(&self.paths).await?;
        let Some(store) = ChunkStore::open(&conn, id).await? else {
            bail!("index '{id}' has not been built yet; run reindex first");
        };
        let emb = {
            let (model, ep, dir) = (ix.embed_model.clone(), app.execution_provider, self.paths.models_dir());
            let chunk_size = embedder::model_spec(&ix.embed_model)
                .map(|m| ix.chunk_size.min(m.max_chunk_chars))
                .unwrap_or(ix.chunk_size);
            let max_len = embedder::max_length_for_chunk_size(chunk_size);
            let variant = ix.model_variant(&app);
            let threads = app.intra_threads();
            tokio::task::spawn_blocking(move || embedder::cached(&model, ep, &dir, max_len, variant, threads)).await??
        };
        let q = query.to_string();
        let vector = tokio::task::spawn_blocking(move || emb.lock().unwrap().embed_query(&q)).await??;
        let limit = limit.clamp(1, 50);
        let candidates = if app.reranker_model.is_some() { limit.max(app.rerank_candidates) } else { limit };
        let mut hits = store.search(query, &vector, candidates).await?;

        if let Some(model) = app.reranker_model.clone().filter(|_| hits.len() > 1) {
            match self.rerank(&model, app.execution_provider, app.fp16_enabled(), query, &hits).await {
                Ok(order) => {
                    hits = order
                        .into_iter()
                        .map(|(i, score)| SearchHit { score, ..hits[i].clone() })
                        .collect();
                }
                Err(e) => {
                    tracing::warn!("rerank failed: {e:#}");
                    notes.push(format!("reranker unavailable ({e}); showing hybrid ranking"));
                }
            }
        }
        hits.truncate(limit);
        let hits = hits
            .into_iter()
            .map(|h| snippet::trim_hit(h, query, app.snippet_max_lines))
            .collect();
        Ok(SearchOutcome { hits, auto_sync, notes })
    }

    /// Cross-encoder scores; returns `(hit index, score)` best first.
    async fn rerank(
        &self,
        model: &str,
        ep: crate::ExecutionProvider,
        fp16: bool,
        query: &str,
        hits: &[SearchHit],
    ) -> Result<Vec<(usize, f32)>> {
        let rr = {
            let (model, dir) = (model.to_string(), self.paths.models_dir());
            tokio::task::spawn_blocking(move || embedder::cached_reranker(&model, ep, &dir, fp16)).await??
        };
        let docs: Vec<String> = hits
            .iter()
            .map(|h| {
                let body: String = h.text.chars().take(3000).collect();
                format!("// File: {}\n{body}", h.path)
            })
            .collect();
        let q = query.to_string();
        let res = tokio::task::spawn_blocking(move || rr.lock().unwrap().rerank(q, docs, false, Some(16)))
            .await??;
        let mut order: Vec<(usize, f32)> = res.into_iter().map(|r| (r.index, r.score)).collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(order)
    }
}
