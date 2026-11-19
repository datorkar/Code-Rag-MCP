//! Incremental index synchronisation: walk -> hash -> diff -> delete stale ->
//! chunk -> length-sorted batched embedding -> upsert -> rebuild FTS.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use serde::Serialize;

use crate::chunker::{chunk_text, Chunk};
use crate::config::AppConfig;
use crate::embedder;
use crate::lock::FileLock;
use crate::manifest::{self, Manifest};
use crate::paths::Paths;
use crate::store::{self, ChunkStore};
use crate::{walker, Cancelled};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Waiting,
    Scanning,
    Hashing,
    Deleting,
    Chunking,
    LoadingModel,
    Embedding,
    BuildingIndex,
    Done,
    Cancelled,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub index_id: String,
    pub phase: Phase,
    pub files_scanned: usize,
    pub files_total: usize,
    pub files_changed: usize,
    pub chunks_embedded: usize,
    pub chunks_total: usize,
    pub chunks_per_sec: f64,
    pub elapsed_secs: f64,
    pub message: Option<String>,
}

pub type ProgressFn = Arc<dyn Fn(&Progress) + Send + Sync>;

#[derive(Clone, Default)]
pub struct SyncOptions {
    pub force_full: bool,
    pub cancel: Arc<AtomicBool>,
    pub progress: Option<ProgressFn>,
    /// How long to wait for the index lock. `None` = wait forever.
    pub lock_timeout: Option<Duration>,
    /// Refuse (without touching anything) if a full rebuild is needed or more
    /// than this many files changed. Used by search-triggered auto-sync.
    pub max_changed_files: Option<usize>,
}

/// Returned by a capped sync that would be too large to run inline.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SyncTooLarge {
    #[error("index needs a full (re)build")]
    FullRebuild,
    #[error("{changed} files changed (auto-sync limit is {cap})")]
    TooManyFiles { changed: usize, cap: usize },
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct SyncReport {
    pub index_id: String,
    pub full_rebuild: bool,
    pub files_scanned: usize,
    pub files_added: usize,
    pub files_modified: usize,
    pub files_removed: usize,
    pub chunks_embedded: usize,
    pub total_chunks: usize,
    pub duration_secs: f64,
}

struct Reporter {
    p: Progress,
    start: Instant,
    embed_start: Option<Instant>,
    cb: Option<ProgressFn>,
    last_emit: Instant,
}

impl Reporter {
    fn emit(&mut self, force: bool) {
        self.p.elapsed_secs = self.start.elapsed().as_secs_f64();
        if let Some(t) = self.embed_start {
            let s = t.elapsed().as_secs_f64();
            if s > 0.0 {
                self.p.chunks_per_sec = self.p.chunks_embedded as f64 / s;
            }
        }
        if force || self.last_emit.elapsed() >= Duration::from_millis(100) {
            self.last_emit = Instant::now();
            if let Some(cb) = &self.cb {
                cb(&self.p);
            }
        }
    }
    fn phase(&mut self, phase: Phase, msg: impl Into<Option<String>>) {
        self.p.phase = phase;
        self.p.message = msg.into();
        self.emit(true);
    }
}

fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!(Cancelled);
    }
    Ok(())
}

pub async fn sync_index(paths: &Paths, index_id: &str, opts: SyncOptions) -> Result<SyncReport> {
    let mut rep = Reporter {
        p: Progress {
            index_id: index_id.to_string(),
            phase: Phase::Waiting,
            files_scanned: 0,
            files_total: 0,
            files_changed: 0,
            chunks_embedded: 0,
            chunks_total: 0,
            chunks_per_sec: 0.0,
            elapsed_secs: 0.0,
            message: None,
        },
        start: Instant::now(),
        embed_start: None,
        cb: opts.progress.clone(),
        last_emit: Instant::now(),
    };
    let result = sync_inner(paths, index_id, &opts, &mut rep).await;
    match &result {
        Ok(r) => rep.phase(
            Phase::Done,
            format!(
                "+{} ~{} -{} files, {} chunks embedded, {} total",
                r.files_added, r.files_modified, r.files_removed, r.chunks_embedded, r.total_chunks
            ),
        ),
        Err(e) if e.is::<Cancelled>() => rep.phase(Phase::Cancelled, "cancelled".to_string()),
        Err(e) => rep.phase(Phase::Error, format!("{e:#}")),
    }
    result
}

async fn sync_inner(
    paths: &Paths,
    index_id: &str,
    opts: &SyncOptions,
    rep: &mut Reporter,
) -> Result<SyncReport> {
    let app = AppConfig::load(paths)?;
    let cfg = app
        .index(index_id)
        .cloned()
        .ok_or_else(|| anyhow!("unknown index '{index_id}'"))?;
    let cancel = opts.cancel.clone();

    rep.phase(Phase::Waiting, "waiting for index lock".to_string());
    let _lock = FileLock::exclusive(&paths.lock_file(index_id), opts.lock_timeout)
        .await?
        .ok_or_else(|| anyhow!("index '{index_id}' is busy (being indexed by another process)"))?;

    let spec = embedder::model_spec(&cfg.embed_model)?;
    let conn = store::connect(paths).await?;
    let mut manifest = Manifest::load(paths, index_id)?;
    let existing = ChunkStore::open(&conn, index_id).await?;

    let full = opts.force_full
        || manifest.fingerprint != cfg.fingerprint()
        || (existing.is_none() && !manifest.files.is_empty());
    if full && opts.max_changed_files.is_some() {
        bail!(SyncTooLarge::FullRebuild);
    }
    if full {
        tracing::info!("[{index_id}] full rebuild");
        drop(existing);
        ChunkStore::drop(&conn, index_id).await?;
        manifest = Manifest { fingerprint: cfg.fingerprint(), files: Default::default() };
        manifest.save(paths, index_id)?;
    }

    // --- walk + hash -------------------------------------------------------
    rep.phase(Phase::Scanning, format!("scanning {}", cfg.root_path));
    let (files, hashed) = {
        let cfg = cfg.clone();
        let cancel = cancel.clone();
        let prev = manifest.files.clone();
        let cb = rep.cb.clone();
        let id = index_id.to_string();
        tokio::task::spawn_blocking(move || -> Result<_> {
            let files = walker::walk(&cfg, &cancel)?;
            let total = files.len();
            let mut last = Instant::now();
            let hashed = walker::hash_all(&files, &prev, &cancel, |n| {
                if let Some(cb) = &cb {
                    if last.elapsed() >= Duration::from_millis(100) || n == total {
                        last = Instant::now();
                        cb(&Progress {
                            index_id: id.clone(),
                            phase: Phase::Hashing,
                            files_scanned: n,
                            files_total: total,
                            files_changed: 0,
                            chunks_embedded: 0,
                            chunks_total: 0,
                            chunks_per_sec: 0.0,
                            elapsed_secs: 0.0,
                            message: None,
                        });
                    }
                }
            })?;
            Ok((files, hashed))
        })
        .await??
    };
    rep.p.files_scanned = files.len();
    rep.p.files_total = files.len();

    let diff = manifest::diff(&manifest.files, &hashed);
    let changed: Vec<String> = diff.changed().cloned().collect();
    rep.p.files_changed = changed.len() + diff.removed.len();
    tracing::info!(
        "[{index_id}] {} files: +{} ~{} -{} ={}",
        files.len(),
        diff.added.len(),
        diff.modified.len(),
        diff.removed.len(),
        diff.unchanged
    );

    if let Some(cap) = opts.max_changed_files {
        let n = changed.len() + diff.removed.len();
        if n > cap {
            bail!(SyncTooLarge::TooManyFiles { changed: n, cap });
        }
    }

    let store = ChunkStore::open_or_create(&conn, index_id, spec.dim).await?;
    let mut report = SyncReport {
        index_id: index_id.to_string(),
        full_rebuild: full,
        files_scanned: files.len(),
        files_added: diff.added.len(),
        files_modified: diff.modified.len(),
        files_removed: diff.removed.len(),
        ..Default::default()
    };

    if diff.is_empty() {
        if !store.has_fts_index().await && store.count().await? > 0 {
            rep.phase(Phase::BuildingIndex, "building full-text index".to_string());
            store.rebuild_indices().await?;
        }
        return finish(paths, index_id, &store, report, rep).await;
    }

    // --- delete stale chunks ----------------------------------------------
    // `added` is included too: a cancelled previous run may have left partial
    // chunks for files that never made it into the manifest.
    rep.phase(Phase::Deleting, "removing stale chunks".to_string());
    let to_delete: Vec<String> = diff.removed.iter().chain(changed.iter()).cloned().collect();
    if !full {
        store.delete_paths(&to_delete).await?;
    }
    for p in &to_delete {
        manifest.files.remove(p);
    }
    manifest.save(paths, index_id)?;
    check(&cancel)?;

    // --- chunk ---------------------------------------------------------------
    rep.phase(Phase::Chunking, format!("chunking {} files", changed.len()));
    let abs: HashMap<&str, &std::path::Path> =
        files.iter().map(|f| (f.rel_path.as_str(), f.abs_path.as_path())).collect();
    let jobs: Vec<(String, std::path::PathBuf)> = changed
        .iter()
        .filter_map(|p| abs.get(p.as_str()).map(|a| (p.clone(), a.to_path_buf())))
        .collect();
    let (chunk_size, overlap) = (cfg.chunk_size, cfg.chunk_overlap);
    let chunks: Vec<Chunk> = {
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Chunk>> {
            let mut out = Vec::new();
            for (rel, path) in jobs {
                check(&cancel)?;
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        out.extend(chunk_text(&rel, &text, chunk_size, overlap));
                    }
                    Err(e) => tracing::warn!("cannot read {}: {e}", path.display()),
                }
            }
            Ok(out)
        })
        .await??
    };

    let mut remaining: HashMap<String, usize> = changed.iter().map(|p| (p.clone(), 0)).collect();
    for c in &chunks {
        *remaining.get_mut(&c.rel_path).expect("chunk of changed file") += 1;
    }
    // Files that produced no chunks are complete already.
    for (p, n) in &remaining {
        if *n == 0 {
            if let Some(e) = hashed.get(p) {
                manifest.files.insert(p.clone(), e.clone());
            }
        }
    }

    // --- embed (length-sorted to minimise padding) ---------------------------
    let mut order: Vec<usize> = (0..chunks.len()).collect();
    order.sort_by_key(|&i| chunks[i].text.len());
    rep.p.chunks_total = chunks.len();

    rep.phase(Phase::LoadingModel, format!("loading {}", spec.id));
    let max_len = embedder::max_length_for_chunk_size(cfg.chunk_size);
    let emb = {
        let (model, ep, dir) = (cfg.embed_model.clone(), app.execution_provider, paths.models_dir());
        tokio::task::spawn_blocking(move || embedder::cached(&model, ep, &dir, max_len)).await??
    };

    rep.phase(Phase::Embedding, None);
    rep.embed_start = Some(Instant::now());
    let batch = app.embed_batch_size.clamp(1, 512);
    const FLUSH_ROWS: usize = 2048;
    let mut pend_chunks: Vec<Chunk> = Vec::new();
    let mut pend_vecs: Vec<Vec<f32>> = Vec::new();
    let mut cancelled = false;

    for ids in order.chunks(batch) {
        if cancel.load(Ordering::Relaxed) {
            cancelled = true;
            break;
        }
        let texts: Vec<String> = ids.iter().map(|&i| chunks[i].embedding_input()).collect();
        let e = emb.clone();
        let vecs = tokio::task::spawn_blocking(move || e.lock().unwrap().embed_documents(&texts, batch))
            .await??;
        pend_chunks.extend(ids.iter().map(|&i| chunks[i].clone()));
        pend_vecs.extend(vecs);
        rep.p.chunks_embedded += ids.len();
        report.chunks_embedded += ids.len();
        rep.emit(false);

        if pend_chunks.len() >= FLUSH_ROWS {
            flush(&store, &mut pend_chunks, &mut pend_vecs, spec.dim, &mut remaining, &hashed, &mut manifest)
                .await?;
            manifest.save(paths, index_id)?;
        }
    }
    flush(&store, &mut pend_chunks, &mut pend_vecs, spec.dim, &mut remaining, &hashed, &mut manifest)
        .await?;
    manifest.save(paths, index_id)?;
    if cancelled {
        // Keep what was written; complete files are in the manifest and the
        // rest will be redone on the next sync.
        bail!(Cancelled);
    }

    rep.phase(Phase::BuildingIndex, "building full-text index".to_string());
    store.rebuild_indices().await?;
    finish(paths, index_id, &store, report, rep).await
}

async fn flush(
    store: &ChunkStore,
    chunks: &mut Vec<Chunk>,
    vecs: &mut Vec<Vec<f32>>,
    dim: usize,
    remaining: &mut HashMap<String, usize>,
    hashed: &std::collections::BTreeMap<String, manifest::FileEntry>,
    manifest: &mut Manifest,
) -> Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    store.add(chunks, std::mem::take(vecs), dim).await?;
    for c in chunks.drain(..) {
        let n = remaining.get_mut(&c.rel_path).expect("tracked");
        *n -= 1;
        if *n == 0 {
            if let Some(e) = hashed.get(&c.rel_path) {
                manifest.files.insert(c.rel_path.clone(), e.clone());
            }
        }
    }
    Ok(())
}

async fn finish(
    paths: &Paths,
    index_id: &str,
    store: &ChunkStore,
    mut report: SyncReport,
    rep: &mut Reporter,
) -> Result<SyncReport> {
    let total = store.count().await?;
    report.total_chunks = total;
    report.duration_secs = rep.start.elapsed().as_secs_f64();
    let id = index_id.to_string();
    AppConfig::update(paths, move |c| {
        if let Some(ix) = c.index_mut(&id) {
            ix.last_indexed_at = Some(Utc::now());
            ix.chunk_count = total as u64;
        }
        Ok(())
    })
    .context("updating config after sync")?;
    Ok(report)
}
