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

/// Seconds spent per phase (writing overlaps embedding).
#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct PhaseTimings {
    pub scan: f64,
    pub delete: f64,
    pub chunk: f64,
    pub load_model: f64,
    pub embed: f64,
    pub write: f64,
    pub index: f64,
}

impl std::fmt::Display for PhaseTimings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "scan {:.1}s · chunk {:.1}s · model {:.1}s · embed {:.1}s · write {:.1}s · index {:.1}s",
            self.scan, self.chunk, self.load_model, self.embed, self.write, self.index
        )
    }
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
    /// Chunks whose vector was reused (unchanged text or duplicate in this sync).
    pub chunks_reused: usize,
    pub total_chunks: usize,
    pub duration_secs: f64,
    pub timings: PhaseTimings,
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
    let mut timings = PhaseTimings::default();
    let t_scan = Instant::now();
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
    timings.scan = t_scan.elapsed().as_secs_f64();

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
        timings,
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
    let t_del = Instant::now();
    // Vectors of the changed files' current chunks, keyed by text hash: chunks
    // whose text is unchanged are not re-embedded.
    let cache: HashMap<String, Vec<f32>> =
        if full { HashMap::new() } else { store.cached_vectors(&changed).await? };
    let to_delete: Vec<String> = diff.removed.iter().chain(changed.iter()).cloned().collect();
    if !full {
        store.delete_paths(&to_delete).await?;
    }
    for p in &to_delete {
        manifest.files.remove(p);
    }
    manifest.save(paths, index_id)?;
    report.timings.delete = t_del.elapsed().as_secs_f64();
    check(&cancel)?;

    // --- chunk ---------------------------------------------------------------
    let t_chunk = Instant::now();
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

    report.timings.chunk = t_chunk.elapsed().as_secs_f64();

    // --- reuse cached vectors / dedupe -----------------------------------------
    let hashes: Vec<String> =
        chunks.iter().map(|c| blake3::hash(c.embedding_input().as_bytes()).to_hex().to_string()).collect();
    let mut vecs: Vec<Option<Vec<f32>>> = hashes.iter().map(|h| cache.get(h).cloned()).collect();
    drop(cache);
    // Unique chunks that still need embedding; duplicates ride along.
    let mut dups: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut unique: Vec<usize> = Vec::new();
    for (i, h) in hashes.iter().enumerate() {
        if vecs[i].is_some() {
            continue;
        }
        let entry = dups.entry(h.as_str()).or_default();
        if entry.is_empty() {
            unique.push(i);
        }
        entry.push(i);
    }
    report.chunks_reused = chunks.len() - unique.len();

    // --- writer task (overlaps embedding) ------------------------------------
    let (tx, mut rx) = tokio::sync::mpsc::channel::<WriteBatch>(2);
    let writer = {
        let store = ChunkStore { table: store.table.clone() };
        let (paths, id, dim) = (paths.clone(), index_id.to_string(), spec.dim);
        let hashed = hashed.clone();
        let mut manifest = std::mem::take(&mut manifest);
        tokio::spawn(async move {
            let mut write_secs = 0.0;
            while let Some(b) = rx.recv().await {
                let t = Instant::now();
                store.add(&b.chunks, &b.hashes, b.vecs, dim).await?;
                for c in &b.chunks {
                    let n = remaining.get_mut(&c.rel_path).expect("tracked");
                    *n -= 1;
                    if *n == 0 {
                        if let Some(e) = hashed.get(&c.rel_path) {
                            manifest.files.insert(c.rel_path.clone(), e.clone());
                        }
                    }
                }
                manifest.save(&paths, &id)?;
                write_secs += t.elapsed().as_secs_f64();
            }
            anyhow::Ok((manifest, write_secs))
        })
    };
    const FLUSH_ROWS: usize = 2048;
    let mut pending = WriteBatch::default();

    // Reused chunks go straight to the writer.
    for i in 0..chunks.len() {
        if let Some(v) = vecs[i].take() {
            pending.push(&chunks[i], &hashes[i], v);
            if pending.len() >= FLUSH_ROWS {
                tx.send(std::mem::take(&mut pending)).await.map_err(|_| anyhow!("writer stopped"))?;
            }
        }
    }

    // --- embed (length-sorted, token-budget batches) --------------------------
    let max_len = embedder::max_length_for_chunk_size(cfg.chunk_size);
    unique.sort_by_key(|&i| chunks[i].text.len());
    let tokens: Vec<usize> =
        unique.iter().map(|&i| crate::batching::est_tokens(chunks[i].text.len(), max_len)).collect();
    rep.p.chunks_total = unique.len();

    let mut cancelled = false;
    if !unique.is_empty() {
        let t_model = Instant::now();
        rep.phase(Phase::LoadingModel, format!("loading {}", spec.id));
        let emb = {
            let (model, ep, dir) = (cfg.embed_model.clone(), app.execution_provider, paths.models_dir());
            let fp16 = app.fp16_enabled();
            tokio::task::spawn_blocking(move || embedder::cached(&model, ep, &dir, max_len, fp16)).await??
        };
        report.timings.load_model = t_model.elapsed().as_secs_f64();

        let gpu = !matches!(app.execution_provider, crate::ExecutionProvider::Cpu);
        let budget = Arc::new(std::sync::atomic::AtomicUsize::new(match app.embed_token_budget {
            0 if gpu => crate::batching::GPU_TOKEN_BUDGET,
            0 => crate::batching::CPU_TOKEN_BUDGET,
            n => n,
        }));
        let reused_msg = (report.chunks_reused > 0).then(|| format!("{} chunks reused", report.chunks_reused));
        rep.phase(Phase::Embedding, reused_msg);
        rep.embed_start = Some(Instant::now());

        let mut i = 0;
        while i < unique.len() {
            if cancel.load(Ordering::Relaxed) {
                cancelled = true;
                break;
            }
            let end = crate::batching::next_batch_end(&tokens, i, budget.load(Ordering::Relaxed), app.embed_batch_size);
            let ids: Vec<usize> = unique[i..end].to_vec();
            let texts: Vec<String> = ids.iter().map(|&k| chunks[k].embedding_input()).collect();
            let (e, b) = (emb.clone(), budget.clone());
            let t = Instant::now();
            let out = tokio::task::spawn_blocking(move || embed_split(&mut e.lock().unwrap(), &texts, &b)).await??;
            report.timings.embed += t.elapsed().as_secs_f64();
            for (k, v) in ids.iter().zip(out) {
                let same = &dups[hashes[*k].as_str()];
                for &d in same {
                    pending.push(&chunks[d], &hashes[d], v.clone());
                }
            }
            rep.p.chunks_embedded += ids.len();
            report.chunks_embedded += ids.len();
            rep.emit(false);
            if pending.len() >= FLUSH_ROWS {
                tx.send(std::mem::take(&mut pending)).await.map_err(|_| anyhow!("writer stopped"))?;
            }
            i = end;
        }
    }
    if !pending.is_empty() {
        tx.send(pending).await.map_err(|_| anyhow!("writer stopped"))?;
    }
    drop(tx);
    let (_manifest, write_secs) = writer.await??;
    report.timings.write = write_secs;
    if cancelled {
        // Keep what was written; complete files are in the manifest and the
        // rest will be redone on the next sync.
        bail!(Cancelled);
    }

    rep.phase(Phase::BuildingIndex, "updating indexes".to_string());
    let t_index = Instant::now();
    store.update_indices(chunks.len() + to_delete.len()).await?;
    report.timings.index = t_index.elapsed().as_secs_f64();
    finish(paths, index_id, &store, report, rep).await
}

#[derive(Default)]
struct WriteBatch {
    chunks: Vec<Chunk>,
    hashes: Vec<String>,
    vecs: Vec<Vec<f32>>,
}

impl WriteBatch {
    fn push(&mut self, c: &Chunk, h: &str, v: Vec<f32>) {
        self.chunks.push(c.clone());
        self.hashes.push(h.to_string());
        self.vecs.push(v);
    }
    fn len(&self) -> usize {
        self.chunks.len()
    }
    fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Embed a batch; on failure (typically GPU out-of-memory) split it in half,
/// retry, and halve the token budget for the following batches.
fn embed_split(
    e: &mut embedder::Embedder,
    texts: &[String],
    budget: &std::sync::atomic::AtomicUsize,
) -> Result<Vec<Vec<f32>>> {
    match e.embed_documents(texts, texts.len()) {
        Ok(v) => Ok(v),
        Err(err) if texts.len() > 1 => {
            let b = budget.load(Ordering::Relaxed);
            let nb = (b / 2).max(512);
            budget.store(nb, Ordering::Relaxed);
            tracing::warn!("embedding batch of {} failed ({err:#}); splitting, token budget {b} -> {nb}", texts.len());
            let mid = texts.len() / 2;
            let mut a = embed_split(e, &texts[..mid], budget)?;
            a.extend(embed_split(e, &texts[mid..], budget)?);
            Ok(a)
        }
        Err(err) => Err(err),
    }
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
