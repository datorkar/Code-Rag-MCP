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
    /// Estimated seconds left for embedding, based on token throughput (chunks
    /// are processed short-to-long, so chunks/s alone underestimates).
    pub eta_secs: Option<f64>,
    /// Chunking phase: files chunked so far / files to chunk.
    pub files_chunked: usize,
    pub files_to_chunk: usize,
    /// Index phase: completed steps / total steps (0/0 outside that phase).
    pub index_step: usize,
    pub index_steps: usize,
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
    /// Index phase split: compaction, keyword index, path index, cleanup.
    pub index_steps: [f64; 4],
}

impl std::fmt::Display for PhaseTimings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "scan {:.1}s · chunk {:.1}s · model {:.1}s · embed {:.1}s · write {:.1}s · index {:.1}s",
            self.scan, self.chunk, self.load_model, self.embed, self.write, self.index
        )?;
        let [c, k, p, u] = self.index_steps;
        if c + k + p + u > 0.0 {
            write!(f, " [compact {c:.1}s · keyword {k:.1}s · path {p:.1}s · cleanup {u:.1}s]")?;
        }
        Ok(())
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
    tokens_total: usize,
    tokens_done: usize,
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
                if self.tokens_done > 0 {
                    let rate = self.tokens_done as f64 / s;
                    self.p.eta_secs = Some(self.tokens_total.saturating_sub(self.tokens_done) as f64 / rate);
                }
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

/// CPU pool for hashing/chunking: all logical threads, or all but 2 in
/// low-impact mode (same rule as ONNX Runtime's threads).
fn cpu_pool(threads: Option<usize>) -> Result<rayon::ThreadPool> {
    let mut b = rayon::ThreadPoolBuilder::new().thread_name(|i| format!("rag-cpu-{i}"));
    if let Some(n) = threads {
        b = b.num_threads(n);
    }
    Ok(b.build()?)
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
            eta_secs: None,
            files_chunked: 0,
            files_to_chunk: 0,
            index_step: 0,
            index_steps: 0,
            message: None,
        },
        start: Instant::now(),
        embed_start: None,
        tokens_total: 0,
        tokens_done: 0,
        cb: opts.progress.clone(),
        last_emit: Instant::now(),
    };
    let _priority = (AppConfig::load(paths).map(|c| c.resource_mode != crate::config::ResourceMode::Max).unwrap_or(true))
        .then(crate::runtime::BackgroundPriority::enter);
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
        let threads = app.intra_threads();
        tokio::task::spawn_blocking(move || -> Result<_> {
            let pool = cpu_pool(threads)?;
            let files = walker::walk(&cfg, &cancel)?;
            let total = files.len();
            let last = std::sync::Mutex::new(Instant::now());
            let hashed = pool.install(|| walker::hash_all(&files, &prev, &cancel, |n| {
                if let Some(cb) = &cb {
                    let mut last = last.lock().unwrap();
                    if last.elapsed() >= Duration::from_millis(100) || n == total {
                        *last = Instant::now();
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
                            eta_secs: None,
                            files_chunked: 0,
                            files_to_chunk: 0,
                            index_step: 0,
                            index_steps: 0,
                            message: None,
                        });
                    }
                }
            }))?;
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
            let _ = store
                .rebuild_indices(&mut |step, steps, label| {
                    rep.p.index_step = step;
                    rep.p.index_steps = steps;
                    rep.phase(Phase::BuildingIndex, label.to_string());
                })
                .await?;
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
    // Never produce chunks longer than the model can read (covers configs
    // saved before this check or edited by hand).
    let chunk_size = cfg.chunk_size.min(spec.max_chunk_chars);
    let overlap = if cfg.chunk_overlap >= chunk_size { chunk_size / 10 } else { cfg.chunk_overlap };
    let chunks: Vec<Chunk> = {
        let cancel = cancel.clone();
        let threads = app.intra_threads();
        let (cb, base) = (rep.cb.clone(), rep.p.clone());
        tokio::task::spawn_blocking(move || -> Result<Vec<Chunk>> {
            // Files are independent: chunk them in parallel, reporting progress.
            use rayon::prelude::*;
            let total = jobs.len();
            let done = std::sync::atomic::AtomicUsize::new(0);
            let last = std::sync::Mutex::new(Instant::now());
            let report = |n: usize| {
                if let Some(cb) = &cb {
                    let mut last = last.lock().unwrap();
                    if last.elapsed() >= Duration::from_millis(100) || n == total {
                        *last = Instant::now();
                        cb(&Progress { phase: Phase::Chunking, files_chunked: n, files_to_chunk: total, ..base.clone() });
                    }
                }
            };
            let per_file: Vec<Vec<Chunk>> = cpu_pool(threads)?.install(|| {
                jobs.par_iter()
                    .map(|(rel, path)| {
                        if cancel.load(Ordering::Relaxed) {
                            return Vec::new();
                        }
                        let out = match std::fs::read(path) {
                            Ok(bytes) => chunk_text(rel, &String::from_utf8_lossy(&bytes), chunk_size, overlap),
                            Err(e) => {
                                tracing::warn!("cannot read {}: {e}", path.display());
                                Vec::new()
                            }
                        };
                        report(done.fetch_add(1, Ordering::Relaxed) + 1);
                        out
                    })
                    .collect()
            });
            check(&cancel)?;
            Ok(per_file.into_iter().flatten().collect())
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
    let max_len = embedder::max_length_for_chunk_size(chunk_size);
    // Longest first: out-of-memory shows up immediately, and the token-based
    // ETA is accurate from the start.
    unique.sort_by_key(|&i| std::cmp::Reverse(chunks[i].text.len()));
    let tokens: Vec<usize> =
        unique.iter().map(|&i| crate::batching::est_tokens(chunks[i].text.len(), max_len)).collect();
    rep.p.chunks_total = unique.len();
    rep.tokens_total = tokens.iter().sum();

    let mut cancelled = false;
    if !unique.is_empty() {
        let t_model = Instant::now();
        rep.phase(Phase::LoadingModel, format!("loading {}", spec.id));
        let gpu = !matches!(app.execution_provider, crate::ExecutionProvider::Cpu);
        let low = app.low_impact();
        let threads = app.intra_threads();
        let n_workers = match app.embed_workers {
            0 if gpu && !low => crate::batching::GPU_WORKERS,
            0 => 1,
            n => n.min(8),
        };
        let mut workers = Vec::with_capacity(n_workers);
        for w in 0..n_workers {
            let (model, ep, dir) = (cfg.embed_model.clone(), app.execution_provider, paths.models_dir());
            let variant = cfg.model_variant(&app);
            workers.push(
                tokio::task::spawn_blocking(move || embedder::cached_worker(&model, ep, &dir, max_len, variant, threads, w))
                    .await??,
            );
        }
        report.timings.load_model = t_model.elapsed().as_secs_f64();

        // The budget is the total across workers: each in-flight batch gets its share.
        let total_budget = match app.embed_token_budget {
            0 if gpu && !low => crate::batching::GPU_TOKEN_BUDGET,
            0 => crate::batching::CPU_TOKEN_BUDGET,
            n => n,
        };
        let budget = Arc::new(std::sync::atomic::AtomicUsize::new((total_budget / n_workers).max(512)));
        let reused_msg = (report.chunks_reused > 0).then(|| format!("{} chunks reused", report.chunks_reused));
        rep.phase(Phase::Embedding, reused_msg);
        rep.embed_start = Some(Instant::now());

        // Up to `n_workers` batches in flight; each worker session is used by
        // one batch at a time. Results are written as they complete.
        let mut inflight = futures::stream::FuturesUnordered::new();
        let mut free: Vec<usize> = (0..workers.len()).collect();
        let mut i = 0;
        let embed_t0 = Instant::now();
        loop {
            while !cancelled && i < unique.len() && !free.is_empty() {
                if cancel.load(Ordering::Relaxed) {
                    cancelled = true;
                    break;
                }
                let w = free.pop().unwrap();
                let end = crate::batching::next_batch_end(&tokens, i, budget.load(Ordering::Relaxed), app.embed_batch_size);
                let ids: Vec<usize> = unique[i..end].to_vec();
                let tok: usize = tokens[i..end].iter().sum();
                let texts: Vec<String> = ids.iter().map(|&k| chunks[k].embedding_input()).collect();
                let (e, b) = (workers[w].clone(), budget.clone());
                inflight.push(async move {
                    let r = tokio::task::spawn_blocking(move || embed_split(&mut e.lock().unwrap(), &texts, &b)).await;
                    (w, ids, tok, r)
                });
                i = end;
            }
            let Some((w, ids, tok, r)) = futures::StreamExt::next(&mut inflight).await else { break };
            free.push(w);
            let out = r??;
            for (k, v) in ids.iter().zip(out) {
                for &d in &dups[hashes[*k].as_str()] {
                    pending.push(&chunks[d], &hashes[d], v.clone());
                }
            }
            rep.p.chunks_embedded += ids.len();
            rep.tokens_done += tok;
            report.chunks_embedded += ids.len();
            rep.emit(false);
            if pending.len() >= FLUSH_ROWS {
                tx.send(std::mem::take(&mut pending)).await.map_err(|_| anyhow!("writer stopped"))?;
            }
        }
        report.timings.embed = embed_t0.elapsed().as_secs_f64();
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
    report.timings.index_steps = store
        .update_indices(chunks.len() + to_delete.len(), &mut |step, steps, label| {
            rep.p.index_step = step;
            rep.p.index_steps = steps;
            rep.phase(Phase::BuildingIndex, label.to_string());
        })
        .await?;
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
