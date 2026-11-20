//! Export / import of a built index as a single `.lcrag` file, so large
//! codebases (engine source, Linux kernel, ...) can be shared instead of
//! re-embedded from scratch.
//!
//! Format (inside one zstd stream, level 3, lossless):
//! `MAGIC | u64 len + meta JSON | u64 len + manifest JSON | Arrow IPC stream`
//! The Arrow stream holds every chunk (path, lines, text, text hash) with its
//! vector stored as FP16 (rankings are unaffected; vectors are ~half the size).
//!
//! On import, every chunk's text is checked against the importer's own copy
//! of the source: files that are missing or differ are dropped (and re-embedded
//! by the next sync), so a mismatched or tampered export cannot inject text
//! that is not in the local source.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lancedb::arrow::arrow_array::{cast::AsArray, Array, ArrayRef, RecordBatch};
use lancedb::arrow::arrow_cast::cast;
use lancedb::arrow::arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};

use crate::config::{AppConfig, IndexConfig};
use crate::lock::FileLock;
use crate::manifest::Manifest;
use crate::paths::Paths;
use crate::store::{self, ChunkStore};

const MAGIC: &[u8; 8] = b"LCRAG\x00\x01\n";
pub const FORMAT: u32 = 1;
/// Bump when chunk boundaries change (tree-sitter grammars, splitter logic).
pub const CHUNKER_VERSION: u32 = 1;
pub const EXTENSION: &str = "lcrag";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMeta {
    pub format: u32,
    pub app_version: String,
    pub chunker_version: u32,
    pub created_at: DateTime<Utc>,
    /// Free text, e.g. "Unreal Engine 5.8.0 Runtime" or "linux v6.18".
    pub label: Option<String>,
    /// Index settings (root path cleared).
    pub index: IndexConfig,
    pub chunk_count: usize,
    pub file_count: usize,
    pub dim: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportReport {
    pub path: String,
    pub bytes: u64,
    pub chunks: usize,
    pub secs: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub index_id: String,
    pub chunks: usize,
    pub files: usize,
    /// Files whose chunks did not match the local source (re-embedded on next sync).
    pub files_rejected: usize,
    pub warnings: Vec<String>,
    pub secs: f64,
}

fn write_block(w: &mut impl Write, bytes: &[u8]) -> Result<()> {
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(())
}

fn read_block(r: &mut impl Read) -> Result<Vec<u8>> {
    let mut len = [0u8; 8];
    r.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    if len > 4 << 30 {
        bail!("corrupt export (block of {len} bytes)");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Replace the `vector` column with one of a different float type.
fn cast_vectors(batch: &RecordBatch, to: DataType) -> Result<RecordBatch> {
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (f, c) in schema.fields().iter().zip(batch.columns()) {
        if f.name() == "vector" {
            let DataType::FixedSizeList(_, dim) = f.data_type() else { bail!("vector column is not a fixed-size list") };
            let target = DataType::FixedSizeList(Arc::new(Field::new("item", to.clone(), true)), *dim);
            cols.push(cast(c, &target)?);
            fields.push(Field::new("vector", target, f.is_nullable()));
        } else {
            cols.push(c.clone());
            fields.push(f.as_ref().clone());
        }
    }
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), cols)?)
}

/// Read just the header of an export (for the import dialog).
pub fn inspect(path: &Path) -> Result<ExportMeta> {
    let mut r = open_reader(path)?;
    read_meta(&mut r)
}

fn open_reader(path: &Path) -> Result<zstd::Decoder<'static, BufReader<std::fs::File>>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = zstd::Decoder::new(f).context("not a .lcrag file (zstd)")?;
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).context("not a .lcrag file")?;
    if &magic != MAGIC {
        bail!("not a .lcrag file");
    }
    Ok(r)
}

fn read_meta(r: &mut impl Read) -> Result<ExportMeta> {
    let meta: ExportMeta = serde_json::from_slice(&read_block(r)?).context("reading export header")?;
    if meta.format > FORMAT {
        bail!("export format {} is newer than this app supports ({FORMAT}); update the app", meta.format);
    }
    Ok(meta)
}

pub async fn export_index(paths: &Paths, index_id: &str, out: &Path, label: Option<String>) -> Result<ExportReport> {
    let t0 = Instant::now();
    let app = AppConfig::load(paths)?;
    let ix = app.index(index_id).cloned().ok_or_else(|| anyhow!("unknown index '{index_id}'"))?;
    // Shared lock: no sync may write while we read.
    let _lock = FileLock::shared(&paths.lock_file(index_id), Some(Duration::from_secs(30)))
        .await?
        .ok_or_else(|| anyhow!("index '{index_id}' is being indexed; try again when it is done"))?;
    let conn = store::connect(paths).await?;
    let store = ChunkStore::open(&conn, index_id).await?.ok_or_else(|| anyhow!("index '{index_id}' has not been built yet"))?;
    let manifest = Manifest::load(paths, index_id)?;
    let chunk_count = store.count().await?;
    let dim = crate::embedder::model_spec(&ix.embed_model)?.dim;

    let mut shared = ix.clone();
    shared.root_path = String::new();
    shared.last_indexed_at = None;
    let meta = ExportMeta {
        format: FORMAT,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        chunker_version: CHUNKER_VERSION,
        created_at: Utc::now(),
        label: label.filter(|l| !l.trim().is_empty()),
        index: shared,
        chunk_count,
        file_count: manifest.files.len(),
        dim,
    };
    // mtimes are machine-specific: drop them so the importer re-hashes.
    let mut files = manifest.files.clone();
    for e in files.values_mut() {
        e.mtime_ns = 0;
    }
    let manifest_out = Manifest { fingerprint: manifest.fingerprint.clone(), files };

    let tmp = out.with_extension("lcrag.part");
    let f = std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut z = zstd::Encoder::new(BufWriter::new(f), 3)?;
    z.write_all(MAGIC)?;
    write_block(&mut z, &serde_json::to_vec(&meta)?)?;
    write_block(&mut z, &serde_json::to_vec(&manifest_out)?)?;

    let cols = ["path", "start_line", "end_line", "text", "text_hash", "vector"];
    let mut stream = store
        .table
        .query()
        .select(lancedb::query::Select::columns(&cols))
        .execute()
        .await?;
    let mut writer: Option<StreamWriter<&mut zstd::Encoder<'_, BufWriter<std::fs::File>>>> = None;
    let mut written = 0usize;
    while let Some(batch) = stream.try_next().await? {
        let b = cast_vectors(&batch, DataType::Float16)?;
        if writer.is_none() {
            writer = Some(StreamWriter::try_new(&mut z, &b.schema())?);
        }
        writer.as_mut().unwrap().write(&b)?;
        written += b.num_rows();
    }
    if let Some(mut w) = writer {
        w.finish()?;
    }
    z.finish()?.flush()?;
    std::fs::rename(&tmp, out).with_context(|| format!("writing {}", out.display()))?;
    Ok(ExportReport {
        path: out.display().to_string(),
        bytes: std::fs::metadata(out).map(|m| m.len()).unwrap_or(0),
        chunks: written,
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// Import an export as a new index rooted at `root_path` (the importer's own
/// copy of the source). Run a normal sync afterwards to pick up differences.
pub async fn import_index(
    paths: &Paths,
    file: &Path,
    root_path: &str,
    id: &str,
    name: Option<String>,
) -> Result<ImportReport> {
    let t0 = Instant::now();
    let mut r = open_reader(file)?;
    let meta = read_meta(&mut r)?;
    let manifest: Manifest = serde_json::from_slice(&read_block(&mut r)?).context("reading export manifest")?;
    let mut warnings = Vec::new();
    if meta.chunker_version != CHUNKER_VERSION {
        warnings.push(format!(
            "export was chunked by chunker v{} (this app: v{CHUNKER_VERSION}); changed files will be re-chunked differently",
            meta.chunker_version
        ));
    }
    let spec = crate::embedder::model_spec(&meta.index.embed_model)
        .with_context(|| format!("export uses model '{}', which this app does not support", meta.index.embed_model))?;
    if spec.dim != meta.dim {
        bail!("export dimension {} does not match model {} ({})", meta.dim, spec.id, spec.dim);
    }
    if !Path::new(root_path).is_dir() {
        bail!("'{root_path}' is not a folder");
    }

    let mut cfg = meta.index.clone();
    cfg.id = id.to_string();
    cfg.name = name.filter(|n| !n.trim().is_empty()).unwrap_or_else(|| meta.label.clone().unwrap_or(cfg.name.clone()));
    cfg.root_path = root_path.to_string();
    cfg.normalize()?;
    if AppConfig::load(paths)?.index(&cfg.id).is_some() {
        bail!("an index with id '{}' already exists", cfg.id);
    }

    let _lock = FileLock::exclusive(&paths.lock_file(&cfg.id), Some(Duration::from_secs(10)))
        .await?
        .ok_or_else(|| anyhow!("index '{}' is busy", cfg.id))?;
    let conn = store::connect(paths).await?;
    ChunkStore::drop(&conn, &cfg.id).await?;
    let store = ChunkStore::open_or_create(&conn, &cfg.id, spec.dim).await?;

    // Load chunks; keep their text per file for verification.
    let reader = StreamReader::try_new(r, None).context("reading export chunks")?;
    let mut texts: HashMap<String, Vec<String>> = HashMap::new();
    for batch in reader {
        let batch = batch.context("reading export chunks")?;
        let b = cast_vectors(&batch, DataType::Float32)?;
        let path_col = b.column_by_name("path").context("path column")?.as_string::<i32>().clone();
        let text_col = b.column_by_name("text").context("text column")?.as_string::<i32>().clone();
        for i in 0..b.num_rows() {
            if !path_col.is_null(i) {
                texts.entry(path_col.value(i).to_string()).or_default().push(text_col.value(i).to_string());
            }
        }
        store.table.add(b).execute().await.context("storing imported chunks")?;
    }

    // Verify against the local source: every chunk's text must occur in the
    // local file. Mismatching / missing files are dropped.
    let root = PathBuf::from(root_path);
    let rejected: Vec<String> = tokio::task::spawn_blocking(move || {
        use rayon::prelude::*;
        texts
            .par_iter()
            .filter_map(|(rel, list)| {
                let local = std::fs::read(root.join(rel)).ok().map(|b| String::from_utf8_lossy(&b).into_owned());
                let ok = local.is_some_and(|s| list.iter().all(|t| s.contains(t.as_str())));
                (!ok).then(|| rel.clone())
            })
            .collect()
    })
    .await?;
    if !rejected.is_empty() {
        store.delete_paths(&rejected).await?;
    }
    let rejected_set: HashSet<&String> = rejected.iter().collect();
    let files: BTreeMap<_, _> = manifest
        .files
        .into_iter()
        .filter(|(p, _)| !rejected_set.contains(p))
        .map(|(p, mut e)| {
            e.mtime_ns = 0; // forces a re-hash of the local file on the next sync
            (p, e)
        })
        .collect();
    let file_count = files.len();
    Manifest { fingerprint: cfg.fingerprint(), files }.save(paths, &cfg.id)?;

    store.rebuild_indices(&mut |_, _, _| {}).await?;
    let total = store.count().await?;
    cfg.last_indexed_at = Some(Utc::now());
    cfg.chunk_count = total as u64;
    let to_save = cfg.clone();
    AppConfig::update(paths, move |c| {
        c.indexes.push(to_save);
        Ok(())
    })?;
    Ok(ImportReport {
        index_id: cfg.id,
        chunks: total,
        files: file_count,
        files_rejected: rejected.len(),
        warnings,
        secs: t0.elapsed().as_secs_f64(),
    })
}
