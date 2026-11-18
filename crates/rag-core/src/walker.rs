//! Gitignore-aware file walking + blake3 hashing.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::UNIX_EPOCH;

use anyhow::{bail, Result};
use ignore::WalkBuilder;

use crate::config::IndexConfig;
use crate::manifest::FileEntry;

#[derive(Debug, Clone)]
pub struct WalkedFile {
    /// Forward-slash relative path (stable across OSes; used as the DB key).
    pub rel_path: String,
    pub abs_path: PathBuf,
    pub size: u64,
    pub mtime_ns: u128,
}

/// List every file under `root_path` that matches the index filters.
pub fn walk(cfg: &IndexConfig, cancel: &AtomicBool) -> Result<Vec<WalkedFile>> {
    let root = PathBuf::from(&cfg.root_path);
    if !root.is_dir() {
        bail!("root_path '{}' is not a directory", cfg.root_path);
    }
    let ignore_dirs: HashSet<String> =
        cfg.ignore_dirs.iter().map(|d| d.to_ascii_lowercase()).collect();
    let exts: HashSet<String> =
        cfg.allowed_extensions.iter().map(|e| e.trim_start_matches('.').to_ascii_lowercase()).collect();

    let walker = WalkBuilder::new(&root)
        .hidden(true)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .require_git(false)
        .follow_links(false)
        .filter_entry(move |e| {
            let is_dir = e.file_type().is_some_and(|t| t.is_dir());
            if !is_dir || e.depth() == 0 {
                return true;
            }
            let name = e.file_name().to_string_lossy().to_ascii_lowercase();
            !ignore_dirs.contains(&name)
        })
        .build();

    let mut out = Vec::new();
    for entry in walker {
        if cancel.load(Ordering::Relaxed) {
            bail!(crate::Cancelled);
        }
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!("walk error: {err}");
                continue;
            }
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        let ext_ok = path
            .extension()
            .map(|x| exts.contains(&x.to_string_lossy().to_ascii_lowercase()))
            .unwrap_or(false);
        if !ext_ok {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if meta.len() > cfg.max_file_bytes || meta.len() == 0 {
            continue;
        }
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        out.push(WalkedFile {
            rel_path: rel_path(&root, path),
            abs_path: path.to_path_buf(),
            size: meta.len(),
            mtime_ns,
        });
    }
    out.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Ok(out)
}

fn rel_path(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

pub fn hash_file(path: &Path) -> std::io::Result<String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update_mmap(path)?;
    Ok(hasher.finalize().to_hex().to_string())
}

/// Build the new manifest file map, reusing hashes for files whose size and
/// mtime are unchanged. `on_file` is called with the running count.
pub fn hash_all(
    files: &[WalkedFile],
    previous: &BTreeMap<String, FileEntry>,
    cancel: &AtomicBool,
    mut on_file: impl FnMut(usize),
) -> Result<BTreeMap<String, FileEntry>> {
    let mut out = BTreeMap::new();
    for (i, f) in files.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            bail!(crate::Cancelled);
        }
        let reuse = previous
            .get(&f.rel_path)
            .filter(|p| p.size == f.size && p.mtime_ns == f.mtime_ns && f.mtime_ns != 0);
        let hash = match reuse {
            Some(p) => p.hash.clone(),
            None => match hash_file(&f.abs_path) {
                Ok(h) => h,
                Err(e) => {
                    tracing::warn!("cannot hash {}: {e}", f.abs_path.display());
                    continue;
                }
            },
        };
        out.insert(
            f.rel_path.clone(),
            FileEntry { hash, size: f.size, mtime_ns: f.mtime_ns },
        );
        if i % 256 == 0 {
            on_file(i + 1);
        }
    }
    on_file(files.len());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_respects_filters() {
        let dir = tempfile::tempdir().unwrap();
        let r = dir.path();
        std::fs::create_dir_all(r.join("Source/Private")).unwrap();
        std::fs::create_dir_all(r.join("Intermediate")).unwrap();
        std::fs::create_dir_all(r.join("Source/Linux")).unwrap();
        std::fs::write(r.join("Source/Private/A.cpp"), "int a;").unwrap();
        std::fs::write(r.join("Source/Private/A.H"), "int a;").unwrap();
        std::fs::write(r.join("Source/readme.md"), "x").unwrap();
        std::fs::write(r.join("Intermediate/Gen.cpp"), "int g;").unwrap();
        std::fs::write(r.join("Source/Linux/L.cpp"), "int l;").unwrap();
        std::fs::write(r.join(".gitignore"), "ignored.cpp\n").unwrap();
        std::fs::write(r.join("ignored.cpp"), "int i;").unwrap();

        let cfg = IndexConfig {
            root_path: r.to_string_lossy().into(),
            allowed_extensions: vec![".cpp".into(), ".h".into()],
            ignore_dirs: vec!["Intermediate".into(), "linux".into()],
            ..Default::default()
        };
        let files = walk(&cfg, &AtomicBool::new(false)).unwrap();
        let names: Vec<_> = files.iter().map(|f| f.rel_path.as_str()).collect();
        assert_eq!(names, vec!["Source/Private/A.H", "Source/Private/A.cpp"]);
    }
}
