//! Per-index file manifest: `relative_path -> blake3 hash` (+ size/mtime so
//! unchanged files are not re-hashed on every incremental sync).

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paths::{write_atomic, Paths};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileEntry {
    pub hash: String,
    pub size: u64,
    /// Modification time, nanoseconds since UNIX epoch (0 if unknown).
    pub mtime_ns: u128,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Manifest {
    /// `IndexConfig::fingerprint()` the stored chunks were built with.
    pub fingerprint: String,
    pub files: BTreeMap<String, FileEntry>,
}

impl Manifest {
    pub fn load(paths: &Paths, index_id: &str) -> Result<Self> {
        let file = paths.manifest_file(index_id);
        match std::fs::read(&file) {
            Ok(b) => {
                serde_json::from_slice(&b).with_context(|| format!("parsing {}", file.display()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", file.display())),
        }
    }

    pub fn save(&self, paths: &Paths, index_id: &str) -> Result<()> {
        write_atomic(&paths.manifest_file(index_id), &serde_json::to_vec(self)?)
    }

    pub fn delete(paths: &Paths, index_id: &str) -> Result<()> {
        match std::fs::remove_file(paths.manifest_file(index_id)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
            _ => Ok(()),
        }
    }
}

/// Result of comparing the current file set with the manifest.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Diff {
    pub added: Vec<String>,
    pub modified: Vec<String>,
    pub removed: Vec<String>,
    pub unchanged: usize,
}

impl Diff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.modified.is_empty() && self.removed.is_empty()
    }
    /// Paths that need (re-)chunking.
    pub fn changed(&self) -> impl Iterator<Item = &String> {
        self.added.iter().chain(self.modified.iter())
    }
}

pub fn diff(old: &BTreeMap<String, FileEntry>, new: &BTreeMap<String, FileEntry>) -> Diff {
    let mut d = Diff::default();
    for (path, entry) in new {
        match old.get(path) {
            None => d.added.push(path.clone()),
            Some(o) if o.hash != entry.hash => d.modified.push(path.clone()),
            Some(_) => d.unchanged += 1,
        }
    }
    for path in old.keys() {
        if !new.contains_key(path) {
            d.removed.push(path.clone());
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(h: &str) -> FileEntry {
        FileEntry { hash: h.into(), size: 1, mtime_ns: 0 }
    }

    #[test]
    fn diff_detects_changes() {
        let old: BTreeMap<_, _> =
            [("a".to_string(), e("1")), ("b".into(), e("2")), ("c".into(), e("3"))].into();
        let new: BTreeMap<_, _> =
            [("a".to_string(), e("1")), ("b".into(), e("X")), ("d".into(), e("4"))].into();
        let d = diff(&old, &new);
        assert_eq!(d.added, vec!["d"]);
        assert_eq!(d.modified, vec!["b"]);
        assert_eq!(d.removed, vec!["c"]);
        assert_eq!(d.unchanged, 1);
    }
}
