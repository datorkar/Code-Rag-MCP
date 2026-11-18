//! Advisory cross-process file locks (one per index + one for config.json).

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use fs2::FileExt;

/// RAII guard; the lock is released when dropped.
#[derive(Debug)]
pub struct FileLock {
    file: File,
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

fn open(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .with_context(|| format!("opening lock file {}", path.display()))
}

impl FileLock {
    pub fn exclusive_blocking(path: &Path) -> Result<Self> {
        let file = open(path)?;
        file.lock_exclusive().context("acquiring exclusive lock")?;
        Ok(Self { file })
    }

    pub fn try_exclusive(path: &Path) -> Result<Option<Self>> {
        let file = open(path)?;
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Some(Self { file })),
            Err(_) => Ok(None),
        }
    }

    pub fn try_shared(path: &Path) -> Result<Option<Self>> {
        let file = open(path)?;
        match FileExt::try_lock_shared(&file) {
            Ok(()) => Ok(Some(Self { file })),
            Err(_) => Ok(None),
        }
    }

    /// Poll for an exclusive lock without blocking the async runtime.
    /// `timeout = None` waits forever.
    pub async fn exclusive(path: &Path, timeout: Option<Duration>) -> Result<Option<Self>> {
        Self::poll(path, timeout, Self::try_exclusive).await
    }

    pub async fn shared(path: &Path, timeout: Option<Duration>) -> Result<Option<Self>> {
        Self::poll(path, timeout, Self::try_shared).await
    }

    async fn poll(
        path: &Path,
        timeout: Option<Duration>,
        f: fn(&Path) -> Result<Option<Self>>,
    ) -> Result<Option<Self>> {
        let start = Instant::now();
        loop {
            if let Some(l) = f(path)? {
                return Ok(Some(l));
            }
            if timeout.is_some_and(|t| start.elapsed() >= t) {
                return Ok(None);
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
    }
}
