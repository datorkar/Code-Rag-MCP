//! Advisory cross-process file locks (one per index + one for config.json),
//! using the standard library's `File::lock` family (LockFileEx / flock).

use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// RAII guard; the lock is released when dropped.
#[derive(Debug)]
pub struct FileLock {
    file: File,
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
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
        file.lock().context("acquiring exclusive lock")?;
        Ok(Self { file })
    }

    pub fn try_exclusive(path: &Path) -> Result<Option<Self>> {
        let file = open(path)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { file })),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(e)) => Err(e).context("acquiring exclusive lock"),
        }
    }

    pub fn try_shared(path: &Path) -> Result<Option<Self>> {
        let file = open(path)?;
        match file.try_lock_shared() {
            Ok(()) => Ok(Some(Self { file })),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(e)) => Err(e).context("acquiring shared lock"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusive_excludes_shared_and_exclusive() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.lock");
        let ex = FileLock::try_exclusive(&p).unwrap().expect("first exclusive");
        assert!(FileLock::try_exclusive(&p).unwrap().is_none());
        assert!(FileLock::try_shared(&p).unwrap().is_none());
        drop(ex);
        let s1 = FileLock::try_shared(&p).unwrap().expect("shared");
        let _s2 = FileLock::try_shared(&p).unwrap().expect("second shared");
        assert!(FileLock::try_exclusive(&p).unwrap().is_none());
        drop(s1);
    }
}
