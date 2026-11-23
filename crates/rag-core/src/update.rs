//! Update check against the project's GitHub Releases (no installer is
//! downloaded; the app links to the release page). Sends only a plain GET with
//! a generic User-Agent.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// GitHub `owner/repo` that publishes releases.
pub const REPO: &str = "datorkar/Code-Rag-MCP";

#[derive(Debug, Clone, Serialize)]
pub struct UpdateInfo {
    pub current: String,
    pub latest: String,
    pub available: bool,
    /// Release page.
    pub url: String,
    /// Direct link to the setup .exe, when present.
    pub download_url: Option<String>,
    pub notes: String,
    pub published_at: Option<String>,
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}

/// `0.1.10` > `0.1.9`; a leading `v` and any `-suffix` are ignored.
pub fn parse_version(v: &str) -> Option<(u64, u64, u64)> {
    let v = v.trim().trim_start_matches(['v', 'V']);
    let core = v.split(['-', '+']).next()?;
    let mut it = core.split('.').map(|p| p.parse::<u64>().ok());
    Some((it.next()??, it.next().flatten().unwrap_or(0), it.next().flatten().unwrap_or(0)))
}

pub fn is_newer(latest: &str, current: &str) -> bool {
    match (parse_version(latest), parse_version(current)) {
        (Some(l), Some(c)) => l > c,
        _ => false,
    }
}

/// Blocking: ask GitHub for the latest published release and compare it
/// with `current` (the running app's version).
pub fn check(current: &str) -> Result<UpdateInfo> {
    use ureq::tls::{RootCerts, TlsConfig, TlsProvider};
    // OS trust store (works behind corporate TLS-inspecting proxies).
    let tls = TlsConfig::builder().provider(TlsProvider::NativeTls).root_certs(RootCerts::PlatformVerifier).build();
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(tls)
        .timeout_global(Some(std::time::Duration::from_secs(15)))
        .build()
        .into();
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let mut resp = agent
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "local-code-rag-update-check")
        .call()
        .context("contacting GitHub")?;
    let body = resp.body_mut().read_to_string().context("reading GitHub response")?;
    let r: Release = serde_json::from_str(&body).context("parsing GitHub release")?;
    let latest = r.tag_name.trim_start_matches(['v', 'V']).to_string();
    let download_url = r
        .assets
        .iter()
        .find(|a| a.name.to_ascii_lowercase().ends_with("-setup.exe"))
        .map(|a| a.browser_download_url.clone());
    Ok(UpdateInfo {
        current: current.to_string(),
        available: !r.draft && !r.prerelease && is_newer(&latest, current),
        latest,
        url: r.html_url,
        download_url,
        notes: r.body.unwrap_or_default(),
        published_at: r.published_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_compare() {
        assert!(is_newer("v0.1.10", "0.1.9"));
        assert!(is_newer("0.2.0", "0.1.99"));
        assert!(!is_newer("v0.1.1", "0.1.1"));
        assert!(!is_newer("0.1.0", "0.1.1"));
        assert!(!is_newer("garbage", "0.1.1"));
        assert_eq!(parse_version("v1.2.3-beta"), Some((1, 2, 3)));
    }

    /// Hits the real GitHub API.
    #[test]
    #[ignore]
    fn live_check() {
        let u = check("0.0.1").unwrap();
        assert!(u.available, "{u:?}");
        println!("{u:?}");
    }
}
