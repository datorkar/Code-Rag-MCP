//! Tauri v2 backend: thin command layer over `rag-core`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rag_core::embedder::{self, ModelSpec};
use rag_core::{
    AppConfig, Cancelled, Engine, ExecutionProvider, IndexConfig, IndexInfo, Progress, SaveResult,
    SyncOptions,
};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

const MCP_SERVER_NAME: &str = "local-code-rag";

struct AppState {
    engine: Engine,
    jobs: Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// Result of the last `check_update`, installed by `install_update`.
    pending_update: Mutex<Option<tauri_plugin_updater::Update>>,
}

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    format!("{e:#}")
}

#[derive(Serialize, Deserialize, Clone)]
struct GlobalSettings {
    active_embed_model: String,
    execution_provider: ExecutionProvider,
    embed_batch_size: usize,
    #[serde(default)]
    embed_token_budget: usize,
    #[serde(default)]
    force_fp32: bool,
    #[serde(default)]
    resource_mode: rag_core::ResourceMode,
    reranker_model: Option<String>,
    rerank_candidates: usize,
    auto_sync_max_files: usize,
    snippet_max_lines: usize,
    #[serde(default)]
    data_dir: String,
    #[serde(default)]
    gpu_features: Vec<String>,
    #[serde(default = "yes")]
    check_updates: bool,
    /// Running app version (read-only).
    #[serde(default)]
    app_version: String,
    #[serde(default)]
    default_precision: rag_core::Precision,
    #[serde(default)]
    setup_completed: bool,
    /// Number of configured indexes (read-only; the setup shows on first launch only).
    #[serde(default)]
    index_count: usize,
}

fn yes() -> bool {
    true
}

#[derive(Serialize, Clone)]
struct FinishedEvent {
    index_id: String,
    ok: bool,
    cancelled: bool,
    message: String,
}

#[derive(Serialize)]
struct ClaudeSnippets {
    exe_path: String,
    exe_exists: bool,
    command: String,
    json: String,
    claude_md: String,
}

#[tauri::command]
fn list_indexes(state: State<'_, AppState>) -> CmdResult<Vec<IndexInfo>> {
    let mut v = state.engine.list().map_err(err)?;
    // A job started from this window is "indexing" even while it waits on the lock.
    let jobs = state.jobs.lock().unwrap();
    for i in &mut v {
        if jobs.contains_key(&i.config.id) {
            i.status = rag_core::IndexStatus::Indexing;
        }
    }
    Ok(v)
}

#[tauri::command]
fn list_models() -> Vec<ModelSpec> {
    embedder::supported_models()
}

#[tauri::command]
fn list_rerankers() -> Vec<String> {
    embedder::supported_rerankers()
}

#[tauri::command]
fn get_settings(state: State<'_, AppState>) -> CmdResult<GlobalSettings> {
    let c = state.engine.config().map_err(err)?;
    Ok(settings_of(&c, &state.engine))
}

fn settings_of(c: &AppConfig, engine: &Engine) -> GlobalSettings {
    let gpu: Vec<String> = if cfg!(windows) {
        vec!["winml".into(), "directml".into()]
    } else {
        vec!["cuda".into()]
    };
    GlobalSettings {
        active_embed_model: c.active_embed_model.clone(),
        execution_provider: c.execution_provider,
        embed_batch_size: c.embed_batch_size,
        embed_token_budget: c.embed_token_budget,
        force_fp32: c.force_fp32,
        resource_mode: c.resource_mode,
        reranker_model: c.reranker_model.clone(),
        rerank_candidates: c.rerank_candidates,
        auto_sync_max_files: c.auto_sync_max_files,
        snippet_max_lines: c.snippet_max_lines,
        data_dir: engine.paths.root().display().to_string(),
        gpu_features: gpu,
        check_updates: c.check_updates,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        default_precision: c.default_precision,
        setup_completed: c.setup_completed,
        index_count: c.indexes.len(),
    }
}

#[tauri::command]
fn save_settings(state: State<'_, AppState>, settings: GlobalSettings) -> CmdResult<GlobalSettings> {
    embedder::resolve_model(&settings.active_embed_model).map_err(err)?;
    let reranker = settings.reranker_model.clone().filter(|m| !m.trim().is_empty());
    if let Some(r) = &reranker {
        embedder::resolve_reranker(r).map_err(err)?;
    }
    let c = state
        .engine
        .update_settings(|c| {
            c.active_embed_model = settings.active_embed_model.clone();
            c.execution_provider = settings.execution_provider;
            c.embed_batch_size = settings.embed_batch_size.min(512);
            c.embed_token_budget = settings.embed_token_budget;
            c.force_fp32 = settings.force_fp32;
            c.resource_mode = settings.resource_mode;
            c.reranker_model = reranker.clone();
            c.rerank_candidates = settings.rerank_candidates.clamp(5, 200);
            c.auto_sync_max_files = settings.auto_sync_max_files;
            c.snippet_max_lines = settings.snippet_max_lines;
            c.check_updates = settings.check_updates;
            c.default_precision = settings.default_precision;
            c.setup_completed = settings.setup_completed;
        })
        .map_err(err)?;
    Ok(settings_of(&c, &state.engine))
}

#[tauri::command]
fn save_index(
    state: State<'_, AppState>,
    index: IndexConfig,
    original_id: Option<String>,
) -> CmdResult<SaveResult> {
    state.engine.save_index(index, original_id.as_deref()).map_err(err)
}

#[tauri::command]
async fn delete_index(state: State<'_, AppState>, index_id: String) -> CmdResult<()> {
    if let Some(c) = state.jobs.lock().unwrap().get(&index_id) {
        c.store(true, Ordering::Relaxed);
    }
    state.engine.delete_index(&index_id).await.map_err(err)
}

#[tauri::command]
fn start_index(
    app: AppHandle,
    state: State<'_, AppState>,
    index_id: String,
    force_full: bool,
) -> CmdResult<()> {
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut jobs = state.jobs.lock().unwrap();
        if jobs.contains_key(&index_id) {
            return Err(format!("index '{index_id}' is already being indexed"));
        }
        jobs.insert(index_id.clone(), cancel.clone());
    }
    let engine = state.engine.clone();
    let emitter = app.clone();
    let progress: rag_core::ProgressFn = Arc::new(move |p: &Progress| {
        let _ = emitter.emit("index-progress", p);
    });
    tauri::async_runtime::spawn(async move {
        let opts = SyncOptions { force_full, cancel, progress: Some(progress), ..Default::default() };
        let res = engine.sync(&index_id, opts).await;
        let ev = match res {
            Ok(r) => FinishedEvent {
                index_id: index_id.clone(),
                ok: true,
                cancelled: false,
                message: format!(
                    "+{} ~{} -{} files · {} chunks embedded, {} reused · {} total · {:.1}s ({})",
                    r.files_added, r.files_modified, r.files_removed, r.chunks_embedded,
                    r.chunks_reused, r.total_chunks, r.duration_secs, r.timings
                ),
            },
            Err(e) => FinishedEvent {
                index_id: index_id.clone(),
                ok: false,
                cancelled: e.is::<Cancelled>(),
                message: format!("{e:#}"),
            },
        };
        let state = app.state::<AppState>();
        let idle = {
            let mut jobs = state.jobs.lock().unwrap();
            jobs.remove(&index_id);
            jobs.keys().all(|k| is_winml_job(k))
        };
        if idle {
            // The manager never searches: release model sessions (RAM/VRAM).
            rag_core::embedder::clear_caches();
        }
        let _ = app.emit("index-finished", ev);
    });
    Ok(())
}

#[derive(Serialize)]
struct HardwareSummary {
    hardware: rag_core::hardware::Hardware,
    recommendation: rag_core::hardware::Recommendation,
}

/// CPU / RAM / GPUs (with VRAM) and the matching model + settings for the first-run setup.
#[tauri::command]
async fn hardware_summary() -> CmdResult<HardwareSummary> {
    tauri::async_runtime::spawn_blocking(|| {
        let hardware = rag_core::hardware::detect();
        let recommendation = rag_core::hardware::recommend(&hardware);
        HardwareSummary { hardware, recommendation }
    })
    .await
    .map_err(err)
}

#[derive(Serialize)]
struct GpuStatus {
    runtime: Option<rag_core::runtime::RuntimeInfo>,
    runtime_error: Option<String>,
    /// Windows ML catalog (Windows builds only).
    providers: Vec<ProviderStatus>,
    providers_error: Option<String>,
    windows_ml: bool,
}

#[derive(Serialize, Clone)]
struct ProviderStatus {
    name: String,
    version: String,
    ready_state: String,
    certified: bool,
    registered: bool,
}

#[cfg(windows)]
#[derive(Serialize, Clone)]
struct WinmlProgress {
    provider: String,
    progress: f64,
    done: bool,
    ok: bool,
    message: Option<String>,
}

#[tauri::command]
async fn gpu_status() -> CmdResult<GpuStatus> {
    tauri::async_runtime::spawn_blocking(|| {
        let (runtime, runtime_error) = match rag_core::runtime::info() {
            Ok(i) => (Some(i), None),
            Err(e) => (None, Some(format!("{e:#}"))),
        };
        #[allow(unused_mut)]
        let (mut providers, mut providers_error) = (Vec::new(), None);
        #[cfg(windows)]
        match rag_core::runtime::winml_providers() {
            Ok(list) => {
                let reg = runtime.as_ref().map(|r| r.registered_providers.clone()).unwrap_or_default();
                providers = list
                    .into_iter()
                    .map(|p| ProviderStatus {
                        registered: reg.contains(&p.name),
                        ready_state: serde_json::to_value(p.ready_state)
                            .ok()
                            .and_then(|v| v.as_str().map(String::from))
                            .unwrap_or_default(),
                        name: p.name,
                        version: p.version,
                        certified: p.certified,
                    })
                    .collect();
            }
            Err(e) => providers_error = Some(format!("{e:#}")),
        }
        GpuStatus { runtime, runtime_error, providers, providers_error, windows_ml: cfg!(windows) }
    })
    .await
    .map_err(err)
}

/// Download + register Windows ML providers (`provider = None`: every certified
/// provider that is not ready yet). Progress: `winml-progress` events.
#[tauri::command]
fn winml_install(app: AppHandle, state: State<'_, AppState>, provider: Option<String>) -> CmdResult<()> {
    #[cfg(not(windows))]
    {
        let _ = (app, state, provider);
        Err("Windows ML is only available on Windows".into())
    }
    #[cfg(windows)]
    {
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut jobs = state.jobs.lock().unwrap();
            if jobs.contains_key(WINML_JOB) {
                return Err("a Windows ML install is already running".into());
            }
            jobs.insert(WINML_JOB.into(), cancel.clone());
        }
        tauri::async_runtime::spawn_blocking(move || {
            let targets: Vec<String> = match &provider {
                Some(p) => vec![p.clone()],
                None => rag_core::runtime::winml_providers()
                    .map(|l| {
                        l.into_iter()
                            .filter(|p| p.certified && p.ready_state != rag_core::winml::ReadyState::Ready)
                            .map(|p| p.name)
                            .collect()
                    })
                    .unwrap_or_default(),
            };
            for name in targets {
                let emit = |progress: f64, done: bool, ok: bool, message: Option<String>| {
                    let _ = app.emit(
                        "winml-progress",
                        WinmlProgress { provider: name.clone(), progress, done, ok, message },
                    );
                };
                emit(0.0, false, true, None);
                let res = rag_core::runtime::install_winml_provider(&name, &cancel, &|p| emit(p, false, true, None));
                match res {
                    Ok(_) => emit(100.0, true, true, None),
                    Err(e) => {
                        let cancelled = e.is::<Cancelled>();
                        emit(0.0, true, false, Some(if cancelled { "cancelled".into() } else { format!("{e:#}") }));
                        if cancelled {
                            break;
                        }
                    }
                }
            }
            app.state::<AppState>().jobs.lock().unwrap().remove(WINML_JOB);
            // Also sent when there was nothing to install (no progress events).
            let _ = app.emit("winml-install-finished", ());
        });
        Ok(())
    }
}

#[cfg(windows)]
const WINML_JOB: &str = "__winml_install__";

/// The Windows ML install shares the job map with index jobs.
fn is_winml_job(key: &str) -> bool {
    key == "__winml_install__"
}

#[tauri::command]
fn winml_cancel(state: State<'_, AppState>) -> bool {
    #[cfg(windows)]
    if let Some(c) = state.jobs.lock().unwrap().get(WINML_JOB) {
        c.store(true, Ordering::Relaxed);
        return true;
    }
    let _ = state;
    false
}

#[tauri::command]
fn cancel_index(state: State<'_, AppState>, index_id: String) -> bool {
    match state.jobs.lock().unwrap().get(&index_id) {
        Some(c) => {
            c.store(true, Ordering::Relaxed);
            true
        }
        None => false,
    }
}

fn mcp_exe_path() -> PathBuf {
    let name = if cfg!(windows) { "rag-mcp.exe" } else { "rag-mcp" };
    let dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_default();
    dir.join(name)
}

fn claude_md(indexes: &[IndexConfig]) -> String {
    let mut s = String::from(
        "## Local code search (MCP: local-code-rag)\n\n\
         Use the `local-code-rag` MCP tools before grepping large codebases:\n\
         - `list_indexes` – show available indexes\n\
         - `search_index(index_id, query, limit?)` – hybrid semantic + keyword search, returns snippets with file:line\n\
         - `reindex_path(index_id, force_full?)` – re-sync an index after large changes\n\n\
         Available indexes:\n",
    );
    for i in indexes {
        s.push_str(&format!("- `{}` – {} (`{}`)\n", i.id, i.name, i.root_path));
    }
    s.push_str(
        "\nPrefer `search_index` with concrete identifiers (e.g. `UCharacterMovementComponent PhysWalking`) \
         or a short behavioural description; open the returned files for full context.\n",
    );
    s
}

#[tauri::command]
fn claude_snippets(state: State<'_, AppState>) -> CmdResult<ClaudeSnippets> {
    let exe = mcp_exe_path();
    let exe_s = exe.display().to_string();
    let cfg = state.engine.config().map_err(err)?;
    let json = serde_json::to_string_pretty(&serde_json::json!({
        "mcpServers": { MCP_SERVER_NAME: { "type": "stdio", "command": exe_s, "args": [] } }
    }))
    .map_err(err)?;
    Ok(ClaudeSnippets {
        command: format!(
            "claude mcp add --transport stdio --scope user {MCP_SERVER_NAME} -- \"{exe_s}\""
        ),
        exe_exists: exe.exists(),
        exe_path: exe_s,
        json,
        claude_md: claude_md(&cfg.indexes),
    })
}

/// Find `claude` on PATH (on Windows also `claude.exe` / `claude.cmd`, the
/// npm shim), so it is started directly rather than through a shell.
fn find_claude() -> Option<PathBuf> {
    let names: &[&str] = if cfg!(windows) { &["claude.exe", "claude.cmd", "claude.bat"] } else { &["claude"] };
    std::env::split_paths(&std::env::var_os("PATH")?)
        .filter(|d| d.is_absolute())
        .flat_map(|d| names.iter().map(move |n| d.join(n)))
        .find(|p| p.is_file())
}

fn claude_cmd(args: &[&str]) -> std::io::Result<std::process::Output> {
    let exe = find_claude()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "`claude` not found on PATH"))?;
    let mut c = std::process::Command::new(exe);
    c.args(args);
    #[cfg(windows)]
    {
        // Rust escapes arguments for .cmd/.bat targets safely (and refuses
        // ones it cannot escape), so no `cmd /C` string is built here.
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    c.output()
}

/// Paste-ready AGENTS.md / CLAUDE.md paragraph for one index.
#[tauri::command]
fn agents_snippet(state: State<'_, AppState>, index_id: String) -> CmdResult<String> {
    let cfg = state.engine.config().map_err(err)?;
    let ix = cfg.index(&index_id).ok_or_else(|| format!("unknown index '{index_id}'"))?;
    Ok(rag_core::agents_md::agents_md_snippet(ix, &cfg.indexes))
}

#[tauri::command]
async fn export_index(
    state: State<'_, AppState>,
    index_id: String,
    path: String,
    label: Option<String>,
) -> CmdResult<rag_core::share::ExportReport> {
    rag_core::share::export_index(&state.engine.paths, &index_id, std::path::Path::new(&path), label)
        .await
        .map_err(err)
}

#[tauri::command]
async fn inspect_export(path: String) -> CmdResult<rag_core::share::ExportMeta> {
    tauri::async_runtime::spawn_blocking(move || rag_core::share::inspect(std::path::Path::new(&path)))
        .await
        .map_err(err)?
        .map_err(err)
}

#[tauri::command]
async fn import_index(
    state: State<'_, AppState>,
    path: String,
    root_path: String,
    index_id: String,
    name: Option<String>,
) -> CmdResult<rag_core::share::ImportReport> {
    rag_core::share::import_index(&state.engine.paths, std::path::Path::new(&path), &root_path, &index_id, name)
        .await
        .map_err(err)
}

#[tauri::command]
async fn register_claude_mcp() -> CmdResult<String> {
    let exe = mcp_exe_path();
    if !exe.exists() {
        return Err(format!(
            "rag-mcp executable not found at {} (build it with `cargo build --release -p rag-mcp`)",
            exe.display()
        ));
    }
    let exe_s = exe.display().to_string();
    tauri::async_runtime::spawn_blocking(move || {
        // Replace an existing registration (ignore "not found").
        let _ = claude_cmd(&["mcp", "remove", "--scope", "user", MCP_SERVER_NAME]);
        let out = claude_cmd(&[
            "mcp", "add", "--transport", "stdio", "--scope", "user", MCP_SERVER_NAME, "--", &exe_s,
        ])
        .map_err(|e| format!("could not run `claude` (is Claude Code installed and on PATH?): {e}"))?;
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if out.status.success() {
            Ok(if text.trim().is_empty() { "Registered.".into() } else { text.trim().to_string() })
        } else {
            Err(format!("claude mcp add failed: {}", text.trim()))
        }
    })
    .await
    .map_err(err)?
}

/// GitHub `owner/repo` that publishes releases (release pages only).
const REPO: &str = "datorkar/Code-Rag-MCP";

#[derive(Serialize, Clone)]
struct UpdateInfo {
    current: String,
    latest: String,
    available: bool,
    notes: String,
    published_at: Option<String>,
    /// Release page ("What's new").
    url: String,
}

#[derive(Serialize, Clone)]
struct UpdateProgress {
    downloaded: u64,
    total: Option<u64>,
    done: bool,
}

/// Release builds get the updater's public key from CI; local/dev builds
/// have none, and updates are disabled there.
fn updater_configured(app: &AppHandle) -> bool {
    app.config()
        .plugins
        .0
        .get("updater")
        .and_then(|u| u.get("pubkey"))
        .and_then(|k| k.as_str())
        .is_some_and(|k| !k.trim().is_empty())
}

/// Look for a newer signed release (GitHub `latest.json`). `manual = false`
/// (startup) respects the "check for updates" setting and runs at most daily.
#[tauri::command]
async fn check_update(
    app: AppHandle,
    state: State<'_, AppState>,
    manual: bool,
) -> CmdResult<Option<UpdateInfo>> {
    use tauri_plugin_updater::UpdaterExt;
    if !updater_configured(&app) {
        return if manual { Err("updates are only available in release builds".into()) } else { Ok(None) };
    }
    if !manual {
        let c = state.engine.config().map_err(err)?;
        let recent = c
            .last_update_check
            .is_some_and(|t| chrono::Utc::now() - t < chrono::Duration::hours(24));
        if !c.check_updates || recent {
            return Ok(None);
        }
        // Record the attempt even if offline, so startup doesn't retry all day.
        let _ = state.engine.update_settings(|c| c.last_update_check = Some(chrono::Utc::now()));
    }
    let current = app.package_info().version.to_string();
    let found = app
        .updater_builder()
        .on_before_exit(move_locked_files_aside)
        .build()
        .map_err(err)?
        .check()
        .await
        .map_err(err)?;
    let info = match &found {
        Some(u) => UpdateInfo {
            current: current.clone(),
            latest: u.version.clone(),
            available: true,
            notes: u.body.clone().unwrap_or_default(),
            published_at: u.date.map(|d| d.to_string()),
            url: format!("https://github.com/{REPO}/releases/tag/v{}", u.version),
        },
        None => UpdateInfo {
            latest: current.clone(),
            url: format!("https://github.com/{REPO}/releases/latest"),
            current,
            available: false,
            notes: String::new(),
            published_at: None,
        },
    };
    *state.pending_update.lock().unwrap() = found;
    Ok(Some(info))
}

/// Download the update found by `check_update`, verify its signature,
/// run the installer (passive) and restart into the new version.
#[tauri::command]
async fn install_update(app: AppHandle, state: State<'_, AppState>) -> CmdResult<()> {
    if state.jobs.lock().unwrap().keys().any(|k| !is_winml_job(k)) {
        return Err("an index is being built; install the update when it is done".into());
    }
    // Cloned, not taken: a failed download can be retried from the banner.
    let update = state
        .pending_update
        .lock()
        .unwrap()
        .clone()
        .ok_or("no update available; check for updates first")?;
    let mut downloaded = 0u64;
    let emitter = app.clone();
    update
        .download_and_install(
            move |chunk, total| {
                downloaded += chunk as u64;
                let _ = emitter.emit("update-progress", UpdateProgress { downloaded, total, done: false });
            },
            || {},
        )
        .await
        .map_err(err)?;
    let _ = app.emit("update-progress", UpdateProgress { downloaded: 0, total: None, done: true });
    app.restart();
}

/// Files the installer replaces that a running `rag-mcp` (kept open by Claude
/// Code) holds locked. Windows allows renaming in-use files, so they are moved
/// aside right before the installer starts; the old process keeps running from
/// the renamed file and the next start uses the new one.
const LOCKED_FILES: &[&str] = &["rag-mcp.exe", "onnxruntime.dll", "DirectML.dll", "Microsoft.Windows.AI.MachineLearning.dll"];

fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe().ok()?.parent().map(|d| d.to_path_buf())
}

fn move_locked_files_aside() {
    let Some(dir) = exe_dir() else { return };
    for name in LOCKED_FILES {
        let src = dir.join(name);
        if !src.exists() {
            continue;
        }
        // An older `.old` may still be in use by an older rag-mcp: pick a free name.
        let dst = (0..10)
            .map(|i| dir.join(if i == 0 { format!("{name}.old") } else { format!("{name}.old{i}") }))
            .find(|p| !p.exists() || std::fs::remove_file(p).is_ok());
        if let Some(dst) = dst {
            if let Err(e) = std::fs::rename(&src, &dst) {
                tracing::warn!("could not move {} aside before update: {e}", src.display());
            }
        }
    }
}

/// After an update: delete files moved aside (once no process uses them). If
/// an install failed and a file is missing, put the old one back.
fn cleanup_moved_files() {
    let Some(dir) = exe_dir() else { return };
    for name in LOCKED_FILES {
        let original = dir.join(name);
        for i in 0..10 {
            let old = dir.join(if i == 0 { format!("{name}.old") } else { format!("{name}.old{i}") });
            if !old.exists() {
                continue;
            }
            if !original.exists() {
                let _ = std::fs::rename(&old, &original);
            } else {
                let _ = std::fs::remove_file(&old);
            }
        }
    }
}

/// Open a release page of this project in the browser.
#[tauri::command]
fn open_release(url: String) -> CmdResult<()> {
    let allowed = format!("https://github.com/{REPO}/releases/");
    if !url.starts_with(&allowed) || url.chars().any(|c| c.is_whitespace() || c == '"') {
        return Err("only this project's release pages can be opened".into());
    }
    open_url(&url)
}

#[cfg(windows)]
fn open_url(url: &str) -> CmdResult<()> {
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;
    let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (verb, target) = (wide("open"), wide(url));
    // Hands the https URL to the default browser (no command shell involved).
    let r = unsafe {
        ShellExecuteW(std::ptr::null_mut(), verb.as_ptr(), target.as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL)
    };
    if (r as isize) <= 32 {
        return Err(format!("could not open the browser (code {})", r as isize));
    }
    Ok(())
}

#[cfg(not(windows))]
fn open_url(url: &str) -> CmdResult<()> {
    let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    std::process::Command::new(opener).arg(url).spawn().map(|_| ()).map_err(err)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,lance=error,lancedb=warn,ort=warn".into()),
        )
        .try_init();

    if let Err(e) = rag_core::runtime::init() {
        tracing::error!("ONNX Runtime unavailable: {e:#}");
    }
    let engine = Engine::new().expect("failed to initialise data directory");
    cleanup_moved_files();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(AppState { engine, jobs: Mutex::new(HashMap::new()), pending_update: Mutex::new(None) })
        .invoke_handler(tauri::generate_handler![
            list_indexes,
            list_models,
            list_rerankers,
            get_settings,
            save_settings,
            save_index,
            delete_index,
            start_index,
            cancel_index,
            claude_snippets,
            agents_snippet,
            register_claude_mcp,
            gpu_status,
            export_index,
            inspect_export,
            import_index,
            winml_install,
            winml_cancel,
            check_update,
            install_update,
            hardware_summary,
            open_release,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
