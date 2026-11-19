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
    reranker_model: Option<String>,
    rerank_candidates: usize,
    auto_sync_max_files: usize,
    snippet_max_lines: usize,
    #[serde(default)]
    data_dir: String,
    #[serde(default)]
    gpu_features: Vec<String>,
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
    #[allow(unused_mut)]
    let mut gpu: Vec<String> = Vec::new();
    #[cfg(feature = "cuda")]
    gpu.push("cuda".to_string());
    #[cfg(feature = "directml")]
    gpu.push("directml".to_string());
    GlobalSettings {
        active_embed_model: c.active_embed_model.clone(),
        execution_provider: c.execution_provider,
        embed_batch_size: c.embed_batch_size,
        reranker_model: c.reranker_model.clone(),
        rerank_candidates: c.rerank_candidates,
        auto_sync_max_files: c.auto_sync_max_files,
        snippet_max_lines: c.snippet_max_lines,
        data_dir: engine.paths.root().display().to_string(),
        gpu_features: gpu,
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
            c.embed_batch_size = settings.embed_batch_size.clamp(1, 512);
            c.reranker_model = reranker.clone();
            c.rerank_candidates = settings.rerank_candidates.clamp(5, 200);
            c.auto_sync_max_files = settings.auto_sync_max_files;
            c.snippet_max_lines = settings.snippet_max_lines;
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
                    "+{} ~{} -{} files · {} chunks embedded · {} total · {:.1}s",
                    r.files_added, r.files_modified, r.files_removed, r.chunks_embedded,
                    r.total_chunks, r.duration_secs
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
        state.jobs.lock().unwrap().remove(&index_id);
        let _ = app.emit("index-finished", ev);
    });
    Ok(())
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

fn claude_cmd(args: &[&str]) -> std::io::Result<std::process::Output> {
    // `claude` is usually an npm .cmd shim on Windows, which CreateProcess cannot run directly.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let mut c = std::process::Command::new("cmd");
        c.arg("/C").arg("claude").args(args).creation_flags(CREATE_NO_WINDOW);
        c.output()
    }
    #[cfg(not(windows))]
    {
        std::process::Command::new("claude").args(args).output()
    }
}

/// Paste-ready AGENTS.md / CLAUDE.md paragraph for one index.
#[tauri::command]
fn agents_snippet(state: State<'_, AppState>, index_id: String) -> CmdResult<String> {
    let cfg = state.engine.config().map_err(err)?;
    let ix = cfg.index(&index_id).ok_or_else(|| format!("unknown index '{index_id}'"))?;
    Ok(rag_core::agents_md::agents_md_snippet(ix, &cfg.indexes))
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,lance=error,lancedb=warn,ort=warn".into()),
        )
        .try_init();

    let engine = Engine::new().expect("failed to initialise data directory");
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .manage(AppState { engine, jobs: Mutex::new(HashMap::new()) })
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
