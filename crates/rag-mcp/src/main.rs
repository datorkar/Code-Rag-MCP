//! `rag-mcp`: MCP server over stdio exposing the user's local code indexes.
//!
//! stdout carries JSON-RPC only — all logging goes to stderr.

use std::fmt::Write as _;

use rag_core::{Engine, IndexStatus, SyncOptions};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerConfig},
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
    ServerHandler, ServiceExt,
};
use serde::Deserialize;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchArgs {
    /// Index id from `list_indexes` (e.g. "ue5-runtime").
    index_id: String,
    /// Natural-language question or code identifiers to look for.
    query: String,
    /// Maximum number of snippets to return (default 6, max 50).
    #[serde(default)]
    limit: Option<usize>,
    /// Optional path restriction, relative to the index root. Glob
    /// (`Runtime/Engine/**`, `**/*Movement*`) or a plain folder prefix (`Source/Runtime`).
    #[serde(default)]
    path_filter: Option<String>,
    /// Optional file extensions to restrict to, e.g. [".h"] for declarations only.
    #[serde(default)]
    extensions: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ReindexArgs {
    /// Index id from `list_indexes`.
    index_id: String,
    /// Drop all stored chunks and rebuild from scratch (default false = incremental).
    #[serde(default)]
    force_full: bool,
}

#[derive(Clone)]
struct RagServer {
    engine: Engine,
    tool_router: ToolRouter<Self>,
}

fn lang_hint(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("").to_ascii_lowercase().as_str() {
        "cpp" | "cc" | "cxx" | "h" | "hpp" | "hh" | "inl" => "cpp",
        "usf" | "ush" | "hlsl" => "hlsl",
        "rs" => "rust",
        "cs" => "csharp",
        "py" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "ini" => "ini",
        "json" => "json",
        _ => "",
    }
}

#[tool_router]
impl RagServer {
    fn new(engine: Engine) -> Self {
        Self { engine, tool_router: Self::tool_router() }
    }

    #[tool(
        description = "List all locally indexed codebases (id, name, root path, embedding model, chunk count, last update, status). Call this first to discover valid index_id values for search_index."
    )]
    async fn list_indexes(&self) -> String {
        let infos = match self.engine.list() {
            Ok(i) => i,
            Err(e) => return format!("Error: {e:#}"),
        };
        if infos.is_empty() {
            return "No indexes configured. Ask the user to add one in the Local Code RAG app."
                .into();
        }
        let mut out = String::new();
        for i in infos {
            let c = &i.config;
            let status = match i.status {
                IndexStatus::Ready => "ready",
                IndexStatus::Indexing => "indexing",
                IndexStatus::Stale => "stale",
                IndexStatus::NeverIndexed => "never indexed",
            };
            let _ = writeln!(out, "- id: {}\n  name: {}", c.id, c.name);
            if let Some(d) = &c.description {
                let _ = writeln!(out, "  description: {d}");
            }
            let _ = writeln!(
                out,
                "  root: {}\n  extensions: {}\n  model: {}\n  chunks: {}\n  last_indexed: {}\n  status: {status}{}",
                c.root_path,
                c.allowed_extensions.join(" "),
                c.embed_model,
                c.chunk_count,
                c.last_indexed_at.map(|t| t.to_rfc3339()).unwrap_or_else(|| "never".into()),
                if i.needs_rebuild { " (settings changed, full rebuild pending)" } else { "" },
            );
        }
        out
    }

    #[tool(
        description = "Semantic + keyword (hybrid BM25/vector, cross-encoder reranked) search over a local code index. Returns the most relevant code excerpts with file paths and line ranges; open the file for surrounding context. Use precise identifiers (class/function names) and/or a natural-language description of the behaviour you are looking for. Narrow with path_filter (e.g. \"Runtime/Engine/**\") or extensions (e.g. [\".h\"]) when you know where to look. Each result shows its enclosing class/function. Small on-disk changes are synced automatically before searching; large ones are reported in a note."
    )]
    async fn search_index(&self, Parameters(args): Parameters<SearchArgs>) -> String {
        let limit = args.limit.unwrap_or(6);
        let filter = rag_core::SearchFilter {
            path_glob: args.path_filter.clone(),
            extensions: args.extensions.clone().unwrap_or_default(),
        };
        let outcome = match self.engine.search_filtered(&args.index_id, &args.query, limit, true, &filter).await {
            Ok(o) => o,
            Err(e) => return format!("Error: {e:#}"),
        };
        let root = self
            .engine
            .config()
            .ok()
            .and_then(|c| c.index(&args.index_id).map(|i| i.root_path.clone()))
            .unwrap_or_default();

        let mut out = String::new();
        if let Some(r) = &outcome.auto_sync {
            if r.files_added + r.files_modified + r.files_removed > 0 {
                let _ = writeln!(
                    out,
                    "(auto-updated index: +{} ~{} -{} files)\n",
                    r.files_added, r.files_modified, r.files_removed
                );
            }
        }
        for n in &outcome.notes {
            let _ = writeln!(out, "(note: {n})");
        }
        if outcome.hits.is_empty() {
            let scope = if args.path_filter.is_some() || args.extensions.as_ref().is_some_and(|e| !e.is_empty()) {
                " with the given path_filter/extensions (try without them)"
            } else {
                ""
            };
            let _ = writeln!(out, "No results for \"{}\" in index '{}'{scope}.", args.query, args.index_id);
            return out;
        }
        let _ = writeln!(out, "Index root: {root}\n");
        for (n, h) in outcome.hits.iter().enumerate() {
            let trimmed = h
                .chunk_lines
                .map(|(a, b)| format!(", excerpt of chunk {a}-{b}"))
                .unwrap_or_default();
            let scope = if h.context.is_empty() { String::new() } else { format!("In: `{}`\n", h.context) };
            let _ = writeln!(
                out,
                "### {}. {}:{}-{}  (score {:.3}{trimmed})\n{scope}```{}\n{}\n```\n",
                n + 1,
                h.path,
                h.start_line,
                h.end_line,
                h.score,
                lang_hint(&h.path),
                h.text.trim_end()
            );
        }
        out
    }

    #[tool(
        description = "Re-synchronise a local code index with the files on disk (incremental by default: only new/modified/deleted files are processed). Set force_full=true to rebuild from scratch. Large updates can take several minutes; prefer asking the user to use the Local Code RAG app for initial builds."
    )]
    async fn reindex_path(&self, Parameters(args): Parameters<ReindexArgs>) -> String {
        let opts = SyncOptions {
            force_full: args.force_full,
            lock_timeout: Some(std::time::Duration::from_secs(10)),
            ..Default::default()
        };
        match self.engine.sync(&args.index_id, opts).await {
            Ok(r) => format!(
                "Index '{}' synced in {:.1}s{}: scanned {} files (+{} added, ~{} modified, -{} removed), embedded {} chunks ({} reused), {} chunks total.",
                r.index_id,
                r.duration_secs,
                if r.full_rebuild { " (full rebuild)" } else { "" },
                r.files_scanned,
                r.files_added,
                r.files_modified,
                r.files_removed,
                r.chunks_embedded,
                r.chunks_reused,
                r.total_chunks
            ),
            Err(e) => format!("Error: {e:#}"),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for RagServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("local-code-rag", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Local semantic code search. Call list_indexes to discover index ids, then \
                 search_index(index_id, query) to retrieve relevant code snippets with file paths \
                 and line numbers. Prefer this over grepping large codebases such as engine source.",
            )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,lance=error,lancedb=warn,ort=warn".into()),
        )
        .init();

    // Load ONNX Runtime (Windows ML on Windows) and register installed GPU/NPU
    // providers. A failure is reported by the tools instead of killing the server.
    if let Err(e) = rag_core::runtime::init() {
        tracing::error!("ONNX Runtime unavailable: {e:#}");
    }
    let engine = Engine::new()?;
    // Free model RAM/VRAM when Claude has not searched for a while; the next
    // search reloads it (a few seconds).
    tokio::spawn(async {
        let idle = std::time::Duration::from_secs(15 * 60);
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            if rag_core::embedder::unload_if_idle(idle) {
                tracing::info!("unloaded idle models");
            }
        }
    });
    tracing::info!("rag-mcp starting; data dir {}", engine.paths.root().display());
    let service = RagServer::new(engine).serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
