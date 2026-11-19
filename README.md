# Local Code RAG

Local semantic code search for Claude Code: index folders (Unreal projects, engine source, anything), embed them on-device with FastEmbed/ONNX, store them in LanceDB, and expose hybrid (vector + BM25) search over MCP. A Tauri v2 desktop app manages the indexes.

```
crates/rag-core   indexing, chunking, embeddings, LanceDB storage, config, file locks
crates/rag-mcp    headless MCP server (stdio, rmcp)
src-tauri + ui/   Tauri v2 desktop manager (React + TS + Tailwind v4)
```

## Build

Requirements: Rust ≥ 1.90, Node 22, `protoc` (LanceDB dependency).
Windows: MSVC build tools + WebView2 (preinstalled on Win10/11).

```powershell
# MCP server (CUDA by default; falls back to CPU at runtime if CUDA/cuDNN are missing)
cargo build --release -p rag-mcp
cargo build --release -p rag-mcp --no-default-features                       # CPU only
cargo build --release -p rag-mcp --no-default-features --features directml   # AMD/Intel DX12 GPUs

# Desktop app: bundles rag-mcp + the ONNX Runtime CUDA provider DLLs next to the exe
npm ci
npx tauri dev                       # development (run `cargo build -p rag-mcp` once first)
npx tauri build                     # installer in target/release/bundle
$env:RAG_FEATURES="none"; npx tauri build      # CPU-only bundle
```

**CUDA:** the user's machine needs the NVIDIA CUDA runtime + cuDNN 9 on `PATH` (not bundled, ~1 GB). The ONNX Runtime CUDA 12 vs 13 build is chosen at compile time from `CUDA_HOME` (`...\CUDA\v12.x` → 12); force it with `ORT_CUDA_VERSION=12` or `13`. The GUI's chunks/sec readout shows whether the GPU is actually used (CPU is ~10–30x slower).

ONNX Runtime binaries are downloaded at build time by `ort`; models are downloaded from Hugging Face on first use into `<data dir>/models`.

## Data layout

`%APPDATA%\local-code-rag\` (Windows) · `~/.local/share/local-code-rag/` (Linux) · override with `LOCAL_CODE_RAG_HOME`.

```
config.json              global settings + per-index config (schema below)
manifests/<id>.json      relative_path -> {blake3, size, mtime}
locks/<id>.lock          fs2 advisory lock (GUI indexing vs MCP search/auto-sync)
lancedb/idx_<id>.lance   chunks: path, start_line, end_line, text, vector
models/                  FastEmbed ONNX cache
```

`config.json` follows the blueprint schema (`active_embed_model`, `execution_provider` = `auto|cuda|directml|cpu`, `indexes[]` with `root_path`, `embed_model`, `allowed_extensions`, `ignore_dirs`, `chunk_size`, `chunk_overlap`, `auto_update_on_search`, `auto_update_cooldown_hours`, `last_indexed_at`, `chunk_count`), plus global `embed_batch_size`, `reranker_model`, `rerank_candidates`, `auto_sync_max_files`, `snippet_max_lines` and `description` / `max_file_bytes` (per index).

## How indexing works

1. Walk `root_path` with `ignore` (respects `.gitignore`; skips `ignore_dirs` by name at any depth, case-insensitive; filters by extension; skips files > `max_file_bytes`).
2. Hash with blake3 — files whose size + mtime match the manifest are not re-read, so a no-change sync of a large tree is just a directory walk.
3. Diff against the manifest → delete chunks of removed/modified files.
4. Chunk new/modified files: tree-sitter AST splitting for C/C++/HLSL (`.usf/.ush`), Rust, C#, Python; plain-text splitter otherwise. Each chunk keeps its 1-based line range.
5. Sort chunks by length (minimises ONNX padding), embed in batches (`embed_batch_size`, default 32), flush to LanceDB every 2048 rows. The embedded text is prefixed with the file path for context.
6. Rebuild the full-text index on `text` (no stemming / stop words — code-friendly), a scalar index on `path`, and an ANN vector index once a table exceeds 100k rows; then compact.

Changing an index's model or chunk settings changes its fingerprint → next sync is a full rebuild (the GUI warns and offers to do it now). Cancelling keeps all completed files; the next sync resumes.

Search embeds the query (with the model's query prefix for BGE/Nomic), runs LanceDB hybrid search (vector + BM25, fused with Reciprocal Rank Fusion) over the top `rerank_candidates` (30), re-scores them with a cross-encoder (`reranker_model`, default `jinaai/jina-reranker-v2-base-multilingual`, `null` disables), and returns the top `limit`. Each result is trimmed to the `snippet_max_lines` (40) lines that best match the query terms (identifiers are split on `::` and camelCase), labelled with the original chunk range.

**Auto-sync on search** only runs inline when at most `auto_sync_max_files` (200) files changed and no full build is needed. Otherwise the search answers from the existing data with a note asking for "Index now" in the app (or `reindex_path`), so a tool call never blocks on a multi-minute build.

> Note: current LanceDB versions ship their own native BM25 inverted index instead of the older Tantivy-based one; the behaviour is the same (hybrid + RRF), with no extra dependency.

## MCP tools

| Tool | Purpose |
|---|---|
| `list_indexes()` | ids, names, roots, models, chunk counts, last update, status |
| `search_index(index_id, query, limit=6)` | hybrid search; incremental auto-sync first if `auto_update_on_search` and past cooldown |
| `reindex_path(index_id, force_full=false)` | manual sync |

Logging goes to stderr only (`RUST_LOG` to adjust). If the GUI is indexing, auto-sync is skipped and search proceeds against the last committed data (LanceDB reads are MVCC-safe).

Register (or use the **Install to Claude Code** button, which also generates a CLAUDE.md snippet):

```powershell
claude mcp add --transport stdio --scope user local-code-rag -- "C:\Program Files\Local Code RAG\rag-mcp.exe"
claude mcp list
```

## Tests

```
cargo test -p rag-core                        # unit tests (walker, manifest diff, chunker, model resolution)
cargo test -p rag-core -- --ignored           # end-to-end: downloads bge-small, indexes, searches, syncs
```
