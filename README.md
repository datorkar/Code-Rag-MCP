# Local Code RAG

Local semantic code search for Claude Code: index folders (Unreal projects, engine source, anything), embed them on-device with FastEmbed/ONNX, store them in LanceDB, and expose hybrid (vector + BM25) search over MCP. A Tauri v2 desktop app manages the indexes.

```
crates/rag-core   indexing, chunking, embeddings, LanceDB storage, config, file locks
crates/rag-mcp    headless MCP server (stdio, rmcp)
src-tauri + ui/   Tauri v2 desktop manager (React + TS + Tailwind v4)
```

## Build

Requirements: Rust ≥ 1.94, Node 22, `protoc` (LanceDB dependency).
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

**CUDA:** the user's machine needs the **CUDA 13** runtime + **cuDNN 9 (for CUDA 13)** on `PATH` (not bundled, ~1 GB) and a recent NVIDIA driver. The pinned ONNX Runtime (`ort 2.0.0-rc.15`) only ships CUDA 13 prebuilt binaries; with CUDA 12 installed it silently runs on CPU. The GUI's chunks/sec readout shows whether the GPU is actually used (CPU is ~10–30x slower).

ONNX Runtime binaries are downloaded at build time by `ort`; models are downloaded from Hugging Face on first use into `<data dir>/models`.

## Windows ML (branch `claude/windows-ml`)

On Windows this branch uses **Windows ML** instead of a bundled ONNX Runtime + CUDA:

- `onnxruntime.dll` (ORT 1.28), `Microsoft.Windows.AI.MachineLearning.dll` and `DirectML.dll` from the NuGet package `Microsoft.Windows.AI.MachineLearning` 2.5.96 ship next to the exe. `scripts/fetch-winml.mjs` downloads and checksum-verifies them; `tauri build` and the Tauri `build.rs` call it automatically. No Windows App SDK install and no CUDA is needed. Requires Windows 11 24H2+ for the vendor provider catalog; DirectML works on any Windows 11.
- At startup the app and `rag-mcp` load that ORT (`ort` `load-dynamic`) and register every certified execution provider Windows ML has already installed (NVIDIA TensorRT-RTX, AMD, Intel OpenVINO, Qualcomm QNN, … depending on the device).
- **GPU** button → lists the devices ORT sees and the Windows ML providers for this machine, with **Download & enable** (progress + cancel). Windows downloads and updates the providers; new ones are used immediately by the app, and by `rag-mcp` after Claude Code restarts it.
- Execution provider setting: **Auto/GPU** → a registered vendor GPU provider via ORT's `PREFER_GPU` policy, else DirectML · **NPU** → `PREFER_NPU` (experimental for embeddings) · **DirectML** · **CPU**.
- Linux/macOS builds are unchanged (bundled ORT, CUDA with CPU fallback).

Dev/test on Windows:

```powershell
node scripts/fetch-winml.mjs
$env:LOCAL_CODE_RAG_WINML_DIR = "$PWD\src-tauri\binaries\winml"
cargo test -p rag-core --features winml -p rag-mcp
cargo test -p rag-core --features winml --test winml_smoke -- --ignored --nocapture   # real embedding, lists catalog
```

rag-core needs an ORT flavour feature: `bundled` (or `cuda`) or `winml`; `rag-mcp` and the app pick it per platform.

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

**Speed features:** each chunk stores a blake3 hash of its embedding input; when a file changes, chunks whose text is unchanged reuse their stored vector and duplicates within a sync are embedded once (results are identical to a full re-embed). Batches are sized by a token budget (32k tokens on GPU, 8k on CPU, configurable) and split automatically on out-of-memory. Database writes run in a separate task while the GPU embeds. Small syncs update the full-text index incrementally; a full index rebuild + compaction only happens when ≥10% of the rows changed. **Half precision:** Settings → "Half precision (FP16) on GPU" (or `LOCAL_CODE_RAG_FP16=1|0`) loads each model's official `onnx/model_fp16.onnx` (embedder and reranker); embeddings match FP32 (cosine ≈ 1.0000, identical top-5 neighbours in our check). Ignored on CPU, where FP16 is slower; falls back to FP32 if a model has no FP16 file. The finished message shows a per-phase time breakdown (scan / chunk / model / embed / write / index).

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

### Per-project AGENTS.md

Each index card has an **AGENTS.md** button that generates a paste-ready paragraph for that project's `AGENTS.md` / `CLAUDE.md`: its index id, root path and file types, when to call `search_index` vs. `reindex_path`, and the other indexes (e.g. engine source) the agent can also search. Generated by `rag_core::agents_md::agents_md_snippet`.

## Build speed

- **Shared dependency features:** `crates/workspace-hack` (generated by [`cargo-hakari`](https://docs.rs/cargo-hakari)) makes every crate in the workspace build LanceDB and the rest of the dependency tree with the same features, so switching between `rag-mcp`, the app and the tests does not recompile them. ONNX Runtime crates are excluded on purpose (`.config/hakari.toml`). After adding or changing dependencies run `cargo hakari generate && cargo hakari manage-deps`; CI fails if it is stale.
- **CI cache:** Linux uses `sccache` (compiled crates in the GitHub Actions cache, reused across runs and branches); Windows caches the whole `target/` with `rust-cache` (sccache can't handle Cargo's long rustc command lines on Windows). Caches are saved even when a job fails.
- **Locally on Windows:** keep the repo on a Dev Drive or exclude `target/` from Defender real-time scanning; optionally `cargo install sccache` and set `RUSTC_WRAPPER=sccache` to share compiled crates between branches.

## Tests

```
cargo test -p rag-core --features bundled     # unit tests (walker, manifest diff, chunker, model resolution)
cargo test -p rag-core --features bundled -- --ignored   # end-to-end: downloads bge-small, indexes, searches, syncs
```
