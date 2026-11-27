# Local Code RAG

Fast, private semantic code search for **Claude Code**. Point it at your folders (an Unreal project, the engine source, any codebase), and Claude can search them by meaning and by name instead of grepping through millions of lines. Everything runs on your machine: code never leaves your PC, and the AI model that understands the code runs on your own GPU or CPU.

> Building it yourself or contributing? See [DEVELOPERS.md](DEVELOPERS.md).

## What you get

- **A desktop app** to add folders, build indexes and watch progress.
- **A search tool for Claude Code** (an MCP server). Claude calls it with questions like *"where is the walking movement calculated"* or *"UCharacterMovementComponent PhysWalking"* and gets the most relevant code with file paths, line numbers and the enclosing class/function.
- **Indexes that stay current.** Only changed files are re-processed, and small changes are picked up automatically when Claude searches.

## Requirements

- Windows 11, 64-bit. GPU acceleration uses Windows ML (any DirectX 12 GPU); on 24H2 or later the app can also download vendor-specific GPU/NPU providers.
- About 1 GB free space for the app and a model, plus index space (roughly 1–2 GB for a full Unreal Engine source index).
- [Claude Code](https://claude.com/claude-code) to use the search from Claude.

## Install

1. Download `LocalCodeRAG_<version>_x64-setup.exe` from the [latest release](https://github.com/datorkar/Code-Rag-MCP/releases/latest) and run it.
   Windows SmartScreen may warn because the installer is not signed by a certificate authority: choose *More info → Run anyway*.
2. Start **Local Code RAG**.
3. Click **Install to Claude Code**. This registers the search tool with Claude Code, or shows the command / JSON to add it yourself.

The app updates itself: when a new version is out you'll see a banner with **Install & restart**. Turn the daily check off, or check manually, under **Settings → Updates**.

## Quick start

1. **+ Add index** → pick the root folder and give it a name. The ID (e.g. `my-game`) is what Claude uses to refer to it.
2. Pick file types and folders to ignore. Presets for Unreal projects, engine headers, Rust/C++, web and Python are included. `.gitignore` is respected automatically.
3. **Create & index.** The first build of a large codebase takes a while (the progress drawer shows files, chunks and time left); later updates only process changed files.
4. In Claude Code, ask about your code. Claude calls `list_indexes` and `search_index` by itself. To make it reliable, click **AGENTS.md** on the index card and paste the paragraph into your project's `AGENTS.md` / `CLAUDE.md`.

## GPU acceleration

Click **GPU** in the header to see which devices are available and to download Windows ML providers for your hardware (NVIDIA TensorRT-RTX, AMD, Intel OpenVINO, Qualcomm). Without them the app uses DirectML or the CPU.

**Settings → Execution provider:** *Auto* (best available GPU) · *GPU* · *NPU* (experimental) · *DirectML* · *CPU*.

A GPU is typically 10–30× faster than the CPU for building indexes. Searching is fast either way.

## Choosing a model

Each index has its own embedding model (the AI that turns code into searchable vectors). The four recommended ones:

| Model | Best for |
|---|---|
| **jina-embeddings-v2-base-code** (default) | Trained on code. The best all-round choice. |
| **bge-small-en-v1.5** | About 5× faster and small. Huge codebases or laptops; slightly less precise. |
| **gte-base-en-v1.5** | Strong general retrieval, good for mixed code + docs. |
| **modernbert-embed-large** | Highest quality, slowest. |

All other supported models are listed below these. Models download automatically on first use. Changing an index's model requires one full rebuild (the app offers it).

**Precision** (per index): *Standard* (full quality, uses FP16 on GPU automatically) or *INT8* (about 1.4× faster on CPU; about 1 in 10 top results differ).

## Settings worth knowing

- **Resource use while indexing:**
  - *Auto*: low priority; laptops and integrated GPUs use *Low*.
  - *Low*: keeps the PC responsive by leaving a CPU core free and using small batches.
  - *Max*: full speed.
- **Reranker:** a second model re-scores the top results for better ranking. It adds about 100–300 ms per search on GPU and is on by default.
- **Auto-sync limit:** when Claude searches, up to this many changed files are re-indexed on the spot. Larger changes are left for **Index now** in the app.
- **Excerpt length:** how many lines of each result Claude gets.

Per index (**Settings** on the card): the description shown to Claude, chunk size, precision, and auto-update on search with its cooldown.

## Sharing an index

Indexing a whole engine takes time, so a team can share it:

- **Export** on an index card saves it as one compressed `.lcrag` file.
- **Import** (header) loads someone's export against *your* copy of the source. Every piece of code is checked against your local files: anything missing or different is dropped and re-indexed, so an export can never inject code you don't have.

Exports contain source code. Only share them with people allowed to have that code (for Unreal Engine: other licensees, e.g. inside your studio).

## Tips

- **Unreal Engine source:** index `Engine/Source` with the *Unreal Engine Headers* or *Unreal Project* preset, and ignore the platform and RHI folders you don't use (presets included).
- **Point Claude at the right place:** `search_index` accepts a path filter (e.g. `Runtime/Engine/**`) and file types (e.g. only `.h` for declarations). The AGENTS.md paragraph tells Claude about this.
- **"No matching files found":** the app refuses to wipe an index when its folder suddenly looks empty (unplugged drive, unsynced workspace). Fix the folder, or use **Full rebuild** if the files were really removed.
- **Slow first build:** check **GPU** and the chunks/sec in the progress drawer. If you see CPU speeds on a GPU machine, download the provider for your GPU.

## Privacy & security

- Code, indexes and searches stay on your PC. The app only goes online to download AI models (Hugging Face), Windows ML GPU providers (Microsoft) and to check for updates (GitHub).
- It uses your system's certificate store and proxy settings, so it works on corporate networks.
- Updates are signed; the app verifies the signature before installing.
- Data location: `%APPDATA%\local-code-rag` (override with the `LOCAL_CODE_RAG_HOME` environment variable).

## Uninstall

Uninstall *Local Code RAG* from Windows Settings → Apps. Remove the Claude Code registration with `claude mcp remove --scope user local-code-rag`, and delete `%APPDATA%\local-code-rag` to remove indexes and models.
