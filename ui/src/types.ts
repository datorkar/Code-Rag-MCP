export type ExecutionProvider = "auto" | "cuda" | "directml" | "cpu";
export type IndexStatus = "ready" | "indexing" | "stale" | "never_indexed";

export interface IndexConfig {
  id: string;
  name: string;
  description?: string | null;
  root_path: string;
  embed_model: string;
  allowed_extensions: string[];
  ignore_dirs: string[];
  chunk_size: number;
  chunk_overlap: number;
  max_file_bytes: number;
  auto_update_on_search: boolean;
  auto_update_cooldown_hours: number;
  last_indexed_at: string | null;
  chunk_count: number;
}

export interface IndexInfo extends IndexConfig {
  status: IndexStatus;
  needs_rebuild: boolean;
}

export interface SaveResult {
  index: IndexConfig;
  created: boolean;
  needs_rebuild: boolean;
}

export interface ModelSpec {
  id: string;
  dim: number;
  max_context: number;
  description: string;
  recommended: boolean;
}

export interface GlobalSettings {
  active_embed_model: string;
  execution_provider: ExecutionProvider;
  embed_batch_size: number;
  reranker_model: string | null;
  rerank_candidates: number;
  auto_sync_max_files: number;
  snippet_max_lines: number;
  data_dir: string;
  gpu_features: string[];
}

export type Phase =
  | "waiting"
  | "scanning"
  | "hashing"
  | "deleting"
  | "chunking"
  | "loading_model"
  | "embedding"
  | "building_index"
  | "done"
  | "cancelled"
  | "error";

export interface Progress {
  index_id: string;
  phase: Phase;
  files_scanned: number;
  files_total: number;
  files_changed: number;
  chunks_embedded: number;
  chunks_total: number;
  chunks_per_sec: number;
  elapsed_secs: number;
  message: string | null;
}

export interface FinishedEvent {
  index_id: string;
  ok: boolean;
  cancelled: boolean;
  message: string;
}

export interface ClaudeSnippets {
  exe_path: string;
  exe_exists: boolean;
  command: string;
  json: string;
  claude_md: string;
}
