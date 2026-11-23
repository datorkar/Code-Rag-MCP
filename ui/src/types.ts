export type ExecutionProvider = "auto" | "gpu" | "npu" | "cuda" | "directml" | "cpu";
export type Precision = "standard" | "int8";
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
  precision: Precision;
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
  note: string | null;
  needs_rebuild: boolean;
}

export interface ModelSpec {
  id: string;
  dim: number;
  max_context: number;
  max_chunk_chars: number;
  description: string;
  size_gb: number;
  recommended: boolean;
}

export interface GlobalSettings {
  active_embed_model: string;
  execution_provider: ExecutionProvider;
  embed_batch_size: number;
  embed_token_budget: number;
  force_fp32: boolean;
  resource_mode: "auto" | "max" | "low";
  reranker_model: string | null;
  rerank_candidates: number;
  auto_sync_max_files: number;
  snippet_max_lines: number;
  data_dir: string;
  gpu_features: string[];
  check_updates: boolean;
  app_version: string;
}

export interface UpdateInfo {
  current: string;
  latest: string;
  available: boolean;
  /** Release page ("What's new"). */
  url: string;
  notes: string;
  published_at: string | null;
}

export interface UpdateProgress {
  downloaded: number;
  total: number | null;
  done: boolean;
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
  eta_secs: number | null;
  files_chunked: number;
  files_to_chunk: number;
  index_step: number;
  index_steps: number;
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

export interface DeviceInfo {
  ep: string;
  ep_vendor: string;
  kind: string;
  vendor: string;
}

export interface RuntimeInfo {
  flavour: "winml" | "bundled";
  dll_dir: string | null;
  registered_providers: string[];
  devices: DeviceInfo[];
  vendor_gpu: boolean;
}

export interface ProviderStatus {
  name: string;
  version: string;
  ready_state: "ready" | "not_ready" | "not_present";
  certified: boolean;
  registered: boolean;
}

export interface GpuStatus {
  runtime: RuntimeInfo | null;
  runtime_error: string | null;
  providers: ProviderStatus[];
  providers_error: string | null;
  windows_ml: boolean;
}

export interface WinmlProgress {
  provider: string;
  progress: number;
  done: boolean;
  ok: boolean;
  message: string | null;
}

export interface ExportMeta {
  format: number;
  app_version: string;
  chunker_version: number;
  created_at: string;
  label: string | null;
  index: IndexConfig;
  chunk_count: number;
  file_count: number;
  dim: number;
}

export interface ExportReport {
  path: string;
  bytes: number;
  chunks: number;
  secs: number;
}

export interface ImportReport {
  index_id: string;
  chunks: number;
  files: number;
  files_rejected: number;
  warnings: string[];
  secs: number;
}
