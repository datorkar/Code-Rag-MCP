import { invoke } from "@tauri-apps/api/core";
import type {
  ClaudeSnippets,
  ExportMeta,
  ExportReport,
  ImportReport,
  GpuStatus,
  GlobalSettings,
  HardwareSummary,
  UpdateInfo,
  IndexConfig,
  IndexInfo,
  ModelSpec,
  SaveResult,
} from "./types";

export const api = {
  listIndexes: () => invoke<IndexInfo[]>("list_indexes"),
  listModels: () => invoke<ModelSpec[]>("list_models"),
  listRerankers: () => invoke<string[]>("list_rerankers"),
  getSettings: () => invoke<GlobalSettings>("get_settings"),
  saveSettings: (settings: GlobalSettings) => invoke<GlobalSettings>("save_settings", { settings }),
  saveIndex: (index: IndexConfig, originalId: string | null) =>
    invoke<SaveResult>("save_index", { index, originalId }),
  deleteIndex: (indexId: string) => invoke<void>("delete_index", { indexId }),
  startIndex: (indexId: string, forceFull: boolean) =>
    invoke<void>("start_index", { indexId, forceFull }),
  cancelIndex: (indexId: string) => invoke<boolean>("cancel_index", { indexId }),
  agentsSnippet: (indexId: string) => invoke<string>("agents_snippet", { indexId }),
  gpuStatus: () => invoke<GpuStatus>("gpu_status"),
  hardwareSummary: () => invoke<HardwareSummary>("hardware_summary"),
  winmlInstall: (provider: string | null) => invoke<void>("winml_install", { provider }),
  winmlCancel: () => invoke<boolean>("winml_cancel"),
  exportIndex: (indexId: string, path: string, label: string | null) =>
    invoke<ExportReport>("export_index", { indexId, path, label }),
  inspectExport: (path: string) => invoke<ExportMeta>("inspect_export", { path }),
  importIndex: (path: string, rootPath: string, indexId: string, name: string | null) =>
    invoke<ImportReport>("import_index", { path, rootPath, indexId, name }),
  claudeSnippets: () => invoke<ClaudeSnippets>("claude_snippets"),
  registerClaudeMcp: () => invoke<string>("register_claude_mcp"),
  /** `manual = false`: respects the setting and checks at most daily (may return null). */
  checkUpdate: (manual: boolean) => invoke<UpdateInfo | null>("check_update", { manual }),
  /** Downloads, verifies and installs the update found by checkUpdate, then restarts the app. */
  installUpdate: () => invoke<void>("install_update"),
  openRelease: (url: string) => invoke<void>("open_release", { url }),
};
