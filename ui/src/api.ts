import { invoke } from "@tauri-apps/api/core";
import type {
  ClaudeSnippets,
  GlobalSettings,
  IndexConfig,
  IndexInfo,
  ModelSpec,
  SaveResult,
} from "./types";

export const api = {
  listIndexes: () => invoke<IndexInfo[]>("list_indexes"),
  listModels: () => invoke<ModelSpec[]>("list_models"),
  getSettings: () => invoke<GlobalSettings>("get_settings"),
  saveSettings: (settings: GlobalSettings) => invoke<GlobalSettings>("save_settings", { settings }),
  saveIndex: (index: IndexConfig, originalId: string | null) =>
    invoke<SaveResult>("save_index", { index, originalId }),
  deleteIndex: (indexId: string) => invoke<void>("delete_index", { indexId }),
  startIndex: (indexId: string, forceFull: boolean) =>
    invoke<void>("start_index", { indexId, forceFull }),
  cancelIndex: (indexId: string) => invoke<boolean>("cancel_index", { indexId }),
  claudeSnippets: () => invoke<ClaudeSnippets>("claude_snippets"),
  registerClaudeMcp: () => invoke<string>("register_claude_mcp"),
};
