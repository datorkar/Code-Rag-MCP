export const EXTENSION_PRESETS: Record<string, string[]> = {
  "Unreal Project": [".cpp", ".h", ".cs", ".usf", ".ush", ".ini"],
  "Unreal Engine Headers": [".h", ".hpp", ".inl"],
  "Rust / C++": [".rs", ".cpp", ".cc", ".h", ".hpp", ".toml"],
  "Web / TS": [".ts", ".tsx", ".js", ".jsx", ".css", ".json"],
  Python: [".py", ".pyi"],
};

export const IGNORE_PRESETS: Record<string, string[]> = {
  "UE5 Build Output": ["Binaries", "Build", "DerivedDataCache", "Intermediate", "Saved"],
  "UE5 Content / 3rd party": ["Content", "ThirdParty"],
  "UE5 Platforms": ["Android", "IOS", "Mac", "Unix", "Linux", "Apple", "TVOS", "VisionOS"],
  "UE5 RHI Backends": ["VulkanRHI", "OpenGLDrv", "D3D11RHI", "D3D12RHI", "MetalRHI"],
  "JS / Rust": ["node_modules", "dist", "target", ".next", "build"],
};

export const DEFAULT_INDEX = {
  id: "",
  name: "",
  description: null,
  root_path: "",
  embed_model: "jinaai/jina-embeddings-v2-base-code",
  allowed_extensions: EXTENSION_PRESETS["Unreal Project"],
  ignore_dirs: [...IGNORE_PRESETS["UE5 Build Output"], ...IGNORE_PRESETS["UE5 Content / 3rd party"]],
  chunk_size: 2000,
  chunk_overlap: 200,
  max_file_bytes: 2 * 1024 * 1024,
  precision: "auto" as const,
  auto_update_on_search: true,
  auto_update_cooldown_hours: 24,
  last_indexed_at: null,
  chunk_count: 0,
};

export function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function timeAgo(iso: string | null): string {
  if (!iso) return "never";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}
