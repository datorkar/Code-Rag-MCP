import type { IndexInfo, Progress } from "../types";
import { timeAgo } from "../presets";

const BADGE: Record<string, string> = {
  ready: "bg-emerald-500/15 text-emerald-300 border-emerald-600/40",
  indexing: "bg-indigo-500/15 text-indigo-300 border-indigo-500/40",
  stale: "bg-amber-500/15 text-amber-300 border-amber-600/40",
  never_indexed: "bg-zinc-700/40 text-zinc-300 border-zinc-600",
};
const LABEL: Record<string, string> = {
  ready: "Ready",
  indexing: "Indexing",
  stale: "Stale",
  never_indexed: "Not indexed",
};

interface Props {
  ix: IndexInfo;
  progress?: Progress;
  onIndex: (force: boolean) => void;
  onEdit: () => void;
  onDelete: () => void;
  onAgentsMd: () => void;
}

export function IndexCard({ ix, progress, onIndex, onEdit, onDelete, onAgentsMd }: Props) {
  const status = progress ? "indexing" : ix.status;
  const busy = status === "indexing";
  const pct =
    progress && progress.chunks_total > 0
      ? (progress.chunks_embedded / progress.chunks_total) * 100
      : progress && progress.files_total > 0
        ? (progress.files_scanned / progress.files_total) * 100
        : 0;

  return (
    <div className="group relative overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/60 p-4 hover:border-zinc-700">
      {busy && (
        <div className="absolute inset-x-0 top-0 h-0.5 bg-zinc-800">
          <div className="h-full bg-indigo-500 transition-[width]" style={{ width: `${pct}%` }} />
        </div>
      )}
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="truncate text-[15px] font-semibold text-zinc-100">{ix.name}</h3>
            <span className={`rounded border px-1.5 py-px text-[11px] font-medium ${BADGE[status]}`}>
              {LABEL[status]}
            </span>
            {ix.needs_rebuild && !busy && (
              <span className="rounded border border-red-600/40 bg-red-500/10 px-1.5 py-px text-[11px] text-red-300">
                Rebuild required
              </span>
            )}
          </div>
          <div className="mt-0.5 font-mono text-xs text-zinc-500">{ix.id}</div>
          <div className="mt-2 truncate font-mono text-xs text-zinc-400" title={ix.root_path}>
            {ix.root_path}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <button className="btn-primary" disabled={busy} onClick={() => onIndex(false)}>
            Index now
          </button>
          <button className="btn-ghost" disabled={busy} onClick={() => onIndex(true)} title="Drop all chunks and rebuild">
            Full rebuild
          </button>
          <button className="btn-ghost" onClick={onAgentsMd} title="Instructions to paste into the project's AGENTS.md / CLAUDE.md">
            AGENTS.md
          </button>
          <button className="btn-ghost" onClick={onEdit}>
            Settings
          </button>
          <button className="btn-danger" disabled={busy} onClick={onDelete}>
            Delete
          </button>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-4 gap-3 border-t border-zinc-800/80 pt-3 text-xs">
        <Stat label="Chunks" value={ix.chunk_count.toLocaleString()} />
        <Stat label="Model" value={ix.embed_model.split("/").pop() ?? ix.embed_model} mono />
        <Stat label="Last indexed" value={timeAgo(ix.last_indexed_at)} title={ix.last_indexed_at ?? ""} />
        <Stat
          label="Auto-sync"
          value={ix.auto_update_on_search ? `on search · ${ix.auto_update_cooldown_hours}h cooldown` : "off"}
        />
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {ix.allowed_extensions.map((e) => (
          <span key={e} className="rounded bg-zinc-800 px-1.5 py-px font-mono text-[11px] text-zinc-400">
            {e}
          </span>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value, mono, title }: { label: string; value: string; mono?: boolean; title?: string }) {
  return (
    <div className="min-w-0" title={title}>
      <div className="text-[10px] tracking-wide text-zinc-500 uppercase">{label}</div>
      <div className={`truncate text-zinc-300 ${mono ? "font-mono" : ""}`}>{value}</div>
    </div>
  );
}
