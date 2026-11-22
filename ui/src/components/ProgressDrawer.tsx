import type { FinishedEvent, IndexInfo, Progress } from "../types";

const PHASE_LABEL: Record<Progress["phase"], string> = {
  waiting: "Waiting for lock",
  scanning: "Scanning files",
  hashing: "Hashing files",
  deleting: "Removing stale chunks",
  chunking: "Chunking",
  loading_model: "Loading model",
  embedding: "Embedding",
  building_index: "Building full-text index",
  done: "Done",
  cancelled: "Cancelled",
  error: "Error",
};

function fmtEta(secs: number): string {
  const s = Math.ceil(secs);
  if (s < 90) return `${s}s`;
  const m = Math.round(s / 60);
  return m < 90 ? `${m} min` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

function Bar({ value, total, color = "bg-indigo-500" }: { value: number; total: number; color?: string }) {
  const pct = total > 0 ? Math.min(100, (value / total) * 100) : 0;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded bg-zinc-800">
      <div className={`h-full ${color} transition-[width] duration-200`} style={{ width: `${pct}%` }} />
    </div>
  );
}

interface Props {
  jobs: Record<string, Progress>;
  finished: Record<string, FinishedEvent>;
  indexes: IndexInfo[];
  open: boolean;
  onToggle: () => void;
  onCancel: (id: string) => void;
  onDismiss: (id: string) => void;
}

export function ProgressDrawer({ jobs, finished, indexes, open, onToggle, onCancel, onDismiss }: Props) {
  const ids = Array.from(new Set([...Object.keys(jobs), ...Object.keys(finished)]));
  if (ids.length === 0) return null;
  const running = Object.keys(jobs).length;
  const name = (id: string) => indexes.find((i) => i.id === id)?.name ?? id;

  return (
    <div className="border-t border-zinc-800 bg-zinc-900">
      <button
        className="flex w-full items-center justify-between px-5 py-2 text-left text-xs text-zinc-400 hover:text-zinc-200"
        onClick={onToggle}
      >
        <span>
          {running > 0 ? (
            <span className="text-indigo-300">● {running} indexing job{running > 1 ? "s" : ""} running</span>
          ) : (
            "Indexing activity"
          )}
        </span>
        <span>{open ? "▾" : "▴"}</span>
      </button>
      {open && (
        <div className="max-h-72 space-y-3 overflow-y-auto px-5 pb-4">
          {ids.map((id) => {
            const p = jobs[id];
            const f = finished[id];
            if (p) {
              const embedding = p.phase === "embedding" || p.chunks_total > 0;
              return (
                <div key={id} className="rounded-md border border-zinc-800 bg-zinc-950/60 p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="text-sm">
                      <span className="font-medium text-zinc-100">{name(id)}</span>
                      <span className="ml-2 text-xs text-zinc-400">{PHASE_LABEL[p.phase]}</span>
                    </div>
                    <button className="btn-danger px-2 py-0.5 text-xs" onClick={() => onCancel(id)}>
                      Cancel
                    </button>
                  </div>
                  <div className="space-y-2 text-xs text-zinc-400">
                    <div>
                      <div className="mb-1 flex justify-between">
                        <span>Files scanned</span>
                        <span className="font-mono">
                          {p.files_scanned.toLocaleString()} / {p.files_total.toLocaleString() || "…"}
                          {p.files_changed > 0 && ` · ${p.files_changed.toLocaleString()} changed`}
                        </span>
                      </div>
                      <Bar value={p.files_scanned} total={p.files_total || 1} color="bg-zinc-500" />
                    </div>
                    {p.phase === "chunking" && p.files_to_chunk > 0 && (
                      <div>
                        <div className="mb-1 flex justify-between">
                          <span>Files chunked</span>
                          <span className="font-mono">
                            {p.files_chunked.toLocaleString()} / {p.files_to_chunk.toLocaleString()}
                          </span>
                        </div>
                        <Bar value={p.files_chunked} total={p.files_to_chunk} color="bg-sky-500" />
                      </div>
                    )}
                    {embedding && (
                      <div>
                        <div className="mb-1 flex justify-between">
                          <span>Chunks embedded</span>
                          <span className="font-mono">
                            {p.chunks_embedded.toLocaleString()} / {p.chunks_total.toLocaleString()}
                            {p.chunks_per_sec > 0 && ` · ${p.chunks_per_sec.toFixed(1)} chunks/s`}
                            {p.eta_secs != null && p.chunks_total > p.chunks_embedded && ` · ~${fmtEta(p.eta_secs)} left`}
                          </span>
                        </div>
                        <Bar value={p.chunks_embedded} total={p.chunks_total} />
                      </div>
                    )}
                    {p.phase === "building_index" && p.index_steps > 0 && (
                      <div>
                        <div className="mb-1 flex justify-between">
                          <span>Indexes</span>
                          <span className="font-mono">
                            step {Math.min(p.index_step + 1, p.index_steps)} / {p.index_steps}
                          </span>
                        </div>
                        <Bar value={p.index_step} total={p.index_steps} color="bg-emerald-500" />
                      </div>
                    )}
                    {p.message && <div className="truncate text-zinc-500">{p.message}</div>}
                  </div>
                </div>
              );
            }
            return (
              <div
                key={id}
                className={`flex items-start justify-between gap-3 rounded-md border px-3 py-2 text-xs ${
                  f.ok
                    ? "border-emerald-700/40 bg-emerald-500/5 text-emerald-300"
                    : f.cancelled
                      ? "border-zinc-700 text-zinc-400"
                      : "border-red-700/40 bg-red-500/5 text-red-300"
                }`}
              >
                <div>
                  <span className="font-medium">{name(id)}</span> —{" "}
                  {f.ok ? "indexed" : f.cancelled ? "cancelled (progress kept)" : "failed"}:{" "}
                  <span className="opacity-80">{f.message}</span>
                </div>
                <button className="text-zinc-500 hover:text-zinc-200" onClick={() => onDismiss(id)}>
                  ×
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
