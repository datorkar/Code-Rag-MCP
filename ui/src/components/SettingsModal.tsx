import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExecutionProvider, GlobalSettings, ModelSpec } from "../types";
import { Modal } from "./Modal";

export function SettingsModal({
  initial,
  models,
  onClose,
  onSave,
}: {
  initial: GlobalSettings;
  models: ModelSpec[];
  onClose: () => void;
  onSave: (s: GlobalSettings) => Promise<void>;
}) {
  const [s, setS] = useState(initial);
  const [error, setError] = useState<string | null>(null);
  const providers: ExecutionProvider[] = s.gpu_features.includes("winml")
    ? ["auto", "gpu", "npu", "directml", "cpu"]
    : ["auto", "cuda", "cpu"];
  const [rerankers, setRerankers] = useState<string[]>([]);
  useEffect(() => {
    api.listRerankers().then(setRerankers);
  }, []);
  const num = (k: keyof GlobalSettings) => (e: { target: { value: string } }) =>
    setS({ ...s, [k]: Number(e.target.value) });

  return (
    <Modal
      title="Settings"
      onClose={onClose}
      footer={
        <>
          <button className="btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn-primary"
            onClick={() => onSave(s).catch((e) => setError(String(e)))}
          >
            Save
          </button>
        </>
      }
    >
      <div className="space-y-4">
        <div>
          <label className="label">Default model for new indexes</label>
          <select
            className="input"
            value={s.active_embed_model}
            onChange={(e) => setS({ ...s, active_embed_model: e.target.value })}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id} [{m.max_context} ctx]
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">Execution provider</label>
          <div className="flex gap-1">
            {providers.map((p) => (
              <button
                key={p}
                className={`btn flex-1 justify-center ${s.execution_provider === p ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"}`}
                onClick={() => setS({ ...s, execution_provider: p })}
              >
                {p.toUpperCase()}
              </button>
            ))}
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            GPU support compiled into this build: {s.gpu_features.length ? s.gpu_features.join(", ") : "none (CPU only)"}.
            Unavailable providers fall back to CPU.
          </p>
        </div>
        <div>
          <label className="label">Embedding batches</label>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="mb-1 text-xs text-zinc-500">Token budget per batch</div>
              <input
                type="number"
                className="input"
                min={0}
                step={1024}
                value={s.embed_token_budget}
                onChange={(e) => setS({ ...s, embed_token_budget: Number(e.target.value) })}
              />
            </div>
            <div>
              <div className="mb-1 text-xs text-zinc-500">Max chunks per batch</div>
              <input
                type="number"
                className="input"
                min={0}
                max={512}
                value={s.embed_batch_size}
                onChange={(e) => setS({ ...s, embed_batch_size: Number(e.target.value) })}
              />
            </div>
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            0 = auto (32k tokens on GPU, 8k on CPU, no chunk cap). Batches shrink automatically if the GPU runs out of
            memory.
          </p>
        </div>
        <div>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="accent-indigo-500"
              checked={s.half_precision}
              disabled={s.execution_provider === "cpu"}
              onChange={(e) => setS({ ...s, half_precision: e.target.checked })}
            />
            Half precision (FP16) on GPU
          </label>
          <p className="mt-1 text-xs text-zinc-500">
            Uses the models' FP16 versions: typically 1.5–2x faster on GPU with the same search results. Ignored on CPU
            (FP16 is slower there). Downloads the FP16 model files on first use. Env override:{" "}
            <code className="font-mono">LOCAL_CODE_RAG_FP16=1|0</code>.
          </p>
        </div>
        <div className="border-t border-zinc-800 pt-4">
          <label className="label">Reranker (cross-encoder)</label>
          <select
            className="input"
            value={s.reranker_model ?? ""}
            onChange={(e) => setS({ ...s, reranker_model: e.target.value || null })}
          >
            <option value="">Off — hybrid (RRF) ranking only</option>
            {rerankers.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
          <div className="mt-2 flex items-center gap-2 text-xs text-zinc-400">
            Re-score the top
            <input type="number" className="input w-20" min={5} max={200} value={s.rerank_candidates} onChange={num("rerank_candidates")} disabled={!s.reranker_model} />
            hybrid candidates per search.
          </div>
          <p className="mt-1 text-xs text-zinc-500">Better ranking, ~100–300 ms per search on GPU. Downloaded on first use.</p>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Auto-sync limit (files)</label>
            <input type="number" className="input" min={0} value={s.auto_sync_max_files} onChange={num("auto_sync_max_files")} />
            <p className="mt-1 text-xs text-zinc-500">Searches only sync inline when at most this many files changed. Full builds always run here in the app.</p>
          </div>
          <div>
            <label className="label">Excerpt length (lines)</label>
            <input type="number" className="input" min={0} value={s.snippet_max_lines} onChange={num("snippet_max_lines")} />
            <p className="mt-1 text-xs text-zinc-500">Results are trimmed to the best-matching lines. 0 = full chunk.</p>
          </div>
        </div>
        <div>
          <label className="label">Data directory</label>
          <div className="rounded bg-zinc-950 px-2.5 py-1.5 font-mono text-xs text-zinc-400">{s.data_dir}</div>
        </div>
        {error && <p className="rounded bg-red-500/10 px-3 py-2 text-sm text-red-400">{error}</p>}
      </div>
    </Modal>
  );
}
