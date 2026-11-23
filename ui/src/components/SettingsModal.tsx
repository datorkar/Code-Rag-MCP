import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExecutionProvider, GlobalSettings, ModelSpec, UpdateInfo } from "../types";
import { Modal } from "./Modal";

export function SettingsModal({
  initial,
  models,
  onClose,
  onSave,
  onUpdateFound,
}: {
  initial: GlobalSettings;
  models: ModelSpec[];
  onClose: () => void;
  onSave: (s: GlobalSettings) => Promise<void>;
  onUpdateFound: (u: UpdateInfo) => void;
}) {
  const [checking, setChecking] = useState(false);
  const [updateMsg, setUpdateMsg] = useState<{ ok: boolean; text: string; info?: UpdateInfo } | null>(null);
  const checkNow = async () => {
    setChecking(true);
    setUpdateMsg(null);
    try {
      const u = await api.checkUpdate(true);
      if (u?.available) {
        setUpdateMsg({ ok: true, text: `Version ${u.latest} is available.`, info: u });
        onUpdateFound(u);
      } else {
        setUpdateMsg({ ok: true, text: `You have the latest version (${initial.app_version}).` });
      }
    } catch (e) {
      setUpdateMsg({ ok: false, text: `Could not check for updates: ${e}` });
    } finally {
      setChecking(false);
    }
  };
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
                {m.id} [{m.dim}d · {m.max_context} ctx · {m.size_gb.toFixed(2)} GB]
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
          <label className="label">Resource use while indexing</label>
          <div className="flex gap-1">
            {(
              [
                ["auto", "Auto"],
                ["low", "Low (laptop)"],
                ["max", "Max"],
              ] as const
            ).map(([v, label]) => (
              <button
                key={v}
                className={`btn flex-1 justify-center ${s.resource_mode === v ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"}`}
                onClick={() => setS({ ...s, resource_mode: v })}
              >
                {label}
              </button>
            ))}
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            {
              {
                auto: "Below-normal priority. Integrated GPUs and CPU-only machines use Low automatically; discrete GPUs run at full speed.",
                low: "Below-normal priority, all CPU threads but 2, one worker, small batches. Keeps laptops responsive.",
                max: "Normal priority, all cores, 2 GPU workers, large batches. Fastest; the PC may feel sluggish.",
              }[s.resource_mode]
            }
          </p>
        </div>
        <div>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="accent-indigo-500"
              checked={s.force_fp32}
              onChange={(e) => setS({ ...s, force_fp32: e.target.checked })}
            />
            Force FP32 (troubleshooting)
          </label>
          <p className="mt-1 text-xs text-zinc-500">
            By default models run in FP16 on GPU (same results, faster) and FP32 on CPU. Tick this only if your GPU is
            slow at FP16 (e.g. GTX 10-series) or shows errors. Env override:{" "}
            <code className="font-mono">LOCAL_CODE_RAG_FP16=0|1</code>.
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
        <div className="rounded-md border border-zinc-800 p-3">
          <div className="flex items-center justify-between gap-3">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                className="accent-indigo-500"
                checked={s.check_updates}
                onChange={(e) => setS({ ...s, check_updates: e.target.checked })}
              />
              Check for updates automatically
            </label>
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-500">Version {s.app_version}</span>
              <button className="btn-ghost" onClick={checkNow} disabled={checking}>
                {checking ? "Checking…" : "Check now"}
              </button>
            </div>
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            At startup, at most once a day, the app checks GitHub for a newer signed release (no data is sent). You
            choose when to install; the installer is verified against the app's built-in key before it runs.
          </p>
          {updateMsg && (
            <p className={`mt-2 text-xs ${updateMsg.ok ? "text-emerald-300" : "text-red-300"}`}>
              {updateMsg.text}
              {updateMsg.info && " Use the banner at the top of the window to install it."}
            </p>
          )}
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
