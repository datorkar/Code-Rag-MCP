import { useState } from "react";
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
  const providers: ExecutionProvider[] = ["auto", "cuda", "directml", "cpu"];

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
          <label className="label">Embedding batch size</label>
          <input
            type="number"
            className="input w-32"
            min={1}
            max={512}
            value={s.embed_batch_size}
            onChange={(e) => setS({ ...s, embed_batch_size: Number(e.target.value) })}
          />
          <p className="mt-1 text-xs text-zinc-500">32–64 is a good range; larger batches need more VRAM.</p>
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
