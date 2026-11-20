import { useMemo, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import type { IndexConfig, ModelSpec } from "../types";
import { EXTENSION_PRESETS, IGNORE_PRESETS, slugify } from "../presets";
import { ChipInput } from "./ChipInput";
import { Modal } from "./Modal";

interface Props {
  initial: IndexConfig;
  isNew: boolean;
  models: ModelSpec[];
  onClose: () => void;
  onSave: (ix: IndexConfig, rebuild: boolean) => Promise<void>;
}

export function IndexModal({ initial, isNew, models, onClose, onSave }: Props) {
  const [ix, setIx] = useState<IndexConfig>(initial);
  const [idTouched, setIdTouched] = useState(!isNew);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = <K extends keyof IndexConfig>(k: K, v: IndexConfig[K]) => setIx((p) => ({ ...p, [k]: v }));

  const built = !isNew && initial.last_indexed_at !== null;
  const rebuildNeeded =
    built &&
    (ix.embed_model !== initial.embed_model ||
      ix.chunk_size !== initial.chunk_size ||
      ix.chunk_overlap !== initial.chunk_overlap ||
      (ix.precision === "int8") !== (initial.precision === "int8"));
  const model = useMemo(() => models.find((m) => m.id === ix.embed_model), [models, ix.embed_model]);

  const pickFolder = async () => {
    const dir = await open({ directory: true, multiple: false, defaultPath: ix.root_path || undefined });
    if (typeof dir === "string") {
      set("root_path", dir);
      if (!ix.name) {
        const base = dir.split(/[\\/]/).filter(Boolean).pop() ?? "";
        setIx((p) => ({ ...p, root_path: dir, name: base, id: idTouched ? p.id : slugify(base) }));
      }
    }
  };

  const submit = async (rebuild: boolean) => {
    setBusy(true);
    setError(null);
    try {
      await onSave(ix, rebuild);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  };

  return (
    <Modal
      title={isNew ? "Add index" : `Edit “${initial.name}”`}
      onClose={onClose}
      wide
      footer={
        <>
          <button className="btn-ghost" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          {rebuildNeeded ? (
            <>
              <button className="btn-ghost" onClick={() => submit(false)} disabled={busy}>
                Save only
              </button>
              <button className="btn-primary" onClick={() => submit(true)} disabled={busy}>
                Save &amp; rebuild
              </button>
            </>
          ) : (
            <button className="btn-primary" onClick={() => submit(isNew)} disabled={busy}>
              {isNew ? "Create & index" : "Save"}
            </button>
          )}
        </>
      }
    >
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="label">Name</label>
          <input
            className="input"
            value={ix.name}
            placeholder="My Unreal Project"
            onChange={(e) => {
              const name = e.target.value;
              setIx((p) => ({ ...p, name, id: idTouched ? p.id : slugify(name) }));
            }}
          />
        </div>
        <div>
          <label className="label">ID (used by Claude)</label>
          <input
            className="input font-mono disabled:opacity-60"
            value={ix.id}
            disabled={!isNew}
            placeholder="my-game"
            onChange={(e) => {
              setIdTouched(true);
              set("id", e.target.value.replace(/[^A-Za-z0-9_-]/g, ""));
            }}
          />
        </div>

        <div className="col-span-2">
          <label className="label">Root folder</label>
          <div className="flex gap-2">
            <input
              className="input font-mono"
              value={ix.root_path}
              placeholder="D:\Projects\MyUnrealGame"
              onChange={(e) => set("root_path", e.target.value)}
            />
            <button className="btn-ghost shrink-0" onClick={pickFolder}>
              Browse…
            </button>
          </div>
        </div>

        <div className="col-span-2">
          <label className="label">Description (optional, shown to Claude)</label>
          <input
            className="input"
            value={ix.description ?? ""}
            placeholder="Gameplay code for project X"
            onChange={(e) => set("description", e.target.value || null)}
          />
        </div>

        <div className="col-span-2">
          <label className="label">Embedding model</label>
          <select className="input" value={ix.embed_model} onChange={(e) => set("embed_model", e.target.value)}>
            {!model && <option value={ix.embed_model}>{ix.embed_model} (unknown)</option>}
            <optgroup label="Recommended">
              {models
                .filter((m) => m.recommended)
                .map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id} [{m.max_context} ctx · {m.dim}d]
                  </option>
                ))}
            </optgroup>
            <optgroup label="Other">
              {models
                .filter((m) => !m.recommended)
                .map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id} [{m.max_context} ctx · {m.dim}d]
                  </option>
                ))}
            </optgroup>
          </select>
          {model && <p className="mt-1 text-xs text-zinc-500">{model.description}</p>}
          {rebuildNeeded && (
            <p className="mt-2 rounded border border-amber-600/40 bg-amber-500/10 px-2.5 py-1.5 text-xs text-amber-300">
              Changing the model or chunking invalidates all {initial.chunk_count.toLocaleString()} stored chunks — a
              full rebuild is required before search works again.
            </p>
          )}
        </div>

        <div className="col-span-2">
          <label className="label">Precision</label>
          <div className="grid grid-cols-4 gap-1">
            {(
              [
                ["auto", "Auto", "FP16 on GPU if enabled in Settings, else FP32"],
                ["fp32", "FP32", "Reference quality"],
                ["fp16", "FP16", "Same results, faster on GPU (FP32 on CPU)"],
                ["int8", "INT8", "~1.4x faster on CPU, ~10% of top results differ"],
              ] as const
            ).map(([value, label, hint]) => (
              <button
                key={value}
                type="button"
                title={hint}
                className={`btn justify-center ${ix.precision === value ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"}`}
                onClick={() => set("precision", value)}
              >
                {label}
              </button>
            ))}
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            {
              {
                auto: "FP16 on GPU when “Half precision” is on in Settings, otherwise FP32.",
                fp32: "Full precision: reference search quality.",
                fp16: "Identical results to FP32 (measured), faster on GPU. Uses FP32 on CPU.",
                int8: "Quantized model: ~1.4x faster on CPU, but ~10% of top results differ from FP32. Best for big read-only indexes on CPU. Switching to/from INT8 rebuilds the index.",
              }[ix.precision]
            }
          </p>
        </div>

        <div className="col-span-2">
          <label className="label">File extensions</label>
          <ChipInput
            values={ix.allowed_extensions}
            onChange={(v) => set("allowed_extensions", v)}
            placeholder=".cpp .h .usf"
            presets={EXTENSION_PRESETS}
            normalize={(s) => (s ? (s.startsWith(".") ? s : `.${s}`).toLowerCase() : s)}
          />
        </div>

        <div className="col-span-2">
          <label className="label">Ignore directories (by name, any depth; .gitignore is also respected)</label>
          <ChipInput
            values={ix.ignore_dirs}
            onChange={(v) => set("ignore_dirs", v)}
            placeholder="Binaries Intermediate Saved"
            presets={IGNORE_PRESETS}
          />
        </div>

        <div>
          <label className="label">Chunk size (chars)</label>
          <input
            type="number"
            className="input"
            min={200}
            step={100}
            value={ix.chunk_size}
            onChange={(e) => set("chunk_size", Number(e.target.value))}
          />
        </div>
        <div>
          <label className="label">Chunk overlap (chars)</label>
          <input
            type="number"
            className="input"
            min={0}
            step={50}
            value={ix.chunk_overlap}
            onChange={(e) => set("chunk_overlap", Number(e.target.value))}
          />
        </div>

        <div className="col-span-2 rounded-md border border-zinc-800 p-3">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="accent-indigo-500"
              checked={ix.auto_update_on_search}
              onChange={(e) => set("auto_update_on_search", e.target.checked)}
            />
            Auto-update incrementally when Claude searches
          </label>
          <div className={`mt-3 flex items-center gap-3 ${ix.auto_update_on_search ? "" : "opacity-40"}`}>
            <span className="w-28 text-xs text-zinc-400">Cooldown</span>
            <input
              type="range"
              min={0}
              max={168}
              step={1}
              className="flex-1 accent-indigo-500"
              disabled={!ix.auto_update_on_search}
              value={ix.auto_update_cooldown_hours}
              onChange={(e) => set("auto_update_cooldown_hours", Number(e.target.value))}
            />
            <input
              type="number"
              className="input w-20"
              min={0}
              step={0.5}
              disabled={!ix.auto_update_on_search}
              value={ix.auto_update_cooldown_hours}
              onChange={(e) => set("auto_update_cooldown_hours", Number(e.target.value))}
            />
            <span className="text-xs text-zinc-500">hours</span>
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            0 = check for changed files on every search (cheap: unchanged files are skipped by size + mtime).
          </p>
        </div>
      </div>
      {error && <p className="mt-4 rounded bg-red-500/10 px-3 py-2 text-sm text-red-400">{error}</p>}
    </Modal>
  );
}
