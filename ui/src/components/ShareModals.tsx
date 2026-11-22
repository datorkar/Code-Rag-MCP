import { useState } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { api } from "../api";
import type { ExportMeta, IndexInfo } from "../types";
import { slugify } from "../presets";
import { Modal } from "./Modal";

const FILTER = [{ name: "Code RAG index", extensions: ["lcrag"] }];
const mb = (b: number) => `${(b / 1e6).toFixed(b > 1e8 ? 0 : 1)} MB`;

export function ExportModal({ ix, onClose }: { ix: IndexInfo; onClose: () => void }) {
  const [label, setLabel] = useState(ix.description ?? ix.name);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const run = async () => {
    const path = await save({ defaultPath: `${ix.id}.lcrag`, filters: FILTER });
    if (!path) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.exportIndex(ix.id, path, label || null);
      setMsg({ ok: true, text: `Exported ${r.chunks.toLocaleString()} chunks to ${r.path} (${mb(r.bytes)}, ${r.secs.toFixed(1)}s)` });
    } catch (e) {
      setMsg({ ok: false, text: String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      title={`Export “${ix.name}”`}
      onClose={onClose}
      locked={busy}
      footer={
        <>
          <button className="btn-ghost" onClick={onClose} disabled={busy}>
            Close
          </button>
          <button className="btn-primary" disabled={busy} onClick={run}>
            {busy ? "Exporting…" : "Export…"}
          </button>
        </>
      }
    >
      <p className="mb-3 text-sm text-zinc-400">
        Saves the built index ({ix.chunk_count.toLocaleString()} chunks) as one compressed <code className="font-mono">.lcrag</code>{" "}
        file that others can import instead of indexing from scratch. Vectors are stored in FP16 (same search results, half
        the size). The file contains the source text — only share it with people allowed to have that source.
      </p>
      <label className="label">Label (shown to whoever imports it)</label>
      <input className="input" value={label} placeholder="e.g. Unreal Engine 5.8.0 Runtime" onChange={(e) => setLabel(e.target.value)} />
      {msg && (
        <p className={`mt-3 rounded px-3 py-2 text-xs whitespace-pre-wrap ${msg.ok ? "bg-emerald-500/10 text-emerald-300" : "bg-red-500/10 text-red-300"}`}>
          {msg.text}
        </p>
      )}
    </Modal>
  );
}

export function ImportModal({ onClose, onImported }: { onClose: () => void; onImported: (id: string) => void }) {
  const [file, setFile] = useState<string | null>(null);
  const [meta, setMeta] = useState<ExportMeta | null>(null);
  const [root, setRoot] = useState("");
  const [name, setName] = useState("");
  const [id, setId] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const pickFile = async () => {
    const f = await open({ multiple: false, filters: FILTER });
    if (typeof f !== "string") return;
    setFile(f);
    setMsg(null);
    try {
      const m = await api.inspectExport(f);
      setMeta(m);
      const n = m.label || m.index.name;
      setName(n);
      setId(slugify(n));
    } catch (e) {
      setMeta(null);
      setMsg({ ok: false, text: String(e) });
    }
  };

  const pickRoot = async () => {
    const d = await open({ directory: true, multiple: false });
    if (typeof d === "string") setRoot(d);
  };

  const run = async () => {
    if (!file) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.importIndex(file, root, id, name || null);
      const warn = r.warnings.length ? `\n${r.warnings.join("\n")}` : "";
      setMsg({
        ok: true,
        text:
          `Imported ${r.chunks.toLocaleString()} chunks from ${r.files.toLocaleString()} files in ${r.secs.toFixed(1)}s.` +
          (r.files_rejected ? ` ${r.files_rejected.toLocaleString()} files differ from your copy and will be re-indexed now.` : "") +
          warn,
      });
      onImported(r.index_id);
    } catch (e) {
      setMsg({ ok: false, text: String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      title="Import index"
      onClose={onClose}
      wide
      locked={busy}
      dismissOnBackdrop={false}
      footer={
        <>
          <button className="btn-ghost" onClick={onClose} disabled={busy}>
            Close
          </button>
          <button className="btn-primary" disabled={busy || !meta || !root || !id} onClick={run}>
            {busy ? "Importing…" : "Import"}
          </button>
        </>
      }
    >
      <div className="space-y-4">
        <div>
          <label className="label">Export file</label>
          <div className="flex gap-2">
            <input className="input font-mono" readOnly value={file ?? ""} placeholder="something.lcrag" />
            <button className="btn-ghost shrink-0" onClick={pickFile}>
              Browse…
            </button>
          </div>
        </div>
        {meta && (
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-md border border-zinc-800 bg-zinc-950/60 p-3 text-xs">
            <span className="text-zinc-500">Label</span>
            <span className="text-zinc-200">{meta.label ?? "—"}</span>
            <span className="text-zinc-500">Model</span>
            <span className="font-mono text-zinc-200">{meta.index.embed_model}</span>
            <span className="text-zinc-500">Content</span>
            <span className="text-zinc-200">
              {meta.chunk_count.toLocaleString()} chunks · {meta.file_count.toLocaleString()} files · {meta.index.allowed_extensions.join(" ")}
            </span>
            <span className="text-zinc-500">Created</span>
            <span className="text-zinc-200">
              {new Date(meta.created_at).toLocaleString()} (app {meta.app_version})
            </span>
          </div>
        )}
        <div>
          <label className="label">Your copy of the source</label>
          <div className="flex gap-2">
            <input className="input font-mono" value={root} placeholder="C:\Program Files\Epic Games\UE_5.8\Engine\Source\Runtime" onChange={(e) => setRoot(e.target.value)} />
            <button className="btn-ghost shrink-0" onClick={pickRoot}>
              Browse…
            </button>
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            Every chunk is checked against these files; anything missing or different is re-indexed locally.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Name</label>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div>
            <label className="label">ID</label>
            <input className="input font-mono" value={id} onChange={(e) => setId(e.target.value.replace(/[^A-Za-z0-9_-]/g, ""))} />
          </div>
        </div>
        {msg && (
          <p className={`rounded px-3 py-2 text-xs whitespace-pre-wrap ${msg.ok ? "bg-emerald-500/10 text-emerald-300" : "bg-red-500/10 text-red-300"}`}>
            {msg.text}
          </p>
        )}
      </div>
    </Modal>
  );
}
