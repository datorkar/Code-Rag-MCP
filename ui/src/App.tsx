import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { ask } from "@tauri-apps/plugin-dialog";
import { api } from "./api";
import { DEFAULT_INDEX } from "./presets";
import type { FinishedEvent, GlobalSettings, IndexConfig, IndexInfo, ModelSpec, Progress, UpdateInfo } from "./types";
import { IndexCard } from "./components/IndexCard";
import { IndexModal } from "./components/IndexModal";
import { ProgressDrawer } from "./components/ProgressDrawer";
import { ClaudeModal } from "./components/ClaudeModal";
import { SettingsModal } from "./components/SettingsModal";
import { AgentsMdModal } from "./components/AgentsMdModal";
import { GpuModal } from "./components/GpuModal";
import { ExportModal, ImportModal } from "./components/ShareModals";
import { UpdateBanner } from "./components/UpdateBanner";
import { SetupWizard } from "./components/SetupWizard";

type Editing = { ix: IndexConfig; isNew: boolean } | null;

export default function App() {
  const [indexes, setIndexes] = useState<IndexInfo[]>([]);
  const [models, setModels] = useState<ModelSpec[]>([]);
  const [settings, setSettings] = useState<GlobalSettings | null>(null);
  const [jobs, setJobs] = useState<Record<string, Progress>>({});
  const [finished, setFinished] = useState<Record<string, FinishedEvent>>({});
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [editing, setEditing] = useState<Editing>(null);
  const [showClaude, setShowClaude] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [agentsFor, setAgentsFor] = useState<IndexInfo | null>(null);
  const [showGpu, setShowGpu] = useState(false);
  const [exportFor, setExportFor] = useState<IndexInfo | null>(null);
  const [showImport, setShowImport] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [update, setUpdate] = useState<UpdateInfo | null>(null);
  const [showSetup, setShowSetup] = useState(false);

  const refresh = useCallback(() => {
    api
      .listIndexes()
      // Keep the old array when nothing changed so memoised cards don't re-render on every poll.
      .then((next) => setIndexes((prev) => (JSON.stringify(prev) === JSON.stringify(next) ? prev : next)))
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    refresh();
    api.listModels().then(setModels).catch((e) => setError(String(e)));
    api
      .getSettings()
      .then((s) => {
        setSettings(s);
        // First launch: guided setup (hardware, GPU providers, model, Claude Code).
        if (!s.setup_completed && s.index_count === 0) setShowSetup(true);
      })
      .catch((e) => setError(String(e)));
    // Automatic update check (setting-controlled, at most daily); failures are silent.
    api
      .checkUpdate(false)
      .then((u) => u?.available && setUpdate(u))
      .catch(() => {});
    // Pick up MCP-side auto-syncs; skip while the window is hidden/minimised.
    const timer = setInterval(() => {
      if (document.visibilityState === "visible") refresh();
    }, 15000);
    const onVisible = () => document.visibilityState === "visible" && refresh();
    document.addEventListener("visibilitychange", onVisible);
    const un1 = listen<Progress>("index-progress", (e) => {
      const p = e.payload;
      if (p.phase === "done" || p.phase === "cancelled" || p.phase === "error") return;
      setJobs((j) => ({ ...j, [p.index_id]: p }));
    });
    const un2 = listen<FinishedEvent>("index-finished", (e) => {
      const f = e.payload;
      setJobs(({ [f.index_id]: _, ...rest }) => rest);
      setFinished((x) => ({ ...x, [f.index_id]: f }));
      refresh();
    });
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
      un1.then((f) => f());
      un2.then((f) => f());
    };
  }, [refresh]);

  const startIndex = useCallback(async (id: string, force: boolean) => {
    if (force && !(await ask(`Drop all chunks of “${id}” and rebuild from scratch?`, { kind: "warning" }))) return;
    try {
      setFinished(({ [id]: _, ...rest }) => rest);
      setJobs((j) => ({
        ...j,
        [id]: {
          index_id: id,
          phase: "waiting",
          files_scanned: 0,
          files_total: 0,
          files_changed: 0,
          chunks_embedded: 0,
          chunks_total: 0,
          chunks_per_sec: 0,
          elapsed_secs: 0,
          eta_secs: null,
          files_chunked: 0,
          files_to_chunk: 0,
          index_step: 0,
          index_steps: 0,
          message: null,
        },
      }));
      setDrawerOpen(true);
      await api.startIndex(id, force);
    } catch (e) {
      setJobs(({ [id]: _, ...rest }) => rest);
      setError(String(e));
    }
  }, []);

  const deleteIndex = useCallback(async (ix: IndexInfo) => {
    if (!(await ask(`Delete index “${ix.name}” and all its stored chunks? Source files are not touched.`, { kind: "warning" })))
      return;
    try {
      await api.deleteIndex(ix.id);
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }, [refresh]);

  // Stable handlers for the memoised cards.
  const onCardIndex = useCallback((ix: IndexInfo, force: boolean) => startIndex(ix.id, force), [startIndex]);
  const onCardEdit = useCallback((ix: IndexInfo) => setEditing({ ix, isNew: false }), []);
  const newIndex = () =>
    setEditing({
      ix: {
        ...DEFAULT_INDEX,
        embed_model: settings?.active_embed_model ?? DEFAULT_INDEX.embed_model,
        precision: settings?.default_precision ?? DEFAULT_INDEX.precision,
      },
      isNew: true,
    });

  const saveIndex = async (ix: IndexConfig, rebuild: boolean) => {
    const res = await api.saveIndex(ix, editing?.isNew ? null : (editing?.ix.id ?? null));
    setEditing(null);
    if (res.note) setNotice(res.note);
    refresh();
    if (rebuild || res.created) startIndex(res.index.id, rebuild && !res.created && res.needs_rebuild);
  };

  const totalChunks = indexes.reduce((n, i) => n + i.chunk_count, 0);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-zinc-50">Local Code RAG</h1>
          <p className="text-xs text-zinc-500">
            {indexes.length} index{indexes.length === 1 ? "" : "es"} · {totalChunks.toLocaleString()} chunks
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button className="btn-ghost" onClick={() => setShowImport(true)}>
            Import
          </button>
          <button className="btn-ghost" onClick={() => setShowGpu(true)}>
            GPU
          </button>
          <button className="btn-ghost" onClick={() => setShowSettings(true)} disabled={!settings}>
            Settings
          </button>
          <button className="btn-ghost" onClick={() => setShowClaude(true)}>
            Install to Claude Code
          </button>
          <button className="btn-primary" onClick={newIndex}>
            + Add index
          </button>
        </div>
      </header>

      {error && (
        <div className="flex items-center justify-between bg-red-500/10 px-5 py-2 text-sm text-red-300">
          <span className="whitespace-pre-wrap">{error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {update && <UpdateBanner update={update} onClose={() => setUpdate(null)} />}

      {notice && (
        <div className="flex items-center justify-between bg-amber-500/10 px-5 py-2 text-sm text-amber-300">
          <span>{notice}</span>
          <button onClick={() => setNotice(null)}>×</button>
        </div>
      )}

      <main className="flex-1 overflow-y-auto p-5">
        {indexes.length === 0 ? (
          <div className="mx-auto mt-24 max-w-md text-center">
            <h2 className="text-base font-medium text-zinc-200">No indexes yet</h2>
            <p className="mt-1 text-sm text-zinc-500">
              Add a source folder (your Unreal project, engine headers, …). It is chunked, embedded locally and
              exposed to Claude Code via MCP.
            </p>
            <button
              className="btn-primary mt-4"
              onClick={newIndex}
            >
              + Add your first index
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {indexes.map((ix) => (
              <IndexCard
                key={ix.id}
                ix={ix}
                progress={jobs[ix.id]}
                onIndex={onCardIndex}
                onEdit={onCardEdit}
                onDelete={deleteIndex}
                onAgentsMd={setAgentsFor}
                onExport={setExportFor}
              />
            ))}
          </div>
        )}
      </main>

      <ProgressDrawer
        jobs={jobs}
        finished={finished}
        indexes={indexes}
        open={drawerOpen}
        onToggle={() => setDrawerOpen((o) => !o)}
        onCancel={(id) => api.cancelIndex(id).catch((e) => setError(String(e)))}
        onDismiss={(id) => setFinished(({ [id]: _, ...rest }) => rest)}
      />

      {editing && (
        <IndexModal
          initial={editing.ix}
          isNew={editing.isNew}
          models={models}
          onClose={() => setEditing(null)}
          onSave={saveIndex}
        />
      )}
      {exportFor && <ExportModal ix={exportFor} onClose={() => setExportFor(null)} />}
      {showImport && (
        <ImportModal
          onClose={() => setShowImport(false)}
          onImported={(id) => {
            refresh();
            startIndex(id, false);
          }}
        />
      )}
      {showGpu && <GpuModal onClose={() => setShowGpu(false)} />}
      {agentsFor && <AgentsMdModal ix={agentsFor} onClose={() => setAgentsFor(null)} />}
      {showClaude && <ClaudeModal onClose={() => setShowClaude(false)} />}
      {showSettings && settings && (
        <SettingsModal
          initial={settings}
          models={models}
          onUpdateFound={setUpdate}
          onRunSetup={() => {
            setShowSetup(true);
            setShowSettings(false);
          }}
          onClose={() => setShowSettings(false)}
          onSave={async (s) => {
            setSettings(await api.saveSettings(s));
            setShowSettings(false);
          }}
        />
      )}
      {showSetup && settings && (
        <SetupWizard
          settings={settings}
          models={models}
          onFinish={async (s, addIndex) => {
            const saved = await api.saveSettings(s);
            setSettings(saved);
            setShowSetup(false);
            if (addIndex)
              setEditing({
                ix: { ...DEFAULT_INDEX, embed_model: saved.active_embed_model, precision: saved.default_precision },
                isNew: true,
              });
          }}
        />
      )}
    </div>
  );
}
