import { useEffect, useMemo, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api } from "../api";
import type { GlobalSettings, GpuStatus, HardwareSummary, ModelSpec, WinmlProgress } from "../types";

const STEPS = ["Welcome", "Hardware", "Model", "Claude Code", "Done"] as const;

const MODEL_NOTES: Record<string, string> = {
  "jinaai/jina-embeddings-v2-base-code": "Trained on code. Best all-round choice.",
  "BAAI/bge-small-en-v1.5": "Small and ~5× faster. Laptops and huge codebases.",
  "Alibaba-NLP/gte-base-en-v1.5": "Strong general retrieval; mixed code + docs.",
  "lightonai/modernbert-embed-large": "Highest quality, slowest. Needs a strong GPU.",
};

type Choice = Pick<GlobalSettings, "active_embed_model" | "default_precision" | "resource_mode">;

/** First-launch setup: hardware + GPU providers, model choice, Claude Code registration. */
export function SetupWizard({
  settings,
  models,
  onFinish,
}: {
  settings: GlobalSettings;
  models: ModelSpec[];
  /** Saves the settings; `addIndex` opens "Add index" afterwards. */
  onFinish: (s: GlobalSettings, addIndex: boolean) => Promise<void>;
}) {
  const [step, setStep] = useState(0);
  const [hw, setHw] = useState<HardwareSummary | null>(null);
  const [gpu, setGpu] = useState<GpuStatus | null>(null);
  const [progress, setProgress] = useState<Record<string, WinmlProgress>>({});
  const [installing, setInstalling] = useState(false);
  const [choice, setChoice] = useState<Choice | null>(null);
  const [claudeMsg, setClaudeMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshGpu = () => api.gpuStatus().then(setGpu).catch((e) => setError(String(e)));

  useEffect(() => {
    api
      .hardwareSummary()
      .then((h) => {
        setHw(h);
        setChoice({
          active_embed_model: h.recommendation.model,
          default_precision: h.recommendation.precision,
          resource_mode: h.recommendation.resource_mode,
        });
      })
      .catch((e) => setError(String(e)));
    refreshGpu();
    const un = listen<WinmlProgress>("winml-progress", (e) => {
      const p = e.payload;
      setProgress((x) => ({ ...x, [p.provider]: p }));
      if (p.done) refreshGpu();
    });
    const un2 = listen("winml-install-finished", () => {
      setInstalling(false);
      refreshGpu();
    });
    return () => {
      un.then((f) => f());
      un2.then((f) => f());
    };
  }, []);

  const pending = gpu?.providers.filter((p) => p.certified && p.ready_state !== "ready") ?? [];
  const ready = gpu?.providers.filter((p) => p.ready_state === "ready") ?? [];
  const recommended = useMemo(() => models.filter((m) => m.recommended), [models]);

  const installProviders = async () => {
    setError(null);
    setInstalling(true);
    try {
      await api.winmlInstall(null);
    } catch (e) {
      setError(String(e));
      setInstalling(false);
    }
  };

  const registerClaude = async () => {
    setBusy(true);
    setClaudeMsg(null);
    try {
      setClaudeMsg({ ok: true, text: await api.registerClaudeMcp() });
    } catch (e) {
      setClaudeMsg({ ok: false, text: String(e) });
    } finally {
      setBusy(false);
    }
  };

  const finish = async (addIndex: boolean, skipped = false) => {
    setBusy(true);
    setError(null);
    try {
      const next: GlobalSettings =
        skipped || !choice
          ? { ...settings, setup_completed: true }
          : { ...settings, ...choice, execution_provider: "auto", setup_completed: true };
      await onFinish(next, addIndex);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  };

  const spec = (id: string) => models.find((m) => m.id === id);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/95 p-4">
      <div className="flex max-h-full w-full max-w-2xl flex-col rounded-lg border border-zinc-800 bg-zinc-900 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div className="flex gap-3 text-xs">
            {STEPS.map((s, i) => (
              <span key={s} className={i === step ? "font-semibold text-indigo-300" : i < step ? "text-zinc-400" : "text-zinc-600"}>
                {i + 1}. {s}
              </span>
            ))}
          </div>
          <button className="text-xs text-zinc-500 hover:text-zinc-200" onClick={() => finish(false, true)} disabled={busy}>
            Skip setup
          </button>
        </div>

        <div className="min-h-[340px] overflow-y-auto px-6 py-5 text-sm text-zinc-300">
          {step === 0 && (
            <div className="space-y-3">
              <h2 className="text-lg font-semibold text-zinc-50">Welcome to Local Code RAG</h2>
              <p>
                This app indexes your code locally so Claude Code can search it by meaning and by name. Nothing leaves
                your PC.
              </p>
              <p>A few quick steps:</p>
              <ol className="list-decimal space-y-1 pl-5 text-zinc-400">
                <li>Check your hardware and enable GPU acceleration.</li>
                <li>Pick an embedding model that suits it.</li>
                <li>Connect the search tool to Claude Code.</li>
              </ol>
              <p className="text-xs text-zinc-500">You can change everything later in Settings and the GPU dialog.</p>
            </div>
          )}

          {step === 1 && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-zinc-50">Your hardware</h2>
              {!hw && !error && <p className="text-zinc-400">Detecting…</p>}
              {hw && (
                <div className="rounded-md border border-zinc-800 p-3 text-xs">
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <span className="text-zinc-500">CPU threads</span> {hw.hardware.cpu_threads}
                    </div>
                    <div>
                      <span className="text-zinc-500">Memory</span> {hw.hardware.ram_gb.toFixed(0)} GB
                    </div>
                  </div>
                  <div className="mt-2 space-y-1">
                    {hw.hardware.gpus.length === 0 && <div className="text-zinc-500">No GPU detected.</div>}
                    {hw.hardware.gpus.map((g) => (
                      <div key={g.name}>
                        <span className="text-zinc-200">{g.name}</span>{" "}
                        <span className="text-zinc-500">
                          · {g.integrated ? "integrated" : "discrete"}
                          {g.vram_gb > 0 && !g.integrated ? ` · ${g.vram_gb.toFixed(0)} GB` : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {gpu?.windows_ml ? (
                <div className="rounded-md border border-zinc-800 p-3">
                  <h3 className="mb-1 font-medium text-zinc-100">GPU / NPU acceleration (Windows ML)</h3>
                  {ready.length > 0 && (
                    <p className="text-xs text-emerald-300">Enabled: {ready.map((p) => p.name).join(", ")}</p>
                  )}
                  {pending.length > 0 ? (
                    <>
                      <p className="mt-1 text-xs text-zinc-400">
                        Windows can download optimized providers for your hardware:{" "}
                        {pending.map((p) => p.name).join(", ")}. Recommended; this may take a few minutes.
                      </p>
                      <div className="mt-2 space-y-1">
                        {Object.values(progress).map((p) => (
                          <div key={p.provider} className="text-xs">
                            {p.provider}:{" "}
                            {p.done ? (p.ok ? "ready" : `failed (${p.message ?? "unknown"})`) : `${p.progress.toFixed(0)}%`}
                          </div>
                        ))}
                      </div>
                      <button className="btn-primary mt-2" onClick={installProviders} disabled={installing}>
                        {installing ? "Downloading…" : "Download & enable"}
                      </button>
                    </>
                  ) : (
                    ready.length === 0 && (
                      <p className="text-xs text-zinc-400">
                        No vendor providers offered for this PC. DirectML (any DirectX 12 GPU) or the CPU will be used.
                      </p>
                    )
                  )}
                  {gpu.providers_error && <p className="mt-1 text-xs text-amber-300">{gpu.providers_error}</p>}
                </div>
              ) : (
                gpu && <p className="text-xs text-zinc-500">GPU acceleration uses CUDA when available, otherwise the CPU.</p>
              )}
            </div>
          )}

          {step === 2 && choice && hw && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-zinc-50">Embedding model</h2>
              <p className="rounded bg-indigo-500/10 px-3 py-2 text-xs text-indigo-200">{hw.recommendation.reason}</p>
              <div className="space-y-1.5">
                {recommended.map((m) => (
                  <label
                    key={m.id}
                    className={`flex cursor-pointer items-start gap-3 rounded-md border px-3 py-2 ${
                      choice.active_embed_model === m.id ? "border-indigo-500 bg-indigo-500/10" : "border-zinc-800 hover:border-zinc-700"
                    }`}
                  >
                    <input
                      type="radio"
                      className="mt-1 accent-indigo-500"
                      checked={choice.active_embed_model === m.id}
                      onChange={() => setChoice({ ...choice, active_embed_model: m.id })}
                    />
                    <div className="min-w-0">
                      <div className="text-zinc-100">
                        {m.id.split("/").pop()}
                        {m.id === hw.recommendation.model && <span className="ml-2 text-xs text-indigo-300">recommended</span>}
                      </div>
                      <div className="text-xs text-zinc-500">
                        {MODEL_NOTES[m.id] ?? m.description} · {m.size_gb.toFixed(2)} GB · {m.max_context} tokens
                      </div>
                    </div>
                  </label>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Precision for new indexes</label>
                  <select
                    className="input"
                    value={choice.default_precision}
                    onChange={(e) => setChoice({ ...choice, default_precision: e.target.value as Choice["default_precision"] })}
                  >
                    <option value="standard">Standard (full quality)</option>
                    <option value="int8">INT8 (faster on CPU)</option>
                  </select>
                </div>
                <div>
                  <label className="label">Resource use while indexing</label>
                  <select
                    className="input"
                    value={choice.resource_mode}
                    onChange={(e) => setChoice({ ...choice, resource_mode: e.target.value as Choice["resource_mode"] })}
                  >
                    <option value="auto">Auto</option>
                    <option value="low">Low (keep the PC responsive)</option>
                    <option value="max">Max (fastest)</option>
                  </select>
                </div>
              </div>
              {spec(choice.active_embed_model) && spec(choice.active_embed_model)!.max_context <= 512 && (
                <p className="text-xs text-zinc-500">This model reads short chunks; chunk size is adjusted automatically.</p>
              )}
            </div>
          )}

          {step === 3 && (
            <div className="space-y-3">
              <h2 className="text-lg font-semibold text-zinc-50">Connect to Claude Code</h2>
              <p>
                Registers the search tool with Claude Code (<code className="font-mono text-xs">claude mcp add</code>, user
                scope), so every project can use your indexes. Requires Claude Code to be installed.
              </p>
              <button className="btn-primary" onClick={registerClaude} disabled={busy}>
                {busy ? "Registering…" : "Register with Claude Code"}
              </button>
              {claudeMsg && (
                <p className={`rounded px-3 py-2 text-xs whitespace-pre-wrap ${claudeMsg.ok ? "bg-emerald-500/10 text-emerald-300" : "bg-red-500/10 text-red-300"}`}>
                  {claudeMsg.text}
                  {!claudeMsg.ok && "\nYou can do this later with “Install to Claude Code” in the header."}
                </p>
              )}
            </div>
          )}

          {step === 4 && choice && (
            <div className="space-y-3">
              <h2 className="text-lg font-semibold text-zinc-50">All set</h2>
              <ul className="space-y-1 text-zinc-400">
                <li>
                  Model: <span className="text-zinc-200">{choice.active_embed_model.split("/").pop()}</span>
                  {choice.default_precision === "int8" && " (INT8)"}
                </li>
                <li>
                  Resource use: <span className="text-zinc-200">{choice.resource_mode}</span>
                </li>
                <li>
                  GPU providers: <span className="text-zinc-200">{ready.length > 0 ? ready.map((p) => p.name).join(", ") : "DirectML / CPU"}</span>
                </li>
              </ul>
              <p>Next, add a folder to index, for example your project or the engine source.</p>
            </div>
          )}

          {error && <p className="mt-4 rounded bg-red-500/10 px-3 py-2 text-xs text-red-300">{error}</p>}
        </div>

        <div className="flex justify-between border-t border-zinc-800 px-5 py-3">
          <button className="btn-ghost" onClick={() => setStep((s) => s - 1)} disabled={step === 0 || busy}>
            Back
          </button>
          {step < STEPS.length - 1 ? (
            <button className="btn-primary" onClick={() => setStep((s) => s + 1)} disabled={(step === 1 && installing) || (step >= 1 && !choice)}>
              {step === 3 && !claudeMsg?.ok ? "Skip" : "Next"}
            </button>
          ) : (
            <div className="flex gap-2">
              <button className="btn-ghost" onClick={() => finish(false)} disabled={busy}>
                Finish
              </button>
              <button className="btn-primary" onClick={() => finish(true)} disabled={busy}>
                Add first index
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
