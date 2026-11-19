import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api } from "../api";
import type { GpuStatus, WinmlProgress } from "../types";
import { Modal } from "./Modal";

const STATE_LABEL: Record<string, { text: string; cls: string }> = {
  ready: { text: "Ready", cls: "border-emerald-600/40 bg-emerald-500/15 text-emerald-300" },
  not_ready: { text: "Update needed", cls: "border-amber-600/40 bg-amber-500/15 text-amber-300" },
  not_present: { text: "Not installed", cls: "border-zinc-600 bg-zinc-700/40 text-zinc-300" },
};

export function GpuModal({ onClose }: { onClose: () => void }) {
  const [status, setStatus] = useState<GpuStatus | null>(null);
  const [progress, setProgress] = useState<Record<string, WinmlProgress>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.gpuStatus().then(setStatus).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    refresh();
    const un = listen<WinmlProgress>("winml-progress", (e) => {
      const p = e.payload;
      setProgress((x) => ({ ...x, [p.provider]: p }));
      if (p.done) refresh();
    });
    return () => {
      un.then((f) => f());
    };
  }, [refresh]);

  const running = Object.values(progress).some((p) => !p.done);
  useEffect(() => {
    if (!running) setBusy(false);
  }, [running]);

  const install = async (provider: string | null) => {
    setError(null);
    setBusy(true);
    try {
      await api.winmlInstall(provider);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  };

  const rt = status?.runtime;
  const pending = status?.providers.filter((p) => p.certified && p.ready_state !== "ready") ?? [];

  return (
    <Modal title="GPU acceleration" onClose={onClose} wide>
      {!status && !error && <p className="text-sm text-zinc-400">Detecting devices…</p>}
      {status?.runtime_error && (
        <p className="mb-4 rounded bg-red-500/10 px-3 py-2 text-sm text-red-300">{status.runtime_error}</p>
      )}

      {rt && (
        <section className="mb-5">
          <h3 className="label">Runtime</h3>
          <p className="text-sm text-zinc-300">
            {rt.flavour === "winml" ? "Windows ML (ONNX Runtime 1.28)" : "Bundled ONNX Runtime (CUDA build)"}
            {rt.dll_dir && <span className="ml-2 font-mono text-xs text-zinc-500">{rt.dll_dir}</span>}
          </p>
          <p className="mt-1 text-xs text-zinc-500">
            {rt.vendor_gpu
              ? "A vendor GPU provider is active: embeddings run on it when the execution provider is Auto/GPU."
              : rt.flavour === "winml"
                ? "No vendor GPU provider registered yet: GPU mode uses DirectML. Install a provider below for best speed."
                : "GPU mode uses CUDA when the CUDA 13 runtime + cuDNN 9 are installed, otherwise CPU."}
          </p>
          <table className="mt-3 w-full text-left text-xs">
            <thead className="text-zinc-500">
              <tr>
                <th className="py-1 font-medium">Device</th>
                <th className="py-1 font-medium">Vendor</th>
                <th className="py-1 font-medium">Execution provider</th>
              </tr>
            </thead>
            <tbody className="font-mono text-zinc-300">
              {rt.devices.map((d, i) => (
                <tr key={i} className="border-t border-zinc-800">
                  <td className="py-1.5 uppercase">{d.kind}</td>
                  <td className="py-1.5">{d.vendor}</td>
                  <td className="py-1.5">{d.ep}</td>
                </tr>
              ))}
              {rt.devices.length === 0 && (
                <tr>
                  <td colSpan={3} className="py-1.5 text-zinc-500">
                    No devices reported
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </section>
      )}

      {status?.windows_ml && (
        <section>
          <div className="mb-2 flex items-center justify-between">
            <h3 className="label mb-0">Windows ML execution providers</h3>
            <div className="flex gap-2">
              {running ? (
                <button className="btn-danger px-2 py-1 text-xs" onClick={() => api.winmlCancel()}>
                  Cancel
                </button>
              ) : (
                <button className="btn-primary px-2.5 py-1 text-xs" disabled={busy || pending.length === 0} onClick={() => install(null)}>
                  {pending.length ? `Download & enable all (${pending.length})` : "All installed"}
                </button>
              )}
            </div>
          </div>
          {status.providers_error && (
            <p className="mb-2 rounded bg-red-500/10 px-3 py-2 text-xs text-red-300">{status.providers_error}</p>
          )}
          <div className="space-y-2">
            {status.providers.map((p) => {
              const pr = progress[p.name];
              const st = STATE_LABEL[p.ready_state] ?? STATE_LABEL.not_present;
              return (
                <div key={p.name} className="rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0 text-sm">
                      <span className="font-medium text-zinc-100">{p.name}</span>
                      {p.version && <span className="ml-2 font-mono text-xs text-zinc-500">{p.version}</span>}
                      <span className={`ml-2 rounded border px-1.5 py-px text-[11px] ${st.cls}`}>{st.text}</span>
                      {p.registered && <span className="ml-2 text-[11px] text-emerald-400">● in use</span>}
                      {!p.certified && <span className="ml-2 text-[11px] text-zinc-500">uncertified</span>}
                    </div>
                    {p.ready_state !== "ready" && (
                      <button
                        className="btn-ghost shrink-0 px-2 py-1 text-xs"
                        disabled={busy || running}
                        onClick={() => install(p.name)}
                      >
                        Download & enable
                      </button>
                    )}
                  </div>
                  {pr && !pr.done && (
                    <div className="mt-2 h-1.5 overflow-hidden rounded bg-zinc-800">
                      <div className="h-full bg-indigo-500 transition-[width]" style={{ width: `${Math.max(2, pr.progress)}%` }} />
                    </div>
                  )}
                  {pr?.done && !pr.ok && <p className="mt-1 text-xs text-red-300">{pr.message}</p>}
                </div>
              );
            })}
            {status.providers.length === 0 && !status.providers_error && (
              <p className="text-xs text-zinc-500">
                Windows ML offers no extra providers for this device (DirectML and CPU are always available).
              </p>
            )}
          </div>
          <p className="mt-3 text-xs text-zinc-500">
            Providers are downloaded and kept up to date by Windows. The app uses newly enabled providers immediately;
            restart Claude Code so the MCP server picks them up too.
          </p>
        </section>
      )}
      {error && <p className="mt-3 rounded bg-red-500/10 px-3 py-2 text-sm text-red-300">{error}</p>}
    </Modal>
  );
}
