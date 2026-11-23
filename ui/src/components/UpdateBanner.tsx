import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api } from "../api";
import type { UpdateInfo, UpdateProgress } from "../types";

/** "Version X is available" bar with one-click install (signed Tauri update). */
export function UpdateBanner({ update, onClose }: { update: UpdateInfo; onClose: () => void }) {
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<UpdateProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const un = listen<UpdateProgress>("update-progress", (e) => setProgress(e.payload));
    return () => {
      un.then((f) => f());
    };
  }, []);

  const install = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.installUpdate(); // the app restarts when done
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  };

  const pct = progress?.total ? Math.round((progress.downloaded / progress.total) * 100) : null;
  return (
    <div className="flex items-center justify-between gap-4 bg-indigo-500/10 px-5 py-2 text-sm text-indigo-200">
      <span>
        {error ? (
          <span className="text-red-300">Update failed: {error}</span>
        ) : busy ? (
          progress?.done ? (
            "Installing… the app will restart."
          ) : (
            `Downloading version ${update.latest}${pct != null ? ` · ${pct}%` : "…"}`
          )
        ) : (
          `Version ${update.latest} is available (you have ${update.current}).`
        )}
      </span>
      <span className="flex shrink-0 items-center gap-3">
        <button className="underline hover:text-white" onClick={() => api.openRelease(update.url)}>
          What's new
        </button>
        {!busy && (
          <>
            <button className="btn-primary px-2 py-0.5 text-xs" onClick={install}>
              Install &amp; restart
            </button>
            <button onClick={onClose} title="Later">
              ×
            </button>
          </>
        )}
      </span>
    </div>
  );
}
