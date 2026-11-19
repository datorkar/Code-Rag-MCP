import { useEffect, useState } from "react";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import { api } from "../api";
import type { ClaudeSnippets } from "../types";
import { Modal } from "./Modal";

export function Snippet({ title, text }: { title: string; text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="mb-4">
      <div className="mb-1 flex items-center justify-between">
        <span className="label mb-0">{title}</span>
        <button
          className="text-xs text-indigo-300 hover:text-indigo-200"
          onClick={async () => {
            await writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
        >
          {copied ? "Copied ✓" : "Copy"}
        </button>
      </div>
      <pre className="max-h-56 overflow-auto rounded-md border border-zinc-800 bg-zinc-950 p-3 font-mono text-xs whitespace-pre-wrap text-zinc-300">
        {text}
      </pre>
    </div>
  );
}

export function ClaudeModal({ onClose }: { onClose: () => void }) {
  const [s, setS] = useState<ClaudeSnippets | null>(null);
  const [result, setResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.claudeSnippets().then(setS).catch((e) => setResult({ ok: false, msg: String(e) }));
  }, []);

  const install = async () => {
    setBusy(true);
    setResult(null);
    try {
      setResult({ ok: true, msg: await api.registerClaudeMcp() });
    } catch (e) {
      setResult({ ok: false, msg: String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title="Install to Claude Code" onClose={onClose} wide>
      {s && (
        <>
          <div className="mb-4 flex items-center justify-between gap-4 rounded-md border border-zinc-800 bg-zinc-950/60 p-3">
            <div className="min-w-0 text-xs">
              <div className="text-zinc-400">MCP server executable</div>
              <div className={`truncate font-mono ${s.exe_exists ? "text-zinc-200" : "text-red-400"}`}>
                {s.exe_path} {!s.exe_exists && "(not found)"}
              </div>
            </div>
            <button className="btn-primary shrink-0" onClick={install} disabled={busy || !s.exe_exists}>
              {busy ? "Registering…" : "Register with claude CLI"}
            </button>
          </div>
          {result && (
            <p
              className={`mb-4 rounded px-3 py-2 text-xs whitespace-pre-wrap ${
                result.ok ? "bg-emerald-500/10 text-emerald-300" : "bg-red-500/10 text-red-300"
              }`}
            >
              {result.msg}
            </p>
          )}
          <Snippet title="Or run manually" text={s.command} />
          <Snippet title="Or add to .mcp.json / claude_desktop_config.json" text={s.json} />
          <Snippet title="CLAUDE.md snippet" text={s.claude_md} />
        </>
      )}
      {!s && result && <p className="text-sm text-red-400">{result.msg}</p>}
    </Modal>
  );
}
