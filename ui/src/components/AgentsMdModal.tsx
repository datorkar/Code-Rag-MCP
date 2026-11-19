import { useEffect, useState } from "react";
import { api } from "../api";
import type { IndexInfo } from "../types";
import { Modal } from "./Modal";
import { Snippet } from "./ClaudeModal";

export function AgentsMdModal({ ix, onClose }: { ix: IndexInfo; onClose: () => void }) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.agentsSnippet(ix.id).then(setText).catch((e) => setError(String(e)));
  }, [ix.id]);

  return (
    <Modal title={`AGENTS.md for “${ix.name}”`} onClose={onClose} wide>
      <p className="mb-3 text-sm text-zinc-400">
        Paste this into <code className="font-mono text-zinc-300">AGENTS.md</code> or{" "}
        <code className="font-mono text-zinc-300">CLAUDE.md</code> at the root of{" "}
        <code className="font-mono text-zinc-300">{ix.root_path}</code> so the agent knows to search this index.
      </p>
      {text && <Snippet title="Instructions" text={text} />}
      {error && <p className="text-sm text-red-400">{error}</p>}
    </Modal>
  );
}
