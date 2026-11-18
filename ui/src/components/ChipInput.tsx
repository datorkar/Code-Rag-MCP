import { useState, type KeyboardEvent } from "react";

interface Props {
  values: string[];
  onChange: (v: string[]) => void;
  placeholder?: string;
  presets?: Record<string, string[]>;
  normalize?: (s: string) => string;
}

export function ChipInput({ values, onChange, placeholder, presets, normalize = (s) => s }: Props) {
  const [draft, setDraft] = useState("");

  const add = (raw: string) => {
    const items = raw
      .split(/[,\s;]+/)
      .map((s) => normalize(s.trim()))
      .filter(Boolean)
      .filter((s) => !values.includes(s));
    if (items.length) onChange([...values, ...items]);
    setDraft("");
  };

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === "," || e.key === " ") {
      e.preventDefault();
      add(draft);
    } else if (e.key === "Backspace" && !draft && values.length) {
      onChange(values.slice(0, -1));
    }
  };

  return (
    <div>
      <div className="flex min-h-9 flex-wrap items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900 px-1.5 py-1 focus-within:border-indigo-500">
        {values.map((v) => (
          <span
            key={v}
            className="inline-flex items-center gap-1 rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-xs text-zinc-200"
          >
            {v}
            <button
              type="button"
              className="text-zinc-500 hover:text-red-400"
              onClick={() => onChange(values.filter((x) => x !== v))}
              aria-label={`remove ${v}`}
            >
              ×
            </button>
          </span>
        ))}
        <input
          className="min-w-24 flex-1 bg-transparent px-1 text-sm outline-none placeholder:text-zinc-600"
          value={draft}
          placeholder={values.length ? "" : placeholder}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKey}
          onBlur={() => draft && add(draft)}
        />
      </div>
      {presets && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {Object.entries(presets).map(([name, items]) => (
            <button
              key={name}
              type="button"
              className="rounded border border-zinc-700 px-1.5 py-0.5 text-[11px] text-zinc-400 hover:border-indigo-500 hover:text-indigo-300"
              onClick={() =>
                onChange([...values, ...items.map(normalize).filter((i) => !values.includes(i))])
              }
            >
              + {name}
            </button>
          ))}
          {values.length > 0 && (
            <button
              type="button"
              className="px-1.5 py-0.5 text-[11px] text-zinc-500 hover:text-red-400"
              onClick={() => onChange([])}
            >
              clear
            </button>
          )}
        </div>
      )}
    </div>
  );
}
