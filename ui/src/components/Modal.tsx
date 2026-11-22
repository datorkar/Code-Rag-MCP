import { useEffect, type ReactNode } from "react";

export function Modal({
  title,
  onClose,
  children,
  footer,
  wide,
  locked,
  dismissOnBackdrop = true,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
  /** Ignore Esc / × / backdrop while an operation is running. */
  locked?: boolean;
  /** Close on a click outside the dialog (off for forms with unsaved input). */
  dismissOnBackdrop?: boolean;
}) {
  const close = () => !locked && onClose();
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && !locked && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose, locked]);

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
      onMouseDown={dismissOnBackdrop ? close : undefined}
    >
      <div
        className={`flex max-h-full w-full flex-col rounded-lg border border-zinc-800 bg-zinc-900 shadow-2xl ${wide ? "max-w-3xl" : "max-w-xl"}`}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <h2 className="text-base font-semibold text-zinc-100">{title}</h2>
          <button
            className="text-xl leading-none text-zinc-500 hover:text-zinc-200 disabled:opacity-30"
            onClick={close}
            disabled={locked}
          >
            ×
          </button>
        </div>
        <div className="overflow-y-auto px-5 py-4">{children}</div>
        {footer && <div className="flex justify-end gap-2 border-t border-zinc-800 px-5 py-3">{footer}</div>}
      </div>
    </div>
  );
}
