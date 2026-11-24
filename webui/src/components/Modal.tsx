import React, { useEffect, useId, useRef } from "react";
import { IconClose } from "./Icons";

/** The one dialog frame of the interface: a header (title, things next to it, a close button), a body that scrolls
 *  and a footer that stays in view with the actions.
 *
 *  Behaviour every dialog gets from here:
 *  - Esc closes it;
 *  - a press that begins AND ends on the dimmed background closes it (dragging a text selection out of an input
 *    and letting go over the background does not);
 *  - the keyboard focus moves into the dialog when it opens, Tab stays inside it, and the focus goes back to
 *    where it was when the dialog closes. */
export const Modal: React.FC<{
  title: React.ReactNode;
  /** next to the title: filters, a tag */
  titleExtra?: React.ReactNode;
  /** on the right of the header, before the close button */
  headerActions?: React.ReactNode;
  width: number;
  maxHeight?: string;
  onClose: () => void;
  /** given: the dialog is a form and this is its submit */
  onSubmit?: (e: React.FormEvent) => void;
  footer?: React.ReactNode;
  children: React.ReactNode;
}> = ({ title, titleExtra, headerActions, width, maxHeight, onClose, onSubmit, footer, children }) => {
  const boxRef = useRef<HTMLElement | null>(null);
  const pressedOnBackdrop = useRef(false);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  const titleId = useId();

  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    const box = boxRef.current;
    if (box) {
      // the field marked `data-initial-focus` gets the focus (its text selected, ready to be typed over);
      // a dialog without one takes the focus itself
      const wanted = box.querySelector<HTMLElement>("[data-initial-focus]");
      if (wanted) {
        wanted.focus();
        if (wanted instanceof HTMLInputElement) wanted.select();
      } else if (!box.contains(document.activeElement)) {
        box.focus();
      }
    }

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        closeRef.current();
        return;
      }
      if (e.key !== "Tab" || !boxRef.current) return;
      const items = Array.from(
        boxRef.current.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])'
        )
      ).filter((el) => el.offsetParent !== null);
      if (items.length === 0) {
        e.preventDefault();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (!active || !boxRef.current.contains(active)) {
        e.preventDefault();
        first.focus();
      } else if (e.shiftKey && (active === first || active === boxRef.current)) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      if (before && document.contains(before) && typeof before.focus === "function") before.focus();
    };
  }, []);

  const boxProps = {
    className: "modal-box",
    // never taller than the room the page leaves under the dialog's fixed top edge
    style: { width, maxHeight: maxHeight ? `min(${maxHeight}, 100%)` : undefined } as React.CSSProperties,
    role: "dialog",
    "aria-modal": true,
    "aria-labelledby": titleId,
    tabIndex: -1,
  };
  const inner = (
    <>
      <div className="modal-header">
        <div className="modal-header-main">
          <span className="modal-title" id={titleId}>
            {title}
          </span>
          {titleExtra}
        </div>
        <div className="modal-header-side">
          {headerActions}
          <button type="button" className="btn btn-sm btn-icon btn-ghost" onClick={onClose} aria-label="关闭" title="关闭">
            <IconClose />
          </button>
        </div>
      </div>
      <div className="modal-body">{children}</div>
      {footer ? <div className="modal-footer">{footer}</div> : null}
    </>
  );

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        pressedOnBackdrop.current = e.target === e.currentTarget;
      }}
      onMouseUp={(e) => {
        const close = pressedOnBackdrop.current && e.target === e.currentTarget;
        pressedOnBackdrop.current = false;
        if (close) onClose();
      }}
    >
      {onSubmit ? (
        <form {...boxProps} ref={boxRef as React.RefObject<HTMLFormElement>} onSubmit={onSubmit}>
          {inner}
        </form>
      ) : (
        <div {...boxProps} ref={boxRef as React.RefObject<HTMLDivElement>}>
          {inner}
        </div>
      )}
    </div>
  );
};
