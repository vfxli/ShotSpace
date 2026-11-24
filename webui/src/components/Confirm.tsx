import React, { useEffect, useState } from "react";
import { Modal } from "./Modal";

/** "Are you sure?" in the page's own dialog instead of the browser's box (which looks like no part of the page and
 *  names the server's address).  Same contract as `window.confirm`, as a promise: true = go on, false = leave it.
 *
 *      if (!(await confirmAction("…是否继续？"))) return;
 *
 *  Esc, 取消 and a click on the dimmed page answer no.  The focus starts on 取消, so a stray Enter removes nothing. */

interface Ask {
  message: string;
  confirmLabel: string;
  resolve: (ok: boolean) => void;
}

let present: ((ask: Ask) => void) | null = null;

export function confirmAction(message: string, confirmLabel = "确认"): Promise<boolean> {
  return new Promise((resolve) => {
    if (!present) {
      // no host on the page (cannot happen in the workbench): fall back to the browser's own box
      resolve(window.confirm(message));
      return;
    }
    present({ message, confirmLabel, resolve });
  });
}

export const ConfirmHost: React.FC = () => {
  const [ask, setAsk] = useState<Ask | null>(null);

  useEffect(() => {
    present = (next) =>
      setAsk((cur) => {
        if (cur) cur.resolve(false); // a question still open is answered "no" by the next one
        return next;
      });
    return () => {
      present = null;
    };
  }, []);

  if (!ask) return null;
  const answer = (ok: boolean) => {
    ask.resolve(ok);
    setAsk(null);
  };

  return (
    <Modal
      title="请确认"
      width={440}
      onClose={() => answer(false)}
      footer={
        <div className="modal-footer-actions">
          <button type="button" className="btn" data-initial-focus onClick={() => answer(false)}>
            取消
          </button>
          <button type="button" className="btn btn-danger is-confirm" onClick={() => answer(true)}>
            {ask.confirmLabel}
          </button>
        </div>
      }
    >
      <p className="confirm-text">{ask.message}</p>
    </Modal>
  );
};
