import React from "react";

/** A slider (`<input type="range">`) that tells the stylesheet how far it is filled, as `--fill`.
 *  The stylesheet draws a slim track with the filled part in the primary colour; a browser's own slider cannot be
 *  given a filled part in every engine without this number.  Every other property goes to the input unchanged. */
export const Range: React.FC<Omit<React.InputHTMLAttributes<HTMLInputElement>, "type">> = ({ style, ...props }) => {
  const min = Number(props.min ?? 0);
  const max = Number(props.max ?? 100);
  const value = Number(props.value ?? min);
  const fill = max > min ? Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100)) : 0;
  return <input {...props} type="range" style={{ ...style, "--fill": `${fill}%` } as React.CSSProperties} />;
};
