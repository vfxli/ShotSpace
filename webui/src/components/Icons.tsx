import React from "react";

/** Icons of the interface, drawn here (no icon font, nothing fetched): each takes the colour of the text around it.
 *  Symbols typed as characters (✕ ▾ ▶ ⏮ …) come from whatever font the machine has, with its own size and
 *  baseline - that is what made some buttons taller than their neighbours and their labels sit off centre. */

interface IconProps {
  size?: number;
  className?: string;
  title?: string;
}

function Svg({ size = 14, className, title, children, fill = false }: IconProps & { children: React.ReactNode; fill?: boolean }) {
  return (
    <svg
      className={`icon${className ? ` ${className}` : ""}`}
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill={fill ? "currentColor" : "none"}
      stroke={fill ? "none" : "currentColor"}
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden={title ? undefined : true}
      role={title ? "img" : undefined}
      focusable="false"
    >
      {title ? <title>{title}</title> : null}
      {children}
    </svg>
  );
}

export const IconClose: React.FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M4 4l8 8M12 4l-8 8" />
  </Svg>
);

export const IconCheck: React.FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M3.2 8.4l3.1 3.1 6.5-7" />
  </Svg>
);

export const IconChevronDown: React.FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M4 6l4 4 4-4" />
  </Svg>
);

export const IconChevronLeft: React.FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M10 4L6 8l4 4" />
  </Svg>
);

export const IconChevronRight: React.FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M6 4l4 4-4 4" />
  </Svg>
);

export const IconPlay: React.FC<IconProps> = (p) => (
  <Svg {...p} fill>
    <path d="M3.9 2.9v10.2a.6.6 0 0 0 .92.5l7.9-5.1a.6.6 0 0 0 0-1l-7.9-5.1a.6.6 0 0 0-.92.5Z" />
  </Svg>
);

export const IconPause: React.FC<IconProps> = (p) => (
  <Svg {...p} fill>
    <rect x="3.5" y="2.8" width="3.2" height="10.4" rx="0.8" />
    <rect x="9.3" y="2.8" width="3.2" height="10.4" rx="0.8" />
  </Svg>
);

export const IconSkipStart: React.FC<IconProps> = (p) => (
  <Svg {...p} fill>
    <rect x="2.6" y="3" width="1.9" height="10" rx="0.7" />
    <path d="M13.2 3.4v9.2a.55.55 0 0 1-.86.46L5.6 8.46a.55.55 0 0 1 0-.92l6.74-4.6a.55.55 0 0 1 .86.46Z" />
  </Svg>
);

export const IconSkipEnd: React.FC<IconProps> = (p) => (
  <Svg {...p} fill>
    <rect x="11.5" y="3" width="1.9" height="10" rx="0.7" />
    <path d="M2.8 3.4v9.2a.55.55 0 0 0 .86.46l6.74-4.6a.55.55 0 0 0 0-.92L3.66 2.94a.55.55 0 0 0-.86.46Z" />
  </Svg>
);

export const IconCamera: React.FC<IconProps> = (p) => (
  <Svg {...p}>
    <rect x="1.8" y="4.6" width="8.6" height="6.8" rx="1.4" />
    <path d="M10.4 7.2l3.8-2v5.6l-3.8-2" />
  </Svg>
);

export const IconFolder: React.FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M1.9 4.4a1.2 1.2 0 0 1 1.2-1.2h3l1.4 1.6h5.4a1.2 1.2 0 0 1 1.2 1.2v5.6a1.2 1.2 0 0 1-1.2 1.2H3.1a1.2 1.2 0 0 1-1.2-1.2V4.4Z" />
  </Svg>
);

export const IconCube: React.FC<IconProps> = (p) => (
  <Svg {...p}>
    <path d="M8 1.9l5.3 3v6.2L8 14.1l-5.3-3V4.9L8 1.9Z" />
    <path d="M2.9 5L8 8l5.1-3M8 8v6" />
  </Svg>
);

/** A cup of coffee with three wisps of steam; the wisps are moved by the stylesheet (`.coffee-note`). */
export const CoffeeCup: React.FC<{ size?: number }> = ({ size = 18 }) => (
  <svg
    className="coffee-cup"
    width={size}
    height={size}
    viewBox="0 0 20 20"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.4}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    focusable="false"
  >
    <path className="steam steam-1" d="M6.6 7.6c-1.1-1.2 1.1-2.4 0-3.7" />
    <path className="steam steam-2" d="M9.6 7.6c-1.1-1.2 1.1-2.4 0-3.7" />
    <path className="steam steam-3" d="M12.6 7.6c-1.1-1.2 1.1-2.4 0-3.7" />
    <path d="M3.4 9.6h12.4v2.9a4.4 4.4 0 0 1-4.4 4.4H7.8a4.4 4.4 0 0 1-4.4-4.4V9.6Z" />
    <path d="M15.8 10.7h.7a1.8 1.8 0 0 1 0 3.6h-1.1" />
  </svg>
);
