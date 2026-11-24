"""Cross-format 3D object & joint identifier rules (pure standard library)."""

from __future__ import annotations

import re
from collections.abc import Container

ORIGINAL = "lab2shot:name"  # Compatible with DCC/USD/Alembic metadata convention

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _spelled(c: str) -> str:
    return f"_u{ord(c):04x}" if ord(c) <= 0xFFFF else f"_U{ord(c):08x}"


def identifier(name: str) -> str:
    """Convert any text into a valid USD/Alembic identifier ([A-Za-z_][A-Za-z0-9_]*)."""
    if _IDENTIFIER.fullmatch(name):
        return name
    out = "".join(c if c.isascii() and c.isalnum() else "_" if c.isascii() else _spelled(c) for c in name)
    return f"_{out}" if out[:1].isdigit() else out or "_"


def unique(base: str, taken: Container[str]) -> str:
    """Return `base`, or the first `base_2`, `base_3`, ... not in `taken`."""
    out, n = base, 1
    while out in taken:
        n += 1
        out = f"{base}_{n}"
    return out


def sibling_unique(names: list[str], parents) -> list[str]:
    """Ensure every joint name is unique among siblings sharing the same parent."""
    taken: dict[int, set[str]] = {}
    out = []
    for name, parent in zip(names, parents):
        seen = taken.setdefault(int(parent), set())
        u = unique(str(name), seen)
        seen.add(u)
        out.append(u)
    return out


def join_path(parts) -> str:
    """Join hierarchy names into a `/`-prefixed path, escaping internal `/` and `\\`."""
    return "/" + "/".join(str(p).replace("\\", "\\\\").replace("/", "\\/") for p in parts)


def split_path(path: str) -> list[str]:
    """Inverse of `join_path`, returning top-down non-empty segment names."""
    parts, cur, escaped = [], "", False
    for c in str(path):
        if escaped:
            cur, escaped = cur + c, False
        elif c == "\\":
            escaped = True
        elif c == "/":
            parts.append(cur)
            cur = ""
        else:
            cur += c
    parts.append(cur)
    return [p for p in parts if p]
