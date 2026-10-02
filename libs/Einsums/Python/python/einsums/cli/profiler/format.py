# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Small text renderers shared by the panels: bars, byte counts, sparklines, colors."""

from __future__ import annotations

import zlib

_EIGHTHS = " ▏▎▍▌▋▊▉█"
_SPARKS = "▁▂▃▄▅▆▇█"

# Warm flame-graph palette.
FLAME_COLORS = (
    "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026", "#fdcc8a",
    "#fc8d59", "#e34a33", "#b30000", "#fff7bc", "#fec44f", "#d95f0e",
)  # fmt: skip


def make_bar(pct: float, width: int = 10) -> str:
    """A bar of *width* cells filled to *pct* percent, in eighth-cell steps."""
    filled = max(0.0, min(100.0, pct)) / 100.0 * width
    full = int(filled)
    partial = _EIGHTHS[int((filled - full) * 8)] if full < width else ""
    return ("█" * full + partial).ljust(width)


def format_bytes(n: int) -> str:
    """``1.5M``-style byte count; empty for zero, so empty table cells stay blank."""
    if n == 0:
        return ""
    sign = "-" if n < 0 else ""
    n = abs(n)
    for shift, suffix in ((30, "G"), (20, "M"), (10, "K")):
        if n >= 1 << shift:
            return f"{sign}{n / (1 << shift):.1f}{suffix}"
    return f"{sign}{n}B"


def heat(text: str, pct: float) -> str:
    """Wrap *text* in Rich markup colored by how large a share *pct* is."""
    if pct > 50:
        return f"[bold red]{text}[/]"
    if pct > 20:
        return f"[yellow]{text}[/]"
    if pct > 5:
        return f"[cyan]{text}[/]"
    return text


def sparkline(values: list[float], width: int = 20) -> str:
    """The last *width* values as block characters, scaled to their own range."""
    recent = values[-width:]
    if not recent:
        return ""
    lo, hi = min(recent), max(recent)
    if hi <= lo:
        return _SPARKS[0] * len(recent)
    return "".join(_SPARKS[min(7, int((v - lo) / (hi - lo) * 7))] for v in recent)


def flame_color(name: str) -> str:
    """A color for *name* that is the same in every run (``hash()`` is salted per process)."""
    return FLAME_COLORS[zlib.crc32(name.encode()) % len(FLAME_COLORS)]
