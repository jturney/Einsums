# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Text panels: node details, source, hotspots, hardware counters, log, disassembly, status."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import RichLog, Static

from ..analysis import aggregate_flat, derived_rates, find_node, summarize_counters
from ..format import format_bytes, make_bar
from ..model import LOG_LEVEL_NAMES, LogEntry, ProfileNode, ProfileSnapshot

LOG_LEVEL_STYLES = {0: "dim", 1: "cyan", 2: "green", 3: "yellow", 4: "red bold", 5: "red bold reverse", 7: "white bold"}


class TextPanel(VerticalScroll):
    """A scrolling panel of Rich markup that only redraws when its text changes."""

    def __init__(self, placeholder: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._placeholder = placeholder
        self._text = ""

    def compose(self) -> ComposeResult:
        yield Static(self._placeholder, classes="panel-text")

    def show(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self.query_one(".panel-text", Static).update(text)


def _annotation_text(value: Any) -> str:
    if not isinstance(value, dict):
        return escape(str(value))
    avg, lo, hi = value.get("avg", ""), value.get("min", ""), value.get("max", "")
    try:
        if lo != "" and float(lo) == float(hi):
            return escape(str(avg))
    except (TypeError, ValueError):
        pass
    return escape(f"avg:{avg}  min:{lo}  max:{hi}")


def node_details(
    node: ProfileNode,
    pct: float,
    *,
    baseline: ProfileSnapshot | None = None,
    history: str = "",
    bookmarked: bool = False,
) -> str:
    """The detail panel's markup for *node*; *pct* is its share of the thread's exclusive time."""
    lines = [
        f"[bold]{escape(node.name)}[/bold]{' ⚑' if bookmarked else ''}",
        f"  exclusive: {node.exclusive_ms:.3f} ms  ({pct:.1f}%)    calls: {node.call_count}    avg: {node.mean_ms:.3f} ms",
        f"  inclusive: {node.inclusive_ms:.3f} ms",
    ]
    if history:
        lines.append(f"  [bold]History:[/bold] {history}")
    if baseline is not None:
        base = find_node(baseline, node.name)
        if base is None:
            lines.append("  [bold]Delta:[/bold] (not in the baseline)")
        else:
            lines.append(
                f"  [bold]Delta vs baseline:[/bold]  Δcount: {node.call_count - base.call_count:+d}    "
                f"Δexclusive: {node.exclusive_ms - base.exclusive_ms:+.3f} ms"
            )
    if node.exclusive_min_ms or node.exclusive_max_ms or node.stddev_ms:
        lines.append(
            f"  min: {node.exclusive_min_ms:.3f} ms    max: {node.exclusive_max_ms:.3f} ms    stddev: {node.stddev_ms:.3f} ms"
        )
    where = "    ".join(part for part in (node.location, f"fn: {node.function}" if node.function else "") if part)
    if where:
        lines.append(f"  {escape(where)}")
    rates = derived_rates(node)
    if rates.gflops is not None:
        lines.append(f"  [bold]GFLOP/s:[/bold] {rates.gflops:.2f}")
    if rates.gb_per_s is not None:
        lines.append(f"  [bold]Bandwidth:[/bold] {rates.gb_per_s:.2f} GB/s")
    if rates.intensity is not None:
        lines.append(f"  [bold]Arithmetic intensity:[/bold] {rates.intensity:.2f} FLOP/byte")
    if node.mem_alloc_bytes or node.mem_free_bytes:
        lines.append(
            f"  [bold]Memory:[/bold]  alloc: {format_bytes(node.mem_alloc_bytes)} ({node.mem_alloc_count} calls)    "
            f"freed: {format_bytes(node.mem_free_bytes)} ({node.mem_free_count} calls)    "
            f"peak: {format_bytes(node.mem_peak_bytes)}    live: {format_bytes(node.mem_current_bytes)}"
        )
    if node.annotations:
        lines.append("  [bold]Annotations:[/bold]")
        lines += [f"    {escape(k)} = {_annotation_text(v)}" for k, v in node.annotations.items()]
    if node.counters:
        lines.append("  [bold]Counters:[/bold]")
        for name, data in node.counters.items():
            text = ", ".join(f"{k}={v}" for k, v in data.items()) if isinstance(data, dict) else str(data)
            lines.append(f"    {escape(name)}: {escape(text)}")
    if node.histogram and any(node.histogram.values()):
        lines.append("  [bold]Per-call histogram:[/bold]")
        peak = max(node.histogram.values())
        lines += [
            f"    {bucket:>8s} | {'█' * int(count / peak * 20)} {count}" for bucket, count in node.histogram.items() if count
        ]
    if node.children:
        lines.append("  [bold]Callees:[/bold]")
        for child in sorted(node.children, key=lambda c: c.inclusive_ms, reverse=True)[:10]:
            share = child.inclusive_ms / node.inclusive_ms * 100.0 if node.inclusive_ms > 0 else 0.0
            lines.append(f"    {share:5.1f}% {make_bar(share)} {escape(child.name)}  ({child.inclusive_ms:.3f} ms)")
    return "\n".join(lines)


def source_context(node: ProfileNode | None, context: int = 15) -> str:
    if node is None or not node.file or node.line <= 0:
        return "No source location for this zone"
    path = Path(node.file)
    try:
        text = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return escape(f"Cannot read {path}: {exc.strerror or exc}")
    first, last = max(1, node.line - context), min(len(text), node.line + context)
    lines = [f"[bold]{escape(str(path))}:{node.line}[/bold]  (fn: {escape(node.function or '?')})"]
    for number in range(first, last + 1):
        code = escape(text[number - 1].rstrip())
        lines.append(f"[bold yellow]>{number:5d}  {code}[/]" if number == node.line else f" {number:5d}  {code}")
    return "\n".join(lines)


def hotspots(roots: Iterable[ProfileNode], top: int = 10) -> str:
    flat = sorted(aggregate_flat(list(roots)), key=lambda n: n.exclusive_ms, reverse=True)
    if not flat:
        return "No data"
    total = sum(n.exclusive_ms for n in flat) or 1.0
    lines = [
        f"[bold]Top {top} by exclusive time, all threads[/bold]",
        f"  {'#':>2s}  {'%':>6s}  {'excl(ms)':>10s}  {'calls':>8s}  {'mean(ms)':>10s}  name",
    ]
    for i, n in enumerate(flat[:top], 1):
        lines.append(
            f"  {i:2d}  {n.exclusive_ms / total * 100:5.1f}%  {n.exclusive_ms:10.3f}  {n.call_count:8d}  "
            f"{n.mean_ms:10.3f}  {escape(n.name)}"
        )
    return "\n".join(lines)


def counter_analysis(node: ProfileNode | None) -> str:
    if node is None:
        return "Select a row to see its hardware counters"
    if not node.counters:
        return "[bold]Hardware counters[/bold]\n  (none recorded; they need Linux perf_event access)"
    c = summarize_counters(node.counters)
    if not c.cycles and not c.instructions:
        return "[bold]Hardware counters[/bold]\n  (no cycle or instruction counts)"
    label, color = c.classify()
    filled = int(min(c.ipc / 4.0, 1.0) * 20)
    lines = [
        f"[bold]Hardware counters[/bold]  [{color}]{label}[/]",
        f"  IPC:  [green]{'█' * filled}[/]{'░' * (20 - filled)}  {c.ipc:.2f}",
        f"  {'metric':<25s}  {'value':>14s}  {'per 1K insn':>12s}",
        f"  {'cycles':<25s}  {c.cycles:>14,d}",
        f"  {'instructions':<25s}  {c.instructions:>14,d}",
    ]
    if c.cache_misses:
        lines.append(f"  {'cache misses':<25s}  {c.cache_misses:>14,d}  {c.cache_misses_per_k:>12.2f}")
    if c.branch_misses:
        lines.append(f"  {'branch misses':<25s}  {c.branch_misses:>14,d}  {c.branch_misses_per_k:>12.2f}")
    return "\n".join(lines)


def highlight_asm(text: str) -> str:
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith((";", "#")):
            out.append(f"[dim]{escape(line)}[/dim]")
        elif line.rstrip().endswith(":") and not line.startswith(" "):
            out.append(f"[bold cyan]{escape(line)}[/bold cyan]")
        else:
            out.append(escape(line))
    return "\n".join(out)


class LogPanel(Widget):
    """``EINSUMS_LOG_*`` messages and ``println`` output streamed from the program."""

    LEVEL_CYCLE = (2, 3, 4, 0, 1)  # INFO, WARN, ERROR, TRACE, DEBUG

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_level = 2
        self._shown = 0  # entries written, counted against Session.log_total

    def compose(self) -> ComposeResult:
        yield RichLog(markup=True, wrap=True, max_lines=2000)

    def cycle_level(self) -> str:
        self.min_level = self.LEVEL_CYCLE[(self.LEVEL_CYCLE.index(self.min_level) + 1) % len(self.LEVEL_CYCLE)]
        self._shown = -1  # force a rewrite at the new level
        return LOG_LEVEL_NAMES[self.min_level]

    def show(self, entries: Iterable[LogEntry], total: int) -> None:
        if total == self._shown:
            return
        log = self.query_one(RichLog)
        entries = list(entries)
        new = total - self._shown
        if self._shown < 0 or new > len(entries):
            log.clear()
            new = len(entries)
        for entry in entries[len(entries) - new :]:
            if entry.level >= self.min_level:
                log.write(self.format_entry(entry))
        self._shown = total

    def reset(self) -> None:
        self._shown = -1

    @staticmethod
    def format_entry(e: LogEntry) -> str:
        style = LOG_LEVEL_STYLES.get(e.level, "")
        stamp = e.timestamp[11:23] if len(e.timestamp) >= 23 else e.timestamp
        where = f"  ({escape(e.file)}:{e.line})" if e.file else ""
        return f"[{style}]{stamp} [{LOG_LEVEL_NAMES.get(e.level, '?'):>6}][/] {escape(e.message)}{where}"


class StatusBar(Static):
    text = reactive("Starting")

    def render(self) -> str:
        return self.text


class ResizeHandle(Widget):
    """Drag to resize the panel named by *target_id*, which sits below the handle."""

    def __init__(self, target_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._target_id = target_id
        self._drag: tuple[int, int] | None = None  # (start y, start height)

    def render(self) -> str:
        return "─── drag to resize ───"

    def on_mouse_down(self, event: Any) -> None:
        target = self.screen.query_one(f"#{self._target_id}")
        self._drag = (event.screen_y, target.size.height)
        self.add_class("-dragging")
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: Any) -> None:
        if self._drag is not None:
            y0, h0 = self._drag
            self.screen.query_one(f"#{self._target_id}").styles.height = max(3, h0 - (event.screen_y - y0))
            event.stop()

    def on_mouse_up(self, event: Any) -> None:
        if self._drag is not None:
            self._drag = None
            self.remove_class("-dragging")
            self.release_mouse()
            event.stop()
