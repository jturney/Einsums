#!/usr/bin/env python3
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Textual TUI for viewing live Einsums profiling data.

Connects to the built-in profiler TCP server and displays a real-time
hierarchical call tree with timing, counts, and annotations.

Usage:
    python profile_viewer.py                    # default localhost:19216 + mDNS
    python profile_viewer.py --port 19217       # custom port
    python profile_viewer.py --port 19216 19217 # multiple ports
    python profile_viewer.py --host 10.0.0.5    # remote host
    python profile_viewer.py --connect h:19216  # explicit host:port endpoint
    python profile_viewer.py --no-mdns          # disable auto-discovery
    python profile_viewer.py --record out.jsonl # record snapshots to file
    python profile_viewer.py --replay out.jsonl # replay a recorded session
    python profile_viewer.py --load session.json # load a saved session
    python profile_viewer.py --load s1.json s2.json # load multiple saved sessions

Key bindings (see command palette for full list):
    /       Filter by name (regex supported)
    b       Toggle bookmark on selected node
    B       Jump to next bookmarked node
    u       Toggle bottom-up view
    H       Toggle top-N hotspots panel
    V       Toggle source code panel
    C       Side-by-side session comparison
    G       Toggle thread Gantt chart
    M       Toggle micro-architecture stall analysis panel
    A       Toggle Assembly/IR panel (MLIR IR | LLVM IR | Assembly)
    S       Save current session to JSON
    Ctrl+S  Save all sessions to JSON
    T       Cycle color theme (none / heat)
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import io
import json
import math
import os.path
import re
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message as TextualMessage
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)

try:
    from textual_plotext import PlotextPlot
    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False

try:
    from zeroconf import ServiceBrowser, ServiceStateChange, Zeroconf
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ProfileNode:
    name: str = ""
    call_count: int = 0
    exclusive_ms: float = 0.0
    inclusive_ms: float = 0.0
    exclusive_min_ms: float = 0.0
    exclusive_max_ms: float = 0.0
    stddev_ms: float = 0.0
    file: str = ""
    line: int = 0
    function: str = ""
    annotations: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, dict] = field(default_factory=dict)
    # Memory tracking
    mem_alloc_count: int = 0
    mem_free_count: int = 0
    mem_alloc_bytes: int = 0
    mem_free_bytes: int = 0
    mem_current_bytes: int = 0
    mem_peak_bytes: int = 0
    histogram: dict[str, int] = field(default_factory=dict)  # bucket_label -> count
    children: list[ProfileNode] = field(default_factory=list)


@dataclass
class ThreadData:
    thread_id: str = ""
    thread_name: str = ""
    children: list[ProfileNode] = field(default_factory=list)


@dataclass
class ProfileSnapshot:
    seq: int = 0
    dropped: int = 0
    threads: dict[str, ThreadData] = field(default_factory=dict)


@dataclass
class ProfileMeta:
    pid: int = 0
    hostname: str = ""
    executable: str = ""
    start_time: str = ""
    counters: list[str] = field(default_factory=list)
    executable_path: str = ""


@dataclass
class LogEntry:
    level: int = 2        # 0=trace..5=critical
    timestamp: str = ""
    file: str = ""
    line: int = 0
    function: str = ""
    message: str = ""


LOG_LEVEL_NAMES = {0: "TRACE", 1: "DEBUG", 2: "INFO", 3: "WARN", 4: "ERROR", 5: "CRITICAL", 7: "OUTPUT"}
LOG_LEVEL_STYLES = {0: "dim", 1: "cyan", 2: "green", 3: "yellow", 4: "red bold", 5: "red bold reverse", 7: "white bold"}


def parse_node(data: dict[str, Any]) -> ProfileNode:
    """Parse a node dict from JSON into a ProfileNode."""
    # Handle annotations: may be strings or objects with avg/min/max
    raw_annotations = data.get("annotations", {})
    annotations: dict[str, Any] = {}
    for k, v in raw_annotations.items():
        if isinstance(v, dict):
            annotations[k] = v
        else:
            annotations[k] = v

    mem = data.get("memory", {})
    # Parse histogram: {"histogram": {"1us": 5, "2us": 3, ...}}
    raw_hist = data.get("histogram", {})
    histogram: dict[str, int] = {}
    if isinstance(raw_hist, dict):
        for k, v in raw_hist.items():
            try:
                histogram[k] = int(v)
            except (ValueError, TypeError):
                pass
    node = ProfileNode(
        name=data.get("name", ""),
        call_count=data.get("call_count", 0),
        exclusive_ms=data.get("exclusive_ms", 0.0),
        inclusive_ms=data.get("inclusive_ms", 0.0),
        exclusive_min_ms=data.get("exclusive_min_ms", 0.0),
        exclusive_max_ms=data.get("exclusive_max_ms", 0.0),
        stddev_ms=data.get("stddev_ms", 0.0),
        file=data.get("file", ""),
        line=data.get("line", 0),
        function=data.get("function", ""),
        annotations=annotations,
        counters=data.get("counters", {}),
        mem_alloc_count=mem.get("alloc_count", 0),
        mem_free_count=mem.get("free_count", 0),
        mem_alloc_bytes=mem.get("alloc_bytes", 0),
        mem_free_bytes=mem.get("free_bytes", 0),
        mem_current_bytes=mem.get("current_bytes", 0),
        mem_peak_bytes=mem.get("peak_bytes", 0),
        histogram=histogram,
    )
    for child_data in data.get("children", []):
        node.children.append(parse_node(child_data))
    return node


def parse_snapshot(data: dict[str, Any]) -> ProfileSnapshot:
    """Parse a snapshot message."""
    snap = ProfileSnapshot(
        seq=data.get("seq", 0),
        dropped=data.get("dropped", 0),
    )
    threads = data.get("threads", {})
    for tid, tdata in threads.items():
        td = ThreadData(
            thread_id=tid,
            thread_name=tdata.get("name", ""),
        )
        for child_data in tdata.get("children", []):
            td.children.append(parse_node(child_data))
        snap.threads[tid] = td
    return snap


def sum_exclusive(nodes: list[ProfileNode]) -> float:
    """Recursively sum exclusive_ms across all nodes."""
    total = 0.0
    for n in nodes:
        total += n.exclusive_ms + sum_exclusive(n.children)
    return total


# ---------------------------------------------------------------------------
# Tree flattening for DataTable display
# ---------------------------------------------------------------------------

@dataclass
class FlatRow:
    depth: int
    node: ProfileNode
    collapsed: bool  # True if this node's children are hidden
    parent_inclusive_ms: float  # Parent's inclusive_ms (0.0 for root nodes)


def _name_matches(name: str, pattern: str) -> bool:
    """Check if name matches pattern. Tries regex first, falls back to substring."""
    if not pattern:
        return True
    try:
        return re.search(pattern, name, re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in name.lower()


def flatten_tree(
    nodes: list[ProfileNode],
    depth: int = 0,
    name_filter: str = "",
    collapsed_paths: set[str] | None = None,
    path_prefix: str = "",
    parent_inclusive_ms: float = 0.0,
) -> list[tuple[FlatRow, str]]:
    """Flatten a tree of ProfileNodes into rows for display.

    Returns list of (FlatRow, path_key) tuples.
    """
    if collapsed_paths is None:
        collapsed_paths = set()
    rows: list[tuple[FlatRow, str]] = []
    for node in nodes:
        matches = not name_filter or _name_matches(node.name, name_filter)
        path_key = f"{path_prefix}/{node.name}" if path_prefix else node.name
        is_collapsed = path_key in collapsed_paths
        has_children = bool(node.children)

        child_rows: list[tuple[FlatRow, str]] = []
        if not is_collapsed:
            child_rows = flatten_tree(
                node.children, depth + 1, name_filter, collapsed_paths, path_key,
                parent_inclusive_ms=node.inclusive_ms,
            )

        if matches or child_rows:
            rows.append((FlatRow(depth=depth, node=node, collapsed=is_collapsed and has_children,
                                 parent_inclusive_ms=parent_inclusive_ms), path_key))
            rows.extend(child_rows)
    return rows


def _make_bar(pct: float, width: int = 10) -> str:
    """Build a visual bar string proportional to pct (0-100), max `width` chars."""
    # Each character represents (100/width)% = 10% at width=10
    # Use 8 sub-block levels:
    blocks = " \u258f\u258e\u258d\u258c\u258b\u258a\u2589\u2588"
    clamped = max(0.0, min(100.0, pct))
    # How many full chars and fractional
    filled_f = clamped / 100.0 * width
    full = int(filled_f)
    frac = filled_f - full
    bar_chars = []
    for i in range(width):
        if i < full:
            bar_chars.append("\u2588")
        elif i == full:
            idx = int(frac * 8)
            if idx == 0:
                bar_chars.append(" ")
            else:
                bar_chars.append(blocks[idx])
        else:
            bar_chars.append(" ")
    return "".join(bar_chars)


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n == 0:
        return ""
    abs_n = abs(n)
    sign = "-" if n < 0 else ""
    if abs_n >= 1 << 30:
        return f"{sign}{abs_n / (1 << 30):.1f}G"
    if abs_n >= 1 << 20:
        return f"{sign}{abs_n / (1 << 20):.1f}M"
    if abs_n >= 1 << 10:
        return f"{sign}{abs_n / (1 << 10):.1f}K"
    return f"{sign}{abs_n}B"


def _color_wrap(text: str, pct: float) -> str:
    """Wrap text in Rich color markup based on percentage."""
    if pct > 50:
        return f"[bold red]{text}[/]"
    elif pct > 20:
        return f"[yellow]{text}[/]"
    elif pct > 5:
        return f"[cyan]{text}[/]"
    return text


def _is_node_anomaly(node: ProfileNode) -> bool:
    """Check if a node's per-call time deviates significantly from its mean."""
    if node.stddev_ms <= 0 or node.call_count <= 0:
        return False
    mean_ms = node.exclusive_ms / node.call_count
    if node.exclusive_max_ms > 0 and node.exclusive_max_ms > mean_ms + 2 * node.stddev_ms:
        return True
    if node.exclusive_min_ms > 0 and mean_ms > node.exclusive_min_ms + 2 * node.stddev_ms:
        return True
    return False


def _make_sparkline(values: list[float], width: int = 20) -> str:
    """Build a sparkline string from a list of float values using Unicode block chars."""
    if not values:
        return ""
    bars = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    lo = min(values)
    hi = max(values)
    rng = hi - lo
    if rng <= 0:
        return bars[0] * min(len(values), width)
    # Take last `width` values
    recent = values[-width:]
    chars = []
    for v in recent:
        idx = int((v - lo) / rng * 7)
        idx = max(0, min(7, idx))
        chars.append(bars[idx])
    return "".join(chars)


def _compute_hotpath(nodes: list[ProfileNode], path_prefix: str = "") -> set[str]:
    """Compute the set of path_keys on the hot path (most expensive exclusive_ms at each level)."""
    if not nodes:
        return set()
    hottest = max(nodes, key=lambda n: n.exclusive_ms)
    path_key = f"{path_prefix}/{hottest.name}" if path_prefix else hottest.name
    result = {path_key}
    if hottest.children:
        result |= _compute_hotpath(hottest.children, path_key)
    return result


# ---------------------------------------------------------------------------
# Network client
# ---------------------------------------------------------------------------

class ProfileClient:
    """Non-blocking TCP client for the profiler server."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._buffer = ""
        self.connected = False
        self.meta: ProfileMeta | None = None
        self.latest_snapshot: ProfileSnapshot | None = None
        self.latest_timeline: list[TimelineEvent] = []
        self.seq = 0
        self._reconnect_delay = 1.0
        self.retry_event = asyncio.Event()
        self._request_counter = 0
        self._pending_responses: dict[str, asyncio.Future] = {}
        self.log_entries: deque[LogEntry] = deque(maxlen=2000)

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    async def connect(self) -> bool:
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=2.0
            )
            self.connected = True
            self._reconnect_delay = 1.0
            return True
        except (OSError, asyncio.TimeoutError):
            self.connected = False
            return False

    async def disconnect(self) -> None:
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        self.connected = False

    async def read_messages(self) -> list[dict[str, Any]]:
        """Read and parse available JSON Lines messages. Non-blocking."""
        if not self._reader:
            return []

        messages = []
        try:
            data = await asyncio.wait_for(self._reader.read(65536), timeout=0.1)
            if not data:
                self.connected = False
                return []
            self._buffer += data.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            pass
        except Exception:
            self.connected = False
            return []

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                messages.append(msg)
            except json.JSONDecodeError:
                pass

        return messages

    async def send_request(self, method: str, params: dict | None = None, timeout: float = 5.0) -> dict[str, Any]:
        """Send a request to the server and wait for the response."""
        if not self._writer or not self.connected:
            return {"error": "not connected"}
        self._request_counter += 1
        req_id = f"req_{self._request_counter}"
        request = {"type": "request", "id": req_id, "method": method, "params": params or {}}
        line = json.dumps(request) + "\n"
        try:
            loop = asyncio.get_event_loop()
            future: asyncio.Future = loop.create_future()
            self._pending_responses[req_id] = future
            self._writer.write(line.encode("utf-8"))
            await self._writer.drain()
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._pending_responses.pop(req_id, None)
            return {"error": "timeout"}
        except Exception as exc:
            self._pending_responses.pop(req_id, None)
            return {"error": str(exc)}

    def process_messages(self, messages: list[dict[str, Any]]) -> bool:
        """Process messages and update state. Returns True if snapshot updated."""
        updated = False
        for msg in messages:
            msg_type = msg.get("type", "")
            if msg_type == "meta":
                self.meta = ProfileMeta(
                    pid=msg.get("pid", 0),
                    hostname=msg.get("hostname", ""),
                    executable=msg.get("executable", ""),
                    start_time=msg.get("start_time", ""),
                    counters=msg.get("counters", []),
                    executable_path=msg.get("executable_path", ""),
                )
            elif msg_type == "snapshot":
                self.latest_snapshot = parse_snapshot(msg)
                self.seq = self.latest_snapshot.seq
                updated = True
            elif msg_type == "timeline":
                events = []
                for edata in msg.get("events", []):
                    events.append(TimelineEvent(
                        thread_id=str(edata.get("tid", "")),
                        name=edata.get("name", ""),
                        start_ms=edata.get("start_ms", 0.0),
                        end_ms=edata.get("end_ms", 0.0),
                    ))
                self.latest_timeline = events
                updated = True
            elif msg_type == "log":
                self.log_entries.append(LogEntry(
                    level=msg.get("level", 2),
                    timestamp=msg.get("timestamp", ""),
                    file=msg.get("file", ""),
                    line=msg.get("line", 0),
                    function=msg.get("function", ""),
                    message=msg.get("message", ""),
                ))
                updated = True
            elif msg_type == "output":
                self.log_entries.append(LogEntry(
                    level=7,  # OUTPUT level
                    timestamp=msg.get("timestamp", ""),
                    message=msg.get("message", ""),
                ))
                updated = True
            elif msg_type == "response":
                req_id = msg.get("id", "")
                future = self._pending_responses.pop(req_id, None)
                if future and not future.done():
                    future.set_result(msg.get("data", {}))
        return updated


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class StatusBar(Static):
    """Status bar showing connection info."""

    status_text = reactive("Disconnected")

    def render(self) -> str:
        return self.status_text


class ProfileDataTable(DataTable):
    """DataTable subclass that forwards space/enter/click to toggle tree nodes."""

    from textual.message import Message as _Message

    class ToggleRequest(_Message):
        """Posted when user wants to toggle a row."""
        def __init__(self, row: int) -> None:
            super().__init__()
            self.row = row

    def _on_key(self, event) -> None:
        if event.key in ("space", "enter"):
            event.prevent_default()
            event.stop()
            self.post_message(self.ToggleRequest(self.cursor_row))
            return
        super()._on_key(event)

    def on_click(self, event) -> None:
        # DataTable handles click internally to move cursor.
        # Post toggle after cursor has been repositioned.
        self.call_later(lambda: self.post_message(self.ToggleRequest(self.cursor_row)))


class ResizeHandle(Widget):
    """Draggable handle for resizing the panel below it."""

    DEFAULT_CSS = """
    ResizeHandle {
        height: 1;
        width: 1fr;
        background: $surface;
        color: $text-muted;
        content-align: center middle;
    }
    ResizeHandle:hover {
        background: $accent;
        color: $text;
    }
    ResizeHandle.-dragging {
        background: $accent;
        color: $text;
    }
    """

    def __init__(self, target_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._target_id = target_id
        self._dragging = False
        self._drag_start_y = 0
        self._start_height = 0

    def render(self) -> str:
        return "─── drag to resize ───"

    def on_mouse_down(self, event) -> None:
        self._dragging = True
        self._drag_start_y = event.screen_y
        try:
            target = self.app.query_one(f"#{self._target_id}")
            self._start_height = target.size.height
        except Exception:
            self._start_height = 20
        self.add_class("-dragging")
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event) -> None:
        if not self._dragging:
            return
        delta = event.screen_y - self._drag_start_y
        new_height = max(3, self._start_height - delta)
        try:
            target = self.app.query_one(f"#{self._target_id}")
            target.styles.height = new_height
        except Exception:
            pass
        event.stop()

    def on_mouse_up(self, event) -> None:
        if self._dragging:
            self._dragging = False
            self.remove_class("-dragging")
            self.release_mouse()
            event.stop()


class DetailPanel(Widget):
    """Shows full details and annotations for the selected node."""

    can_focus = True

    DEFAULT_CSS = """
    DetailPanel {
        height: 20;
        min-height: 5;
        max-height: 30;
        background: $surface;
        border-top: solid $accent;
    }
    DetailPanel VerticalScroll {
        height: auto;
        max-height: 100%;
    }
    DetailPanel .detail-content {
        padding: 0 1;
    }
    """

    _last_content: str = ""

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("Select a row to see details", classes="detail-content", id="detail-content")

    def update_node(
        self,
        node: ProfileNode | None,
        pct: float = 0.0,
        baseline_snap: ProfileSnapshot | None = None,
        sparkline: str = "",
        is_bookmarked: bool = False,
    ) -> None:
        if node is None:
            content = "Select a row to see details"
        else:
            mean_ms = node.exclusive_ms / node.call_count if node.call_count > 0 else 0.0

            bm_str = " \u2691" if is_bookmarked else ""
            lines = [
                f"[bold]{node.name}[/bold]{bm_str}",
                f"  exclusive: {node.exclusive_ms:.3f} ms  ({pct:.1f}%)    "
                f"calls: {node.call_count}    "
                f"avg: {mean_ms:.3f} ms",
                f"  inclusive: {node.inclusive_ms:.3f} ms",
            ]

            # Sparkline of recent exclusive_ms values
            if sparkline:
                lines.append(f"  [bold]History:[/bold] {sparkline}")

            # Diff/delta info
            if baseline_snap is not None:
                baseline_node = _find_node_in_snapshot(baseline_snap, node.name)
                if baseline_node is not None:
                    delta_count = node.call_count - baseline_node.call_count
                    delta_excl = node.exclusive_ms - baseline_node.exclusive_ms
                    lines.append(
                        f"  [bold]Delta (vs baseline):[/bold]  "
                        f"\u0394count: {delta_count:+d}    "
                        f"\u0394exclusive: {delta_excl:+.3f} ms"
                    )
                else:
                    lines.append("  [bold]Delta:[/bold] (no baseline match)")

            # Min/max/stddev
            if node.exclusive_min_ms > 0 or node.exclusive_max_ms > 0 or node.stddev_ms > 0:
                lines.append(
                    f"  min: {node.exclusive_min_ms:.3f} ms    "
                    f"max: {node.exclusive_max_ms:.3f} ms    "
                    f"stddev: {node.stddev_ms:.3f} ms"
                )

            # File/line/function
            if node.file or node.function:
                loc_parts = []
                if node.file:
                    loc = node.file
                    if node.line > 0:
                        loc += f":{node.line}"
                    loc_parts.append(loc)
                if node.function:
                    loc_parts.append(f"fn: {node.function}")
                lines.append(f"  {'    '.join(loc_parts)}")

            # Derived metrics from annotations
            flops = _extract_numeric_annotation(node.annotations, "flops")
            bytes_read = _extract_numeric_annotation(node.annotations, "bytes_read")
            bytes_written = _extract_numeric_annotation(node.annotations, "bytes_written")

            if flops is not None and node.exclusive_ms > 0:
                gflops = flops / (node.exclusive_ms / 1000.0) / 1e9
                lines.append(f"  [bold]GFLOPS:[/bold] {gflops:.2f}")

            if (bytes_read is not None or bytes_written is not None) and node.exclusive_ms > 0:
                total_bytes = (bytes_read or 0) + (bytes_written or 0)
                gb_per_s = total_bytes / (node.exclusive_ms / 1000.0) / 1e9
                lines.append(f"  [bold]Bandwidth:[/bold] {gb_per_s:.2f} GB/s")
                if flops is not None and total_bytes > 0:
                    ai = flops / total_bytes
                    lines.append(f"  [bold]Arithmetic Intensity:[/bold] {ai:.2f} FLOP/byte")

            # Memory tracking
            if node.mem_alloc_bytes > 0 or node.mem_free_bytes > 0:
                lines.append(
                    f"  [bold]Memory:[/bold]  "
                    f"alloc: {_format_bytes(node.mem_alloc_bytes)} ({node.mem_alloc_count} calls)    "
                    f"freed: {_format_bytes(node.mem_free_bytes)} ({node.mem_free_count} calls)    "
                    f"peak: {_format_bytes(node.mem_peak_bytes)}    "
                    f"live: {_format_bytes(node.mem_current_bytes)}"
                )

            # Annotations
            if node.annotations:
                lines.append("  [bold]Annotations:[/bold]")
                for k, v in node.annotations.items():
                    if isinstance(v, dict):
                        avg = v.get("avg", "")
                        vmin = v.get("min", "")
                        vmax = v.get("max", "")
                        # Show plain value when min==max (single sample)
                        try:
                            if vmin != "" and vmax != "" and float(vmin) == float(vmax):
                                lines.append(f"    {k} = {avg}")
                            else:
                                lines.append(f"    {k} = avg:{avg}  min:{vmin}  max:{vmax}")
                        except (ValueError, TypeError):
                            lines.append(f"    {k} = avg:{avg}  min:{vmin}  max:{vmax}")
                    else:
                        lines.append(f"    {k} = {v}")

            # Counters
            if node.counters:
                lines.append("  [bold]Counters:[/bold]")
                for cname, cdata in node.counters.items():
                    if isinstance(cdata, dict):
                        parts = [f"{ck}={cv}" for ck, cv in cdata.items()]
                        lines.append(f"    {cname}: {', '.join(parts)}")
                    else:
                        lines.append(f"    {cname}: {cdata}")

            # Histogram (from C++ per-call log2 histogram)
            histogram = getattr(node, "histogram", None)
            if histogram and any(v > 0 for v in histogram.values()):
                lines.append("  [bold]Per-call histogram:[/bold]")
                max_count = max(histogram.values())
                for bucket_label, count in histogram.items():
                    if count > 0:
                        bar_w = int(count / max_count * 20) if max_count > 0 else 0
                        bar = "\u2588" * bar_w
                        lines.append(f"    {bucket_label:>8s} | {bar} {count}")

            # Callee breakdown (top 10 children by inclusive_ms)
            if node.children:
                sorted_children = sorted(node.children, key=lambda c: c.inclusive_ms, reverse=True)[:10]
                lines.append("  [bold]Callee breakdown:[/bold]")
                for child in sorted_children:
                    if node.inclusive_ms > 0:
                        child_pct = child.inclusive_ms / node.inclusive_ms * 100.0
                    else:
                        child_pct = 0.0
                    bar = _make_bar(child_pct, width=10)
                    lines.append(f"    {child_pct:5.1f}% {bar} {child.name}  ({child.inclusive_ms:.3f} ms)")

            content = "\n".join(lines)

        if content != self._last_content:
            self._last_content = content
            try:
                self.query_one("#detail-content", Static).update(content)
            except Exception:
                pass


def _extract_numeric_annotation(annotations: dict[str, Any], key: str) -> float | None:
    """Extract a numeric value from annotations. Handles both plain values and avg/min/max dicts."""
    val = annotations.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        # Use avg if available
        avg = val.get("avg")
        if avg is not None:
            try:
                return float(avg)
            except (ValueError, TypeError):
                return None
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _find_node_in_snapshot(snap: ProfileSnapshot, name: str) -> ProfileNode | None:
    """Find a node by name in a snapshot (searches all threads, depth-first)."""
    for tdata in snap.threads.values():
        result = _find_node_by_name(tdata.children, name)
        if result is not None:
            return result
    return None


def _find_node_by_name(nodes: list[ProfileNode], name: str) -> ProfileNode | None:
    """Depth-first search for a node by name."""
    for n in nodes:
        if n.name == name:
            return n
        found = _find_node_by_name(n.children, name)
        if found is not None:
            return found
    return None


def _sum_mem_current(nodes: list[ProfileNode]) -> int:
    """Recursively sum mem_current_bytes across all nodes."""
    total = 0
    for n in nodes:
        total += n.mem_current_bytes + _sum_mem_current(n.children)
    return total


def _sum_mem_peak(nodes: list[ProfileNode]) -> int:
    """Recursively find max mem_peak_bytes across all nodes."""
    peak = 0
    for n in nodes:
        peak = max(peak, n.mem_peak_bytes, _sum_mem_peak(n.children))
    return peak


def _sum_mem_alloc(nodes: list[ProfileNode]) -> int:
    """Recursively sum total allocated bytes across all nodes."""
    total = 0
    for n in nodes:
        total += n.mem_alloc_bytes + _sum_mem_alloc(n.children)
    return total


def _collect_all_nodes(nodes: list[ProfileNode], depth: int = 0) -> list[tuple[ProfileNode, int]]:
    """Recursively collect all nodes with their depth for CSV export."""
    result: list[tuple[ProfileNode, int]] = []
    for n in nodes:
        result.append((n, depth))
        result.extend(_collect_all_nodes(n.children, depth + 1))
    return result


def _collect_node_exclusive_ms(nodes: list[ProfileNode], out: dict[str, float]) -> None:
    """Walk all nodes and collect exclusive_ms keyed by name."""
    for n in nodes:
        out[n.name] = n.exclusive_ms
        _collect_node_exclusive_ms(n.children, out)


def _aggregate_flat(nodes: list[ProfileNode]) -> list[ProfileNode]:
    """Aggregate all nodes by name into a flat list.

    For each unique name: sum exclusive_ms, call_count, mem bytes;
    take max of inclusive_ms, peak; merge annotations (last wins).
    """
    agg: dict[str, ProfileNode] = {}

    def _walk(node_list: list[ProfileNode]) -> None:
        for n in node_list:
            if n.name in agg:
                a = agg[n.name]
                a.exclusive_ms += n.exclusive_ms
                a.inclusive_ms = max(a.inclusive_ms, n.inclusive_ms)
                a.call_count += n.call_count
                a.exclusive_min_ms = min(a.exclusive_min_ms, n.exclusive_min_ms) if a.exclusive_min_ms > 0 else n.exclusive_min_ms
                a.exclusive_max_ms = max(a.exclusive_max_ms, n.exclusive_max_ms)
                a.stddev_ms = max(a.stddev_ms, n.stddev_ms)
                a.mem_alloc_count += n.mem_alloc_count
                a.mem_free_count += n.mem_free_count
                a.mem_alloc_bytes += n.mem_alloc_bytes
                a.mem_free_bytes += n.mem_free_bytes
                a.mem_current_bytes += n.mem_current_bytes
                a.mem_peak_bytes = max(a.mem_peak_bytes, n.mem_peak_bytes)
                # Merge annotations (last value wins for strings; sum for numerics)
                for k, v in n.annotations.items():
                    if isinstance(v, dict) and k in a.annotations and isinstance(a.annotations[k], dict):
                        old = a.annotations[k]
                        # Sum avg values for numeric annotations
                        for field in ("avg", "min", "max"):
                            if field in v and field in old:
                                try:
                                    if field == "min":
                                        old[field] = min(float(old[field]), float(v[field]))
                                    elif field == "max":
                                        old[field] = max(float(old[field]), float(v[field]))
                                    else:
                                        old[field] = float(old[field]) + float(v[field])
                                except (ValueError, TypeError):
                                    pass
                    else:
                        a.annotations[k] = v
                a.counters.update(n.counters)
            else:
                # Deep copy so we don't mutate the original
                agg[n.name] = ProfileNode(
                    name=n.name,
                    call_count=n.call_count,
                    exclusive_ms=n.exclusive_ms,
                    inclusive_ms=n.inclusive_ms,
                    exclusive_min_ms=n.exclusive_min_ms,
                    exclusive_max_ms=n.exclusive_max_ms,
                    stddev_ms=n.stddev_ms,
                    file=n.file,
                    line=n.line,
                    function=n.function,
                    annotations=dict(n.annotations),
                    counters=dict(n.counters),
                    mem_alloc_count=n.mem_alloc_count,
                    mem_free_count=n.mem_free_count,
                    mem_alloc_bytes=n.mem_alloc_bytes,
                    mem_free_bytes=n.mem_free_bytes,
                    mem_current_bytes=n.mem_current_bytes,
                    mem_peak_bytes=n.mem_peak_bytes,
                    children=[],  # flat view has no children
                )
            _walk(n.children)

    _walk(nodes)
    return list(agg.values())


def _collect_roofline_points(
    nodes: list[ProfileNode],
) -> list[tuple[str, float, float, float]]:
    """Walk all nodes and collect (name, ai, gflops, exclusive_ms) for roofline plot.

    Only includes nodes that have 'flops' and at least one of 'bytes_read'/'bytes_written'
    annotations, with non-zero exclusive time.
    """
    points: list[tuple[str, float, float, float]] = []

    def _walk(node_list: list[ProfileNode]) -> None:
        for n in node_list:
            if n.exclusive_ms > 0:
                flops = _extract_numeric_annotation(n.annotations, "flops")
                br = _extract_numeric_annotation(n.annotations, "bytes_read")
                bw = _extract_numeric_annotation(n.annotations, "bytes_written")
                if flops is not None and flops > 0:
                    total_bytes = (br or 0) + (bw or 0)
                    if total_bytes > 0:
                        ai = flops / total_bytes  # FLOP/byte
                        gflops = flops / (n.exclusive_ms / 1000.0) / 1e9
                        points.append((n.name, ai, gflops, n.exclusive_ms))
            _walk(n.children)

    _walk(nodes)
    return points


# ---------------------------------------------------------------------------
# Timeline panel (requires textual-plotext)
# ---------------------------------------------------------------------------

if HAS_PLOTEXT:
    class TimelinePanel(Widget):
        """Dual-plot panel showing CPU exclusive time rate and memory usage over time."""

        DEFAULT_CSS = """
        TimelinePanel {
            height: 18;
            border-top: solid $accent;
        }

        TimelinePanel .timeline-plot {
            width: 1fr;
            height: 100%;
        }
        """

        MAX_HISTORY = 120  # ~2 min at 1 update/sec

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._timestamps: deque[float] = deque(maxlen=self.MAX_HISTORY)
            self._cpu_rate: deque[float] = deque(maxlen=self.MAX_HISTORY)
            self._mem_alloc: deque[float] = deque(maxlen=self.MAX_HISTORY)
            self._start_time: float = time.monotonic()
            self._last_exclusive_ms: float = 0.0
            self._last_wall_time: float = 0.0

        def compose(self) -> ComposeResult:
            with Horizontal():
                yield PlotextPlot(id="cpu-plot", classes="timeline-plot")
                yield PlotextPlot(id="mem-plot", classes="timeline-plot")

        def record(self, snap: ProfileSnapshot) -> None:
            """Record a data point from the latest snapshot."""
            now = time.monotonic()
            elapsed = now - self._start_time

            # CPU: compute total exclusive_ms across all threads/nodes
            total_excl = 0.0
            total_mem = 0
            for tdata in snap.threads.values():
                total_excl += sum_exclusive(tdata.children)
                total_mem += _sum_mem_current(tdata.children)

            # CPU rate: delta exclusive ms per delta wall-clock second
            wall_dt = now - self._last_wall_time if self._last_wall_time > 0 else 0.5
            excl_delta = total_excl - self._last_exclusive_ms
            cpu_rate = (excl_delta / (wall_dt * 1000.0)) * 100.0 if wall_dt > 0 else 0.0
            cpu_rate = max(0.0, min(cpu_rate, 10000.0))  # clamp

            self._last_exclusive_ms = total_excl
            self._last_wall_time = now

            self._timestamps.append(elapsed)
            self._cpu_rate.append(cpu_rate)
            self._mem_alloc.append(total_mem / (1 << 20))  # bytes -> MiB

            self._replot()

        def _replot(self) -> None:
            """Redraw both plots."""
            ts = list(self._timestamps)
            if len(ts) < 2:
                return

            # CPU plot
            try:
                cpu_widget = self.query_one("#cpu-plot", PlotextPlot)
                plt = cpu_widget.plt
                plt.clear_data()
                plt.clear_figure()
                plt.title("CPU Utilization (%)")
                plt.xlabel("time (s)")
                plt.ylabel("%")
                plt.ylim(0, max(max(self._cpu_rate), 10))
                plt.plot(ts, list(self._cpu_rate), marker="braille")
                cpu_widget.refresh()
            except Exception:
                pass

            # Memory plot
            try:
                mem_widget = self.query_one("#mem-plot", PlotextPlot)
                plt = mem_widget.plt
                plt.clear_data()
                plt.clear_figure()
                plt.title("Live Memory (MiB)")
                plt.xlabel("time (s)")
                plt.ylabel("MiB")
                mem_data = list(self._mem_alloc)
                plt.ylim(0, max(max(mem_data), 1))
                plt.plot(ts, mem_data, marker="braille")
                mem_widget.refresh()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Roofline plot panel (requires textual-plotext)
# ---------------------------------------------------------------------------

if HAS_PLOTEXT:
    class RooflinePanel(Widget):
        """Roofline model plot: arithmetic intensity vs attained GFLOPS.

        Shows each profiled function as a point on a log-log roofline chart.
        Reference lines show peak bandwidth and peak compute bounds.
        """

        DEFAULT_CSS = """
        RooflinePanel {
            height: 22;
        }

        RooflinePanel .roofline-plot {
            width: 1fr;
            height: 100%;
        }
        """

        # Default peak values (can be overridden)
        # Reasonable defaults for a modern CPU (e.g., Apple M1/M2 or Xeon)
        PEAK_GFLOPS = 200.0    # peak compute (GFLOPS)
        PEAK_BW_GBS = 50.0     # peak memory bandwidth (GB/s)

        def __init__(
            self,
            peak_gflops: float = 200.0,
            peak_bw_gbs: float = 50.0,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self._points: list[tuple[str, float, float, float]] = []
            self.PEAK_GFLOPS = peak_gflops
            self.PEAK_BW_GBS = peak_bw_gbs

        def compose(self) -> ComposeResult:
            yield PlotextPlot(id="roofline-plot", classes="roofline-plot")

        def update_data(self, nodes: list[ProfileNode]) -> None:
            """Recompute roofline points from the profile tree."""
            self._points = _collect_roofline_points(nodes)
            self._replot()

        def _replot(self) -> None:
            if not self._points:
                return

            try:
                plot_widget = self.query_one("#roofline-plot", PlotextPlot)
                plt = plot_widget.plt
                plt.clear_data()
                plt.clear_figure()
                plt.title("Roofline Model")
                plt.xlabel("Arithmetic Intensity (FLOP/byte)")
                plt.ylabel("Attained GFLOPS")
                plt.xscale("log")
                plt.yscale("log")

                # Collect data points
                ais = [p[1] for p in self._points]
                gflops_vals = [p[2] for p in self._points]
                names = [p[0] for p in self._points]

                if not ais:
                    return

                # Axis limits
                ai_min = max(min(ais) * 0.5, 0.01)
                ai_max = max(ais) * 2.0
                gf_min = max(min(gflops_vals) * 0.5, 0.001)
                gf_max = max(max(gflops_vals) * 2.0, self.PEAK_GFLOPS * 1.2)

                plt.xlim(ai_min, ai_max)
                plt.ylim(gf_min, gf_max)

                # Roofline envelope: min(peak_compute, peak_bw * AI)
                ridge_ai = self.PEAK_GFLOPS / self.PEAK_BW_GBS  # ridge point
                # Bandwidth-bound line: GFLOPS = BW * AI (for AI < ridge)
                roof_ais = []
                roof_gf = []
                ai_val = ai_min
                while ai_val <= ai_max:
                    roof_ais.append(ai_val)
                    roof_gf.append(min(self.PEAK_GFLOPS, self.PEAK_BW_GBS * ai_val))
                    ai_val *= 1.1
                plt.plot(roof_ais, roof_gf, marker="braille", label=f"roof ({self.PEAK_GFLOPS:.0f} GF, {self.PEAK_BW_GBS:.0f} GB/s)")

                # Data points
                plt.scatter(ais, gflops_vals, marker="dot")

                # Label a few top points (by GFLOPS)
                top_n = sorted(self._points, key=lambda p: p[2], reverse=True)[:5]
                for name, ai, gf, _ in top_n:
                    # Truncate long names
                    short = name[:20] if len(name) > 20 else name
                    plt.text(short, ai, gf)

                plot_widget.refresh()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Flame graph (icicle chart) widget
# ---------------------------------------------------------------------------

# Warm palette for flamegraph bars (traditional flame colors)
_FLAME_COLORS = [
    "#fee090",  # light yellow
    "#fdae61",  # light orange
    "#f46d43",  # orange
    "#d73027",  # red-orange
    "#a50026",  # dark red
    "#fdcc8a",  # peach
    "#fc8d59",  # salmon
    "#e34a33",  # vermilion
    "#b30000",  # crimson
    "#fff7bc",  # pale yellow
    "#fec44f",  # gold
    "#d95f0e",  # amber
]


def _flame_color(name: str) -> str:
    """Pick a stable color for a node name by hashing it."""
    h = hash(name) & 0xFFFFFFFF
    return _FLAME_COLORS[h % len(_FLAME_COLORS)]


@dataclass
class _FlameSpan:
    """A single bar in the flamegraph."""
    col_start: int
    col_end: int
    profile_node: ProfileNode
    depth: int


class FlameGraphPanel(Widget):
    """Interactive icicle/flamegraph view of the profiling tree.

    Root at top, children below.  Bar width proportional to inclusive time.
    Click a bar to zoom in; press Escape/Backspace to zoom out.
    """

    DEFAULT_CSS = """
    FlameGraphPanel {
        height: 22;
    }
    """

    MAX_DEPTH = 20

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fg_nodes: list[ProfileNode] = []
        self._current_roots: list[ProfileNode] = []  # what's currently displayed
        self._spans: list[_FlameSpan] = []
        self._zoom_stack: list[list[ProfileNode]] = []

    def update_data(self, nodes: list[ProfileNode]) -> None:
        """Set the tree data and rebuild layout."""
        self._fg_nodes = nodes
        if not self._zoom_stack:
            self._current_roots = nodes
        self._rebuild_from(self._current_roots)
        self.refresh()

    def zoom_in(self, node: ProfileNode) -> None:
        """Zoom into a specific node (it becomes the new root)."""
        if not node.children:
            return
        self._zoom_stack.append(self._current_roots)
        self._current_roots = [node]
        self._rebuild_from(self._current_roots)
        self.refresh()

    def zoom_out(self) -> None:
        """Pop zoom stack and restore previous view."""
        if self._zoom_stack:
            self._current_roots = self._zoom_stack.pop()
            self._rebuild_from(self._current_roots)
            self.refresh()

    def zoom_reset(self) -> None:
        """Reset zoom to show full tree."""
        self._zoom_stack.clear()
        self._current_roots = self._fg_nodes
        self._rebuild_from(self._current_roots)
        self.refresh()

    def _rebuild_from(self, roots: list[ProfileNode]) -> None:
        """Compute all spans from the given root nodes."""
        self._spans = []
        width = self.size.width if self.size.width > 0 else 120
        total_incl = sum(n.inclusive_ms for n in roots)
        if total_incl <= 0:
            return
        self._layout(roots, 0, 0, width, total_incl)

    def _layout(
        self,
        nodes: list[ProfileNode],
        depth: int,
        x_start: int,
        x_end: int,
        total_ms: float,
    ) -> None:
        """Recursively lay out spans. x_start/x_end are in terminal columns."""
        if depth >= self.MAX_DEPTH or x_end <= x_start:
            return

        # Distribute width among children proportional to inclusive_ms
        x = float(x_start)
        for node in nodes:
            if total_ms <= 0:
                break
            w = (node.inclusive_ms / total_ms) * (x_end - x_start)
            col_s = int(x + 0.5)
            col_e = int(x + w + 0.5)
            if col_e <= col_s:
                if w > 0.3:
                    col_e = col_s + 1
                else:
                    x += w
                    continue

            self._spans.append(_FlameSpan(
                col_start=col_s, col_end=col_e, profile_node=node, depth=depth
            ))

            if node.children:
                child_total = sum(c.inclusive_ms for c in node.children)
                self._layout(node.children, depth + 1, col_s, col_e, child_total)

            x += w

    def render(self) -> Text:
        """Render the flamegraph as colored Rich text."""
        width = self.size.width if self.size.width > 0 else 120
        height = self.size.height if self.size.height > 0 else 20

        if not self._spans:
            return Text("No data (press f to toggle, Esc to zoom out)")

        max_depth = max(s.depth for s in self._spans) + 1
        max_depth = min(max_depth, height)

        # Build a 2D grid: rows[depth] = list of (col_start, col_end, node)
        rows: dict[int, list[_FlameSpan]] = {}
        for span in self._spans:
            if span.depth < max_depth:
                rows.setdefault(span.depth, []).append(span)

        lines: list[Text] = []

        # Zoom indicator
        if self._zoom_stack:
            zoom_text = Text(f" Zoomed ({len(self._zoom_stack)} levels deep) — Esc to zoom out ", style="bold on #333333")
            lines.append(zoom_text)

        for d in range(max_depth):
            line = Text()
            spans = sorted(rows.get(d, []), key=lambda s: s.col_start)
            pos = 0
            for span in spans:
                # Fill gap before this span
                if span.col_start > pos:
                    line.append(" " * (span.col_start - pos))
                    pos = span.col_start

                bar_width = span.col_end - span.col_start
                if bar_width <= 0:
                    continue

                # Fit the label into the bar
                label = span.profile_node.name
                incl_str = f" {span.profile_node.inclusive_ms:.1f}ms"
                if bar_width >= len(label) + len(incl_str) + 2:
                    display = f" {label}{incl_str} "
                elif bar_width >= len(label) + 2:
                    display = f" {label} "
                elif bar_width >= 4:
                    display = f" {label[:bar_width - 2]} "
                else:
                    display = "\u2588" * bar_width

                # Pad or truncate to exact bar width
                if len(display) < bar_width:
                    display = display + " " * (bar_width - len(display))
                elif len(display) > bar_width:
                    display = display[:bar_width]

                color = _flame_color(span.profile_node.name)
                line.append(display, style=f"black on {color}")
                pos = span.col_end

            # Fill remaining width
            if pos < width:
                line.append(" " * (width - pos))

            lines.append(line)

        # Join lines
        result = Text()
        for i, line in enumerate(lines):
            if i > 0:
                result.append("\n")
            result.append_text(line)

        return result

    def on_click(self, event) -> None:
        """Handle click: zoom into the clicked span."""
        col = event.x
        row = event.y

        # Account for zoom indicator line
        if self._zoom_stack:
            row -= 1

        if row < 0:
            return

        # Find the span at (row, col)
        for span in self._spans:
            if span.depth == row and span.col_start <= col < span.col_end:
                self.zoom_in(span.profile_node)
                # Post a message so the app can update the detail panel
                self.post_message(self.SpanClicked(span.profile_node))
                return

    def on_key(self, event) -> None:
        if event.key in ("escape", "backspace"):
            if self._zoom_stack:
                self.zoom_out()
                event.prevent_default()
                event.stop()

    class SpanClicked(TextualMessage):
        """Posted when a flamegraph span is clicked."""
        def __init__(self, profile_node: ProfileNode) -> None:
            self.profile_node = profile_node
            super().__init__()


@dataclass
class TimelineEvent:
    """A single event for Gantt chart display."""
    thread_id: str = ""
    name: str = ""
    start_ms: float = 0.0
    end_ms: float = 0.0


class GanttPanel(Widget):
    """Thread Gantt chart showing recent function execution spans."""

    DEFAULT_CSS = """
    GanttPanel {
        height: 22;
        border-top: solid $accent;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._events: list[TimelineEvent] = []

    def update_events(self, events: list[TimelineEvent]) -> None:
        self._events = events
        self.refresh()

    def render(self) -> Text:
        if not self._events:
            return Text("No timeline data (requires live connection)")

        width = self.size.width if self.size.width > 0 else 120
        height = self.size.height if self.size.height > 0 else 20

        # Find time range
        all_start = min(e.start_ms for e in self._events)
        all_end = max(e.end_ms for e in self._events)
        time_range = all_end - all_start
        if time_range <= 0:
            return Text("No time range")

        # Group by thread
        threads: dict[str, list[TimelineEvent]] = {}
        for e in self._events:
            threads.setdefault(e.thread_id, []).append(e)

        lines: list[Text] = []
        # Header with time scale
        label_w = 12
        chart_w = width - label_w - 1
        if chart_w < 10:
            chart_w = 10

        header = Text()
        header.append(f"{'Thread':<{label_w}}", style="bold")
        # Time markers
        for i in range(0, chart_w, max(1, chart_w // 5)):
            t = all_start + (i / chart_w) * time_range
            marker = f"{t:.0f}ms"
            remaining = chart_w - i
            if len(marker) <= remaining:
                header.append(marker)
                padding = min(max(1, chart_w // 5) - len(marker), remaining - len(marker))
                if padding > 0:
                    header.append(" " * padding)
        lines.append(header)

        # Limit number of threads shown
        max_threads = height - 2
        for tid_idx, (tid, tevents) in enumerate(sorted(threads.items())):
            if tid_idx >= max_threads:
                break
            line = Text()
            label = f"T{tid}"[:label_w - 1].ljust(label_w)
            line.append(label, style="bold")

            # Render events as colored bars
            row = [" "] * chart_w
            for evt in tevents:
                col_start = int((evt.start_ms - all_start) / time_range * chart_w)
                col_end = int((evt.end_ms - all_start) / time_range * chart_w)
                col_start = max(0, min(col_start, chart_w - 1))
                col_end = max(col_start + 1, min(col_end, chart_w))
                for c in range(col_start, col_end):
                    row[c] = "\u2588"

            # Color the bars by function name
            i = 0
            while i < chart_w:
                if row[i] == "\u2588":
                    # Find the event at this position to get the color
                    bar_start = i
                    while i < chart_w and row[i] == "\u2588":
                        i += 1
                    # Find matching event
                    mid_col = (bar_start + i) // 2
                    mid_time = all_start + (mid_col / chart_w) * time_range
                    best_evt = None
                    for evt in tevents:
                        if evt.start_ms <= mid_time <= evt.end_ms:
                            best_evt = evt
                            break
                    color = _flame_color(best_evt.name) if best_evt else "#ffffff"
                    bar_text = "\u2588" * (i - bar_start)
                    # Try to fit the name
                    if best_evt and len(best_evt.name) <= len(bar_text) - 2:
                        padded = f" {best_evt.name} ".ljust(len(bar_text))[:len(bar_text)]
                        line.append(padded, style=f"black on {color}")
                    else:
                        line.append(bar_text, style=color)
                else:
                    line.append(" ")
                    i += 1

            lines.append(line)

        result = Text()
        for i, line in enumerate(lines):
            if i > 0:
                result.append("\n")
            result.append_text(line)
        return result


class SourcePanel(Static):
    """Shows source code context for the selected node."""

    DEFAULT_CSS = """
    SourcePanel {
        height: auto;
        max-height: 35;
        min-height: 3;
        padding: 0 1;
        background: $surface;
        border-top: solid $accent;
    }
    """

    _last_content: str = ""
    _context_lines: int = 15  # lines above and below

    def update_node(self, node: ProfileNode | None) -> None:
        if node is None or not node.file or node.line <= 0:
            content = "No source location available"
            if content != self._last_content:
                self._last_content = content
                self.update(content)
            return
        filepath = node.file
        if not os.path.isfile(filepath):
            content = f"File not found: {filepath}"
            if content != self._last_content:
                self._last_content = content
                self.update(content)
            return
        try:
            with open(filepath, "r", errors="replace") as f:
                all_lines = f.readlines()
        except OSError as exc:
            content = f"Cannot read {filepath}: {exc}"
            if content != self._last_content:
                self._last_content = content
                self.update(content)
            return
        target = node.line  # 1-based
        start = max(1, target - self._context_lines)
        end = min(len(all_lines), target + self._context_lines)
        lines = [f"[bold]{filepath}:{target}[/bold]  (fn: {node.function or '?'})"]
        for i in range(start, end + 1):
            line_text = all_lines[i - 1].rstrip()
            # Escape Rich markup
            line_text = line_text.replace("[", "\\[")
            if i == target:
                lines.append(f"[bold yellow]>{i:5d}[/bold yellow] [bold yellow]{line_text}[/bold yellow]")
            else:
                lines.append(f" {i:5d}  {line_text}")
        content = "\n".join(lines)
        if content != self._last_content:
            self._last_content = content
            self.update(content)


class HotspotsPanel(Static):
    """Shows the top-N hottest functions by exclusive time across all threads."""

    DEFAULT_CSS = """
    HotspotsPanel {
        height: auto;
        max-height: 15;
        padding: 0 1;
        background: $surface;
        border-bottom: solid $accent;
    }
    """

    _last_content: str = ""

    def update_hotspots(self, sessions: dict[str, "SessionState"]) -> None:
        """Aggregate all threads across sessions and show top 10 by exclusive_ms."""
        all_nodes: list[ProfileNode] = []
        for session in sessions.values():
            snap = session.latest_snapshot
            if snap:
                for tdata in snap.threads.values():
                    all_nodes.extend(tdata.children)
        if not all_nodes:
            content = "No data"
        else:
            flat = _aggregate_flat(all_nodes)
            total_ms = sum(n.exclusive_ms for n in flat)
            if total_ms <= 0:
                total_ms = 1.0
            flat.sort(key=lambda n: n.exclusive_ms, reverse=True)
            top = flat[:10]
            lines = ["[bold]Top-10 Hotspots (by exclusive time)[/bold]"]
            lines.append(f"  {'#':>2s}  {'%':>6s}  {'excl(ms)':>10s}  {'calls':>8s}  {'mean(ms)':>10s}  name")
            for i, n in enumerate(top, 1):
                pct = n.exclusive_ms / total_ms * 100.0
                mean = n.exclusive_ms / n.call_count if n.call_count > 0 else 0.0
                lines.append(
                    f"  {i:2d}  {pct:5.1f}%  {n.exclusive_ms:10.3f}  {n.call_count:8d}  {mean:10.3f}  {n.name}"
                )
            content = "\n".join(lines)
        if content != self._last_content:
            self._last_content = content
            self.update(content)


class MicroArchPanel(Static):
    """Top-down micro-architecture stall analysis from hardware counters."""

    DEFAULT_CSS = """
    MicroArchPanel {
        height: auto;
        max-height: 20;
        min-height: 3;
        padding: 0 1;
        background: $surface;
        border-top: solid $accent;
    }
    """

    _last_content: str = ""

    def update_node(self, node: ProfileNode | None) -> None:
        if node is None:
            content = "Select a row to see micro-architecture analysis"
            if content != self._last_content:
                self._last_content = content
                self.update(content)
            return

        counters = node.counters
        if not counters:
            content = "[bold]Micro-Architecture Analysis[/bold]\n  (no hardware counters available)"
            if content != self._last_content:
                self._last_content = content
                self.update(content)
            return

        # Extract counter totals
        cycles = 0
        instructions = 0
        cache_misses = 0
        branch_misses = 0
        for name, cdata in counters.items():
            total = cdata.get("total", 0) if isinstance(cdata, dict) else 0
            nl = name.lower()
            if "cycle" in nl and "instruction" not in nl:
                cycles = total
            elif "instruction" in nl:
                instructions = total
            elif "cache" in nl and "miss" in nl:
                cache_misses = total
            elif "branch" in nl and "miss" in nl:
                branch_misses = total

        if cycles == 0 and instructions == 0:
            content = "[bold]Micro-Architecture Analysis[/bold]\n  (no cycle/instruction counters)"
            if content != self._last_content:
                self._last_content = content
                self.update(content)
            return

        # Compute derived metrics
        ipc = instructions / cycles if cycles > 0 else 0.0
        cache_miss_rate = (cache_misses / instructions * 1000.0) if instructions > 0 else 0.0
        branch_miss_rate = (branch_misses / instructions * 1000.0) if instructions > 0 else 0.0

        # Simple top-down stall classification
        classification = "Unknown"
        class_color = "white"
        if ipc >= 2.0:
            classification = "Retiring"
            class_color = "green"
        elif branch_miss_rate > 5.0:
            classification = "Bad Speculation"
            class_color = "red"
        elif cache_miss_rate > 10.0:
            classification = "Back-end Bound (memory)"
            class_color = "yellow"
        elif ipc < 0.5:
            classification = "Front-end Bound"
            class_color = "magenta"
        elif ipc < 2.0:
            classification = "Compute Bound"
            class_color = "cyan"

        # Build visual bar (20 chars wide)
        bar_width = 20
        ipc_frac = min(ipc / 4.0, 1.0)  # normalize to 0-1 (4.0 IPC = full bar)
        filled = int(ipc_frac * bar_width)
        bar = "[green]" + "\u2588" * filled + "[/green]" + "\u2591" * (bar_width - filled)

        lines = [
            f"[bold]Micro-Architecture Analysis[/bold]  [{class_color}]{classification}[/{class_color}]",
            f"  IPC:  {bar}  {ipc:.2f}",
            f"  {'Metric':<25s}  {'Value':>12s}  {'Per 1K insn':>12s}",
            f"  {'─' * 25}  {'─' * 12}  {'─' * 12}",
            f"  {'Cycles':<25s}  {cycles:>12,d}",
            f"  {'Instructions':<25s}  {instructions:>12,d}",
        ]
        if cache_misses > 0:
            lines.append(
                f"  {'Cache misses':<25s}  {cache_misses:>12,d}  {cache_miss_rate:>11.2f}"
            )
        if branch_misses > 0:
            lines.append(
                f"  {'Branch misses':<25s}  {branch_misses:>12,d}  {branch_miss_rate:>11.2f}"
            )
        content = "\n".join(lines)
        if content != self._last_content:
            self._last_content = content
            self.update(content)


class LogPanel(Widget):
    """Scrollable log message panel with auto-scroll to bottom."""

    DEFAULT_CSS = """
    LogPanel {
        height: 20;
        min-height: 5;
        max-height: 30;
        background: $surface;
        border-top: solid $accent;
    }
    LogPanel RichLog {
        height: 1fr;
    }
    """
    _min_level: int = 2  # INFO by default
    _last_count: int = 0  # track how many entries we've written

    def compose(self) -> ComposeResult:
        yield RichLog(id="log-richlog", markup=True, wrap=True, max_lines=500)

    def cycle_level(self) -> int:
        """Cycle through min display level: INFO->WARN->ERROR->TRACE->DEBUG->INFO"""
        cycle = [2, 3, 4, 0, 1]
        idx = cycle.index(self._min_level) if self._min_level in cycle else 0
        self._min_level = cycle[(idx + 1) % len(cycle)]
        # Reset so we rewrite all entries at new level
        self._last_count = 0
        return self._min_level

    def update_logs(self, entries: deque) -> None:
        try:
            log_widget = self.query_one("#log-richlog", RichLog)
        except Exception:
            return

        total = len(entries)
        if total == self._last_count:
            return  # nothing new

        if self._last_count == 0 or total < self._last_count:
            # Full rebuild (first time, level change, or entries wrapped around)
            log_widget.clear()
            filtered = [e for e in entries if e.level >= self._min_level]
            for e in filtered:
                log_widget.write(self._format_entry(e))
        else:
            # Incremental: only write new entries
            new_entries = list(entries)[self._last_count:]
            for e in new_entries:
                if e.level >= self._min_level:
                    log_widget.write(self._format_entry(e))

        self._last_count = total

    @staticmethod
    def _format_entry(e: LogEntry) -> str:
        level_name = LOG_LEVEL_NAMES.get(e.level, "?")
        style = LOG_LEVEL_STYLES.get(e.level, "")
        ts_short = e.timestamp[11:23] if len(e.timestamp) >= 23 else e.timestamp
        loc = f"{e.file}:{e.line}" if e.file else ""
        safe_msg = e.message.replace("[", "\\[")
        return f"[{style}]{ts_short} [{level_name:>5}][/{style}] {safe_msg}  ({loc})"


import subprocess
import shutil


class AssemblyPanel(Widget):
    """Shows JIT kernel IR and native assembly, or executable disassembly."""

    DEFAULT_CSS = """
    AssemblyPanel {
        height: 40;
        min-height: 10;
        padding: 0 1;
        background: $surface;
        border-top: solid $accent;
    }

    AssemblyPanel TabbedContent {
        height: 1fr;
    }

    AssemblyPanel .asm-scroll {
        height: 1fr;
    }

    AssemblyPanel Static {
        height: auto;
    }
    """

    _last_hash: int | None = None
    _last_func: str | None = None
    _cache: dict[str, str] = {}  # cache key -> content

    def compose(self) -> ComposeResult:
        with TabbedContent(id="asm-tabs"):
            with TabPane("MLIR IR", id="asm-mlir"):
                with VerticalScroll(classes="asm-scroll"):
                    yield Static("(no data)", id="asm-mlir-content")
            with TabPane("LLVM IR", id="asm-llvm"):
                with VerticalScroll(classes="asm-scroll"):
                    yield Static("(no data)", id="asm-llvm-content")
            with TabPane("Assembly", id="asm-native"):
                with VerticalScroll(classes="asm-scroll"):
                    yield Static("(no data)", id="asm-native-content")

    def _syntax_highlight_asm(self, text: str) -> str:
        """Basic syntax highlighting for assembly."""
        lines = []
        for line in text.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith(";") or stripped.startswith("#"):
                lines.append(f"[dim]{line}[/dim]")
            elif ":" in line and not line.startswith(" "):
                lines.append(f"[bold cyan]{line}[/bold cyan]")
            else:
                # Escape Rich markup
                line = line.replace("[", "\\[")
                lines.append(line)
        return "\n".join(lines)

    def _syntax_highlight_ir(self, text: str) -> str:
        """Basic syntax highlighting for MLIR/LLVM IR."""
        lines = []
        for line in text.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith(";") or stripped.startswith("//"):
                lines.append(f"[dim]{line}[/dim]")
            elif stripped.startswith("func") or stripped.startswith("define") or stripped.startswith("module"):
                line_esc = line.replace("[", "\\[")
                lines.append(f"[bold]{line_esc}[/bold]")
            else:
                line = line.replace("[", "\\[")
                lines.append(line)
        return "\n".join(lines)

    async def update_for_jit_kernel(self, kernel_hash: int, client: "ProfileClient | None") -> None:
        """Request IR and disassembly for a JIT kernel from the server."""
        if kernel_hash == self._last_hash:
            return
        self._last_hash = kernel_hash
        self._last_func = None

        if not client or not client.connected:
            self._set_all("(not connected to server)")
            return

        # Request MLIR IR
        cache_key_mlir = f"mlir_{kernel_hash}"
        if cache_key_mlir in self._cache:
            self._set_tab("asm-mlir-content", self._syntax_highlight_ir(self._cache[cache_key_mlir]))
        else:
            resp = await client.send_request("get_jit_mlir_ir", {"hash": str(kernel_hash)})
            ir = resp.get("ir", resp.get("error", ""))
            self._cache[cache_key_mlir] = ir
            self._set_tab("asm-mlir-content", self._syntax_highlight_ir(ir))

        # Request LLVM IR
        cache_key_llvm = f"llvm_{kernel_hash}"
        if cache_key_llvm in self._cache:
            self._set_tab("asm-llvm-content", self._syntax_highlight_ir(self._cache[cache_key_llvm]))
        else:
            resp = await client.send_request("get_jit_llvm_ir", {"hash": str(kernel_hash)})
            ir = resp.get("ir", resp.get("error", ""))
            self._cache[cache_key_llvm] = ir
            self._set_tab("asm-llvm-content", self._syntax_highlight_ir(ir))

        # Request disassembly
        cache_key_disasm = f"disasm_{kernel_hash}"
        if cache_key_disasm in self._cache:
            self._set_tab("asm-native-content", self._syntax_highlight_asm(self._cache[cache_key_disasm]))
        else:
            resp = await client.send_request("get_jit_disasm", {"hash": str(kernel_hash)})
            disasm = resp.get("disasm", resp.get("error", ""))
            self._cache[cache_key_disasm] = disasm
            self._set_tab("asm-native-content", self._syntax_highlight_asm(disasm))

    def update_for_executable_func(self, func_name: str, executable_path: str) -> None:
        """Disassemble a function from the executable or its shared libraries."""
        if func_name == self._last_func:
            return
        self._last_func = func_name
        self._last_hash = None

        self._set_tab("asm-mlir-content", "(not a JIT kernel)")
        self._set_tab("asm-llvm-content", "(not a JIT kernel)")

        if not executable_path or not os.path.isfile(executable_path):
            self._set_tab("asm-native-content", f"Executable not available locally: {executable_path or '(unknown)'}")
            return

        cache_key = f"exe_{func_name}"
        if cache_key in self._cache:
            self._set_tab("asm-native-content", self._syntax_highlight_asm(self._cache[cache_key]))
            return

        objdump = shutil.which("llvm-objdump")
        if not objdump:
            self._set_tab("asm-native-content", "llvm-objdump not found in PATH")
            return

        # Build list of binaries to search: executable first, then shared libs in same dir
        exe_dir = os.path.dirname(executable_path)
        binaries = [executable_path]
        if os.path.isdir(exe_dir):
            for f in sorted(os.listdir(exe_dir)):
                if f.endswith((".dylib", ".so")) or ".so." in f:
                    full = os.path.join(exe_dir, f)
                    if full not in binaries:
                        binaries.append(full)
        # Also check ../lib relative to executable (common install layout)
        lib_dir = os.path.join(exe_dir, "..", "lib")
        if os.path.isdir(lib_dir):
            for f in sorted(os.listdir(lib_dir)):
                if f.endswith((".dylib", ".so")) or ".so." in f:
                    full = os.path.join(lib_dir, f)
                    if full not in binaries:
                        binaries.append(full)

        for binary in binaries:
            try:
                result = subprocess.run(
                    [objdump, "-d", "--no-show-raw-insn", "--demangle", binary],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    continue
                func_section = self._extract_function(result.stdout, func_name)
                if not func_section.startswith("Function '"):
                    # Found it
                    header = f"; from {os.path.basename(binary)}\n"
                    func_section = header + func_section
                    self._cache[cache_key] = func_section
                    self._set_tab("asm-native-content", self._syntax_highlight_asm(func_section))
                    return
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue

        self._set_tab("asm-native-content", f"Function '{func_name}' not found in executable or shared libraries")

    def _extract_function(self, objdump_output: str, func_name: str) -> str:
        """Extract a specific function's disassembly from llvm-objdump output.

        With --demangle, function headers look like:
            0000000000001234 <einsums::blas::dgemm(...)>:
        We match func_name as a substring of the demangled name.
        A function ends when the next function header appears (line ending with '>:').
        """
        lines = objdump_output.split("\n")
        in_function = False
        result_lines: list[str] = []
        for line in lines:
            if not in_function:
                # Look for a function header containing our name
                if func_name in line and line.rstrip().endswith(">:"):
                    in_function = True
                    result_lines.append(line)
            else:
                # Next function header = end of current function
                if line.rstrip().endswith(">:") and "<" in line:
                    break
                result_lines.append(line)
        # Strip trailing blank lines
        while result_lines and result_lines[-1].strip() == "":
            result_lines.pop()
        if result_lines:
            return "\n".join(result_lines)
        return f"Function '{func_name}' not found in disassembly"

    def update_not_jit(self) -> None:
        """Show that the selected node is not a JIT kernel."""
        self._last_hash = None
        self._last_func = None
        self._set_all("Select a JIT kernel node and press A to view assembly")

    def _set_all(self, text: str) -> None:
        for tab_id in ("asm-mlir-content", "asm-llvm-content", "asm-native-content"):
            self._set_tab(tab_id, text)

    def _set_tab(self, tab_id: str, content: str) -> None:
        try:
            widget = self.query_one(f"#{tab_id}", Static)
            widget.update(content)
        except Exception:
            pass


def _build_bottom_up(nodes: list[ProfileNode]) -> list[ProfileNode]:
    """Invert the tree: each callee becomes a root, callers become children.

    Returns a flat list of ProfileNodes where each node's children represent
    unique callers with proportional time.
    """
    # Map: callee_name -> {caller_name -> (caller_excl_ms, caller_count)}
    callee_map: dict[str, dict[str, tuple[float, int]]] = {}
    # Also track callee totals
    callee_totals: dict[str, tuple[float, float, int]] = {}  # excl, incl, count

    def _walk(node_list: list[ProfileNode], caller_name: str) -> None:
        for n in node_list:
            if n.name not in callee_totals:
                callee_totals[n.name] = (0.0, 0.0, 0)
            old = callee_totals[n.name]
            callee_totals[n.name] = (old[0] + n.exclusive_ms, max(old[1], n.inclusive_ms), old[2] + n.call_count)

            if caller_name:
                if n.name not in callee_map:
                    callee_map[n.name] = {}
                cm = callee_map[n.name]
                if caller_name not in cm:
                    cm[caller_name] = (0.0, 0)
                old_c = cm[caller_name]
                cm[caller_name] = (old_c[0] + n.exclusive_ms, old_c[1] + n.call_count)

            _walk(n.children, n.name)

    _walk(nodes, "")

    result: list[ProfileNode] = []
    for callee_name, (excl, incl, count) in callee_totals.items():
        callee_node = ProfileNode(
            name=callee_name,
            exclusive_ms=excl,
            inclusive_ms=incl,
            call_count=count,
        )
        # Add caller nodes as children
        callers = callee_map.get(callee_name, {})
        for caller_name, (caller_excl, caller_count) in sorted(callers.items(), key=lambda x: x[1][0], reverse=True):
            caller_node = ProfileNode(
                name=caller_name,
                exclusive_ms=caller_excl,
                inclusive_ms=caller_excl,
                call_count=caller_count,
            )
            callee_node.children.append(caller_node)
        result.append(callee_node)
    return result


class ThreadTreeView(Widget):
    """Displays the profiling tree for a single thread as a DataTable."""

    DEFAULT_CSS = """
    ThreadTreeView {
        height: 1fr;
    }
    """

    def __init__(self, thread_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.thread_id = thread_id
        self._data: list[ProfileNode] = []
        self._filter = ""
        self._total_ms: float = 1.0
        # Collapse/expand state: set of path keys that are collapsed
        self._collapsed: set[str] = set()
        # Maps row index -> (ProfileNode, path_key)
        self._row_map: list[tuple[ProfileNode, str]] = []
        # Row keys from DataTable for in-place updates
        self._row_keys: list[Any] = []
        # Column keys (set once on first build)
        self._col_keys: list[Any] = []
        # Whether columns have been added
        self._initialized: bool = False
        # Sort settings
        self._sort_key: str = "natural"
        self._sort_reverse: bool = True
        # Hot-path mode
        self._hotpath_mode: bool = False
        self._hotpath_keys: set[str] = set()
        # Flat profile mode
        self._flat_mode: bool = False
        # Bottom-up mode
        self._bottom_up_mode: bool = False
        # Bookmarks (shared reference from SessionState)
        self._bookmarks: set[str] = set()
        # Color theme: "none", "heat"
        self._color_theme: str = "heat"

    def compose(self) -> ComposeResult:
        yield ProfileDataTable(id=f"tree-{self.thread_id}")

    def on_mount(self) -> None:
        table = self.query_one(ProfileDataTable)
        table.cursor_type = "row"

    def update_data(self, nodes: list[ProfileNode], name_filter: str = "") -> None:
        self._data = nodes
        self._filter = name_filter
        self._refresh_table()

    def _sort_nodes(self, nodes: list[ProfileNode]) -> list[ProfileNode]:
        """Sort nodes and their children recursively."""
        result = []
        for node in nodes:
            # Deep copy to avoid mutating original data
            n = copy.copy(node)
            n.children = self._sort_nodes(node.children)
            result.append(n)
        if self._sort_key == "natural":
            return result  # preserve original insertion order
        key_funcs = {
            "exclusive": lambda n: n.exclusive_ms,
            "count": lambda n: n.call_count,
            "mean": lambda n: (n.exclusive_ms / n.call_count if n.call_count > 0 else 0),
            "name": lambda n: n.name.lower(),
        }
        reverse = self._sort_reverse
        if self._sort_key == "name":
            reverse = not self._sort_reverse  # name sorts ascending by default
        return sorted(result, key=key_funcs.get(self._sort_key, key_funcs["exclusive"]), reverse=reverse)

    def _is_anomaly(self, node: ProfileNode) -> bool:
        """Check if a node's latest per-call time deviates significantly from its mean."""
        return _is_node_anomaly(node)

    def _row_cells(self, row: FlatRow, pct: float) -> tuple[str, ...]:
        """Compute the cell values for a FlatRow."""
        node = row.node
        mean_ms = node.exclusive_ms / node.call_count if node.call_count > 0 else 0.0
        indent = "  " * row.depth
        has_children = bool(node.children)
        if has_children:
            marker = ">" if row.collapsed else "v "
        else:
            marker = "  "

        # Bookmark indicator
        bookmark_flag = "\u2691" if node.name in self._bookmarks else ""  # ⚑

        # Anomaly indicator (shown in its own column)
        anomaly_flag = "!" if self._is_anomaly(node) else ""

        name = f"{indent}{marker}{node.name}"

        pct_str = f"{pct:5.1f}%"
        bar = _make_bar(pct)

        # % parent calculation
        if row.parent_inclusive_ms > 0:
            pct_parent = node.exclusive_ms / row.parent_inclusive_ms * 100.0
        else:
            # Root node: same as global %
            pct_parent = pct
        pct_parent_str = f"{pct_parent:5.1f}%"

        # Color coding for % and bar columns
        pct_str = _color_wrap(pct_str, pct)
        pct_parent_str = _color_wrap(pct_parent_str, pct_parent)
        bar = _color_wrap(bar, pct)

        # Heat-map coloring for numeric columns
        if self._color_theme == "heat":
            excl_str = _color_wrap(f"{node.exclusive_ms:10.3f}", pct)
            incl_str = _color_wrap(f"{node.inclusive_ms:10.3f}", pct)
            mean_str = _color_wrap(f"{mean_ms:8.3f}", pct)
            name = _color_wrap(name, pct)
        else:
            excl_str = f"{node.exclusive_ms:10.3f}"
            incl_str = f"{node.inclusive_ms:10.3f}"
            mean_str = f"{mean_ms:8.3f}"

        alloc_str = _format_bytes(node.mem_alloc_bytes) if node.mem_alloc_bytes else ""
        peak_str = _format_bytes(node.mem_peak_bytes) if node.mem_peak_bytes else ""

        return (
            ">" if row.collapsed else ("v" if has_children else " "),
            pct_str,
            pct_parent_str,
            bar,
            excl_str,
            incl_str,
            f"{node.call_count}",
            mean_str,
            alloc_str,
            peak_str,
            bookmark_flag,
            anomaly_flag,
            name,
        )

    def _row_cells_hotpath(self, row: FlatRow, pct: float, is_hot: bool) -> tuple[str, ...]:
        """Compute cell values, with hot-path marker if applicable."""
        cells = list(self._row_cells(row, pct))
        if is_hot:
            # Mark the name column (last element) with a hot-path indicator
            cells[-1] = f"[bold magenta]* {cells[-1]}[/]"
        return tuple(cells)

    def _ensure_columns(self) -> None:
        """Add columns once on first use."""
        if self._initialized:
            return
        table = self.query_one(ProfileDataTable)
        self._col_keys = list(table.add_columns(
            "", "%", "% parent", "bar", "excl(ms)", "incl(ms)", "count", "mean(ms)", "alloc", "peak", "\u2691", "anom", "name"
        ))
        self._initialized = True

    def _refresh_table(self, force_rebuild: bool = False) -> None:
        table = self.query_one(ProfileDataTable)
        self._ensure_columns()

        self._total_ms = sum_exclusive(self._data)
        if self._total_ms <= 0:
            self._total_ms = 1.0

        # In flat mode or bottom-up mode, aggregate by function name (no tree structure)
        if self._bottom_up_mode:
            agg = _build_bottom_up(self._data)
            sorted_data = self._sort_nodes(agg)
            self._hotpath_keys = set()
            flat = [
                (FlatRow(depth=0, node=n, collapsed=False, parent_inclusive_ms=0.0), n.name)
                for n in sorted_data
                if _name_matches(n.name, self._filter)
            ]
            # Expand to show callers as children (1-level deep)
            expanded: list[tuple[FlatRow, str]] = []
            for frow, path_key in flat:
                expanded.append((frow, path_key))
                if path_key not in self._collapsed and frow.node.children:
                    for child in frow.node.children:
                        child_key = f"{path_key}/{child.name}"
                        expanded.append((
                            FlatRow(depth=1, node=child, collapsed=False, parent_inclusive_ms=frow.node.inclusive_ms),
                            child_key,
                        ))
            flat = expanded
        elif self._flat_mode:
            agg = _aggregate_flat(self._data)
            sorted_data = self._sort_nodes(agg)
            self._hotpath_keys = set()
            # Flat rows at depth 0, no children, no collapse
            flat = [
                (FlatRow(depth=0, node=n, collapsed=False, parent_inclusive_ms=0.0), n.name)
                for n in sorted_data
                if _name_matches(n.name, self._filter)
            ]
        else:
            # Sort a copy of the data
            sorted_data = self._sort_nodes(self._data)

            # Compute hot-path keys if in hot-path mode
            if self._hotpath_mode:
                self._hotpath_keys = _compute_hotpath(sorted_data)
            else:
                self._hotpath_keys = set()

            flat = flatten_tree(
                sorted_data, name_filter=self._filter, collapsed_paths=self._collapsed
            )

        new_count = len(flat)
        old_count = len(self._row_keys)

        # Check if cursor is at the last row (follow-tail behavior)
        was_at_end = old_count > 0 and table.cursor_row >= old_count - 1

        # Update existing rows in place
        new_row_map: list[tuple[ProfileNode, str]] = []
        new_row_keys: list[Any] = list(self._row_keys)

        for i, (row, path_key) in enumerate(flat):
            pct = (row.node.exclusive_ms / self._total_ms * 100.0) if self._total_ms > 0 else 0.0
            is_hot = path_key in self._hotpath_keys
            cells = self._row_cells_hotpath(row, pct, is_hot)

            if i < old_count:
                # Update existing row
                row_key = self._row_keys[i]
                for j, col_key in enumerate(self._col_keys):
                    table.update_cell(row_key, col_key, cells[j], update_width=False)
            else:
                # Add new row
                row_key = table.add_row(*cells)
                new_row_keys.append(row_key)

            new_row_map.append((row.node, path_key))

        # Remove excess rows (iterate in reverse so indices stay valid)
        if new_count < old_count:
            for i in range(old_count - 1, new_count - 1, -1):
                table.remove_row(self._row_keys[i])

        self._row_map = new_row_map
        self._row_keys = new_row_keys[:new_count]

        # If cursor was at the end, keep it at the new end
        if was_at_end and new_count > 0 and new_count > old_count:
            table.move_cursor(row=new_count - 1)

    def toggle_row(self, row_idx: int) -> None:
        """Toggle collapse/expand for the given row."""
        if row_idx < 0 or row_idx >= len(self._row_map):
            return
        node, path_key = self._row_map[row_idx]
        if not node.children:
            return
        if path_key in self._collapsed:
            self._collapsed.discard(path_key)
        else:
            self._collapsed.add(path_key)
        self._refresh_table()

    def get_node_at_row(self, row_idx: int) -> ProfileNode | None:
        if 0 <= row_idx < len(self._row_map):
            return self._row_map[row_idx][0]
        return None

    def expand_all(self) -> None:
        self._collapsed.clear()
        self._refresh_table()

    def collapse_all(self) -> None:
        """Collapse all nodes that have children."""
        self._collapsed.clear()
        self._collapse_nodes(self._data, "")
        self._refresh_table()

    def _collapse_nodes(self, nodes: list[ProfileNode], prefix: str) -> None:
        for node in nodes:
            path_key = f"{prefix}/{node.name}" if prefix else node.name
            if node.children:
                self._collapsed.add(path_key)
                self._collapse_nodes(node.children, path_key)

    def toggle_flat_mode(self) -> None:
        """Toggle between tree and flat (aggregated) profile view."""
        self._flat_mode = not self._flat_mode
        self._bottom_up_mode = False
        self._refresh_table()

    def toggle_bottom_up_mode(self) -> None:
        """Toggle bottom-up view (callees as roots, callers as children)."""
        self._bottom_up_mode = not self._bottom_up_mode
        self._flat_mode = False
        self._refresh_table()

    def cycle_sort(self) -> None:
        """Cycle through sort keys and refresh."""
        cycle = ["natural", "exclusive", "count", "mean", "name"]
        idx = cycle.index(self._sort_key) if self._sort_key in cycle else 0
        self._sort_key = cycle[(idx + 1) % len(cycle)]
        self._refresh_table()

    def find_matching_rows(self, name_filter: str) -> list[int]:
        """Return row indices matching the name filter (regex or substring)."""
        if not name_filter:
            return []
        return [
            i for i, (node, _path_key) in enumerate(self._row_map)
            if _name_matches(node.name, name_filter)
        ]

    def find_bookmarked_rows(self) -> list[int]:
        """Return row indices of bookmarked nodes."""
        return [
            i for i, (node, _path_key) in enumerate(self._row_map)
            if node.name in self._bookmarks
        ]


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class SaveDialog(ModalScreen[str | None]):
    """Modal dialog to choose a filename for saving."""

    CSS = """
    SaveDialog {
        align: center middle;
    }

    SaveDialog #dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $accent;
    }

    SaveDialog #dialog Label {
        width: 100%;
        margin-bottom: 1;
    }

    SaveDialog #dialog Input {
        width: 100%;
        margin-bottom: 1;
    }

    SaveDialog #dialog Horizontal {
        width: 100%;
        height: auto;
        align: right middle;
    }

    SaveDialog #dialog Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, title: str = "Save to file", default_name: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._default_name = default_name

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._title)
            yield Input(value=self._default_name, placeholder="filename.json", id="save-input")
            with Horizontal():
                yield Button("Save", variant="primary", id="save-ok")
                yield Button("Cancel", id="save-cancel")

    def on_mount(self) -> None:
        self.query_one("#save-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if value:
            self.dismiss(value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-ok":
            value = self.query_one("#save-input", Input).value.strip()
            if value:
                self.dismiss(value)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SaveAllDialog(ModalScreen[tuple[list[str], str] | None]):
    """Modal dialog to select sessions and choose a filename for saving."""

    CSS = """
    SaveAllDialog {
        align: center middle;
    }

    SaveAllDialog #dialog {
        width: 70;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $surface;
        border: thick $accent;
    }

    SaveAllDialog #dialog Label {
        width: 100%;
        margin-bottom: 1;
    }

    SaveAllDialog #dialog SelectionList {
        width: 100%;
        height: auto;
        max-height: 12;
        margin-bottom: 1;
    }

    SaveAllDialog #dialog Input {
        width: 100%;
        margin-bottom: 1;
    }

    SaveAllDialog #dialog .button-row {
        width: 100%;
        height: auto;
        align: right middle;
    }

    SaveAllDialog #dialog Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        sessions: list[tuple[str, str]],
        default_name: str = "einsums_all_sessions.json",
        **kwargs: Any,
    ) -> None:
        """sessions: list of (session_id, label) tuples."""
        super().__init__(**kwargs)
        self._sessions = sessions
        self._default_name = default_name

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Select sessions to save")
            yield SelectionList[str](
                *((label, sid, True) for sid, label in self._sessions),
                id="session-selection",
            )
            with Horizontal(classes="button-row"):
                yield Button("All", id="select-all")
                yield Button("None", id="select-none")
            yield Label("Filename")
            yield Input(value=self._default_name, placeholder="filename.json", id="save-input")
            with Horizontal(classes="button-row"):
                yield Button("Save", variant="primary", id="save-ok")
                yield Button("Cancel", id="save-cancel")

    def on_mount(self) -> None:
        self.query_one("#save-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._try_save()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-ok":
            self._try_save()
        elif event.button.id == "select-all":
            sel = self.query_one("#session-selection", SelectionList)
            sel.select_all()
        elif event.button.id == "select-none":
            sel = self.query_one("#session-selection", SelectionList)
            sel.deselect_all()
        elif event.button.id == "save-cancel":
            self.dismiss(None)

    def _try_save(self) -> None:
        sel = self.query_one("#session-selection", SelectionList)
        selected = list(sel.selected)
        path = self.query_one("#save-input", Input).value.strip()
        if not selected:
            self.notify("No sessions selected", severity="warning")
            return
        if not path:
            return
        self.dismiss((selected, path))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmDialog(ModalScreen[bool]):
    """Modal yes/no confirmation dialog."""

    CSS = """
    ConfirmDialog {
        align: center middle;
    }

    ConfirmDialog #dialog {
        width: 50;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $accent;
    }

    ConfirmDialog #dialog Label {
        width: 100%;
        margin-bottom: 1;
    }

    ConfirmDialog #dialog Horizontal {
        width: 100%;
        height: auto;
        align: right middle;
    }

    ConfirmDialog #dialog Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._message)
            with Horizontal():
                yield Button("Yes", variant="primary", id="confirm-yes")
                yield Button("No", id="confirm-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")

    def action_cancel(self) -> None:
        self.dismiss(False)


class HelpDialog(ModalScreen[None]):
    """Comprehensive help dialog explaining all keybindings, columns, and panels."""

    CSS = """
    HelpDialog {
        align: center middle;
    }

    HelpDialog #help-container {
        width: 80;
        height: 80%;
        padding: 1 2;
        background: $surface;
        border: thick $accent;
    }

    HelpDialog #help-scroll {
        height: 1fr;
    }

    HelpDialog #help-content {
        height: auto;
    }

    HelpDialog .help-heading {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("question_mark", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            yield Label("[bold]Einsums Profile Viewer — Help[/bold]")
            with VerticalScroll(id="help-scroll"):
                yield Static(self._help_text(), id="help-content")

    @staticmethod
    def _help_text() -> str:
        return """\
[bold underline]Keybindings[/bold underline]

[bold]Navigation & General[/bold]
  [bold cyan]q[/]           Quit
  [bold cyan]?[/]           Show this help dialog
  [bold cyan]/[/]           Filter nodes by name (regex supported)
  [bold cyan]Escape[/]      Clear active filter
  [bold cyan]n[/] / [bold cyan]N[/]       Jump to next / previous filter match
  [bold cyan]e[/]           Expand all tree nodes
  [bold cyan]w[/]           Collapse all tree nodes
  [bold cyan]r[/]           Force refresh the display

[bold]Sorting & Views[/bold]
  [bold cyan]s[/]           Cycle sort mode: natural → exclusive → count → mean → name
  [bold cyan]a[/]           Toggle flat (aggregated) view — merges all calls to the same
              function regardless of call site
  [bold cyan]u[/]           Toggle bottom-up view — shows callees with their callers
              as children (useful for finding who calls a hot function)
  [bold cyan]h[/]           Highlight the hot path (deepest exclusive-time chain)
  [bold cyan]H[/]           Toggle hotspots panel — flat list of top functions by
              exclusive time

[bold]Panels & Visualizations[/bold]
  [bold cyan]f[/]           Toggle flame graph (requires textual-plotext)
  [bold cyan]g[/]           Toggle timeline chart
  [bold cyan]G[/]           Toggle Gantt chart
  [bold cyan]o[/]           Toggle roofline model plot
  [bold cyan]V[/]           Toggle source code view for selected node
  [bold cyan]M[/]           Toggle micro-architecture analysis panel — shows IPC,
              cache/branch miss rates, and stall classification from
              hardware counters (Linux perf_event only)
  [bold cyan]L[/]           Toggle log message panel — streams EINSUMS_LOG_*
              messages in real-time from the profiler server
  [bold cyan]Shift+L[/]     Cycle log level filter (INFO -> WARN -> ERROR ->
              TRACE -> DEBUG)
  [bold cyan]A[/]           Toggle assembly/IR panel — tabbed view showing:
              • MLIR IR (post-pipeline dialect IR for JIT kernels)
              • LLVM IR (pre-JIT LLVM IR for JIT kernels)
              • Native assembly (disassembly of JIT kernels or executable
                functions via llvm-objdump)

[bold]Bookmarks & Annotations[/bold]
  [bold cyan]b[/]           Toggle bookmark on selected node
  [bold cyan]B[/]           Jump to next bookmarked node
  [bold cyan]y[/]           Copy selected node info to clipboard

[bold]Sessions & Connections[/bold]
  [bold cyan]t[/]           Connect to a new profiler server (host:port dialog)
  [bold cyan]c[/]           Retry failed connections
  [bold cyan]D[/]           Toggle listen mode — accept incoming connections
              instead of connecting out
  [bold cyan]p[/]           Pause / resume live updates
  [bold cyan]R[/]           Toggle recording incoming snapshots to a .jsonl file
  [bold cyan]+[/] / [bold cyan]-[/]       Speed up / slow down replay playback
  [bold cyan]S[/]           Save current session to a JSON file
  [bold cyan]Ctrl+S[/]      Save all sessions to a JSON file
  [bold cyan]C[/]           Compare two sessions side-by-side (need ≥2 loaded)

[bold]Diff & Export[/bold]
  [bold cyan]d[/]           Diff current snapshot against previous (shows delta)
  [bold cyan]x[/]           Export profile data to JSON file
  [bold cyan]T[/]           Cycle color theme: heat → cool → mono

[bold underline]Table Columns[/bold underline]

  [bold]Column[/]      [bold]Description[/]
  (indent)    Tree depth indicator (arrows show expand/collapse state)
  [bold]%[/]           Percentage of total exclusive time across all top-level nodes
  [bold]% parent[/]    Percentage of parent node's inclusive time
  [bold]bar[/]         Visual bar showing relative exclusive time
  [bold]excl(ms)[/]    Exclusive time — time spent in this function only (not
              in children)
  [bold]incl(ms)[/]    Inclusive time — total time including all children
  [bold]count[/]       Number of times this function was called
  [bold]mean(ms)[/]    Mean exclusive time per call (excl / count)
  [bold]alloc[/]       Total memory allocated in this zone (bytes)
  [bold]peak[/]        Peak (net) memory usage in this zone (alloc - free)
  [bold]⚑[/]           Bookmark flag — shows ⚑ if this node is bookmarked
  [bold]anom[/]        Anomaly flag — shows ! if this node has unusually high
              variance (coefficient of variation > 0.5)
  [bold]name[/]        Function/zone name (with tree indentation)

[bold underline]Micro-Architecture Panel (M)[/bold underline]

Computes derived metrics from hardware performance counters:
  • [bold]IPC[/] (Instructions Per Cycle) — higher is better
  • [bold]Cache miss rate[/] per 1K instructions
  • [bold]Branch mispredict rate[/] per 1K instructions

Stall classification:
  [green]Retiring[/]            IPC ≥ 2.0 (CPU is doing useful work)
  [cyan]Compute Bound[/]       IPC 0.5–2.0 (limited by execution units)
  [magenta]Front-end Bound[/]     IPC < 0.5, low cache misses
  [yellow]Back-end Bound[/]      High cache miss rate (memory stalls)
  [red]Bad Speculation[/]     High branch mispredict rate

Note: Hardware counters require Linux perf_event. On macOS, this
panel shows "(no hardware counters available)".

[bold underline]Assembly Panel (A)[/bold underline]

For [bold]JIT kernel nodes[/] (annotated with jit_kernel_hash):
  Requests MLIR IR, LLVM IR, and native disassembly from the
  running profiler server. Requires a live connection.

For [bold]other functions[/]:
  Runs llvm-objdump locally on the executable and its shared
  libraries (e.g. libEinsums.dylib) to find the function's
  native disassembly.

[bold underline]Command Line[/bold underline]

  profile_viewer.py [bold]--port[/] PORT     Connect to profiler (default: 19216)
  profile_viewer.py [bold]--load[/] FILE     Load saved session JSON
  profile_viewer.py [bold]--replay[/] FILE   Replay recorded .jsonl
  profile_viewer.py [bold]--record[/] FILE   Record snapshots to .jsonl
  profile_viewer.py [bold]--speed[/] N       Playback speed multiplier
  profile_viewer.py [bold]--connect[/] H:P   Connect to additional host:port
  profile_viewer.py [bold]--no-mdns[/]       Disable auto-discovery

[dim]Press Escape or ? to close this dialog.[/dim]"""

    def action_close(self) -> None:
        self.dismiss(None)


class CompareScreen(ModalScreen[None]):
    """Side-by-side comparison of two sessions' flat profiles."""

    CSS = """
    CompareScreen {
        align: center middle;
    }

    CompareScreen #compare-container {
        width: 90%;
        height: 80%;
        background: $surface;
        border: thick $accent;
        padding: 1;
    }

    CompareScreen #compare-container Label {
        width: 100%;
        margin-bottom: 1;
    }

    CompareScreen #compare-container DataTable {
        height: 1fr;
    }

    CompareScreen #compare-container .button-row {
        width: 100%;
        height: auto;
        align: right middle;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    def __init__(self, session_a: "SessionState", session_b: "SessionState", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._session_a = session_a
        self._session_b = session_b

    def compose(self) -> ComposeResult:
        with Vertical(id="compare-container"):
            yield Label(f"[bold]Compare:[/bold] {self._session_a.label}  vs  {self._session_b.label}")
            yield DataTable(id="compare-table")
            with Horizontal(classes="button-row"):
                yield Button("Close", id="compare-close")

    def on_mount(self) -> None:
        table = self.query_one("#compare-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("name", "excl A(ms)", "excl B(ms)", "delta(ms)", "delta%", "calls A", "calls B", "delta calls")

        # Flat-aggregate both sessions
        nodes_a: list[ProfileNode] = []
        nodes_b: list[ProfileNode] = []
        snap_a = self._session_a.latest_snapshot
        snap_b = self._session_b.latest_snapshot
        if snap_a:
            for tdata in snap_a.threads.values():
                nodes_a.extend(tdata.children)
        if snap_b:
            for tdata in snap_b.threads.values():
                nodes_b.extend(tdata.children)

        flat_a = {n.name: n for n in _aggregate_flat(nodes_a)}
        flat_b = {n.name: n for n in _aggregate_flat(nodes_b)}
        all_names = set(flat_a.keys()) | set(flat_b.keys())

        rows: list[tuple[str, float, float, float, float, int, int, int]] = []
        for name in all_names:
            na = flat_a.get(name)
            nb = flat_b.get(name)
            excl_a = na.exclusive_ms if na else 0.0
            excl_b = nb.exclusive_ms if nb else 0.0
            calls_a = na.call_count if na else 0
            calls_b = nb.call_count if nb else 0
            delta = excl_b - excl_a
            delta_pct = (delta / excl_a * 100.0) if excl_a > 0 else 0.0
            rows.append((name, excl_a, excl_b, delta, delta_pct, calls_a, calls_b, calls_b - calls_a))

        # Sort by absolute delta descending
        rows.sort(key=lambda r: abs(r[3]), reverse=True)

        for name, excl_a, excl_b, delta, delta_pct, calls_a, calls_b, delta_calls in rows:
            # Color-code delta
            if delta < -0.01:
                delta_str = f"[green]{delta:+.3f}[/green]"
                pct_str = f"[green]{delta_pct:+.1f}%[/green]"
            elif delta > 0.01:
                delta_str = f"[red]{delta:+.3f}[/red]"
                pct_str = f"[red]{delta_pct:+.1f}%[/red]"
            else:
                delta_str = f"{delta:+.3f}"
                pct_str = f"{delta_pct:+.1f}%"
            dc_str = f"{delta_calls:+d}" if delta_calls != 0 else "0"
            table.add_row(name, f"{excl_a:.3f}", f"{excl_b:.3f}", delta_str, pct_str,
                          str(calls_a), str(calls_b), dc_str)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class SessionPickerDialog(ModalScreen[tuple[str, str] | None]):
    """Pick two sessions to compare."""

    CSS = """
    SessionPickerDialog {
        align: center middle;
    }

    SessionPickerDialog #dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $accent;
    }

    SessionPickerDialog #dialog Label {
        width: 100%;
        margin-bottom: 1;
    }

    SessionPickerDialog #dialog SelectionList {
        width: 100%;
        height: auto;
        max-height: 12;
        margin-bottom: 1;
    }

    SessionPickerDialog #dialog Horizontal {
        width: 100%;
        height: auto;
        align: right middle;
    }

    SessionPickerDialog #dialog Button {
        margin-left: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, sessions: list[tuple[str, str]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Select exactly 2 sessions to compare")
            yield SelectionList[str](
                *((label, sid, False) for sid, label in self._sessions),
                id="compare-selection",
            )
            with Horizontal():
                yield Button("Compare", variant="primary", id="compare-ok")
                yield Button("Cancel", id="compare-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "compare-ok":
            sel = self.query_one("#compare-selection", SelectionList)
            selected = list(sel.selected)
            if len(selected) != 2:
                self.notify("Select exactly 2 sessions", severity="warning")
                return
            self.dismiss((selected[0], selected[1]))
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConnectDialog(ModalScreen[str | None]):
    """Modal dialog to enter a host:port to connect to."""

    CSS = """
    ConnectDialog {
        align: center middle;
    }

    ConnectDialog #dialog {
        width: 50;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $accent;
    }

    ConnectDialog #dialog Label {
        width: 100%;
        margin-bottom: 1;
    }

    ConnectDialog #dialog Input {
        width: 100%;
        margin-bottom: 1;
    }

    ConnectDialog #dialog Horizontal {
        width: 100%;
        height: auto;
        align: right middle;
    }

    ConnectDialog #dialog Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Connect to profiler server")
            yield Input(placeholder="host:port  (e.g. 127.0.0.1:19217 or just 19217)", id="connect-input")
            with Horizontal():
                yield Button("Connect", variant="primary", id="connect-ok")
                yield Button("Cancel", id="connect-cancel")

    def on_mount(self) -> None:
        self.query_one("#connect-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "connect-ok":
            self.dismiss(self.query_one("#connect-input", Input).value.strip())
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


@dataclass
class SessionState:
    """Per-session state for one connected program run."""
    session_id: str
    label: str
    meta: ProfileMeta | None = None
    latest_snapshot: ProfileSnapshot | None = None
    thread_views: dict[str, ThreadTreeView] = field(default_factory=dict)
    node_history: dict[str, deque] = field(default_factory=dict)
    baseline_snapshot: ProfileSnapshot | None = None
    bookmarks: set[str] = field(default_factory=set)  # bookmarked node names
    timeline_events: list[TimelineEvent] = field(default_factory=list)
    client: ProfileClient | None = None  # associated client (for log entries etc.)


class ProfileViewerApp(App):
    """Einsums Profile Viewer -- real-time TUI."""

    TITLE = "Einsums Profile Viewer"

    CSS = """
    Screen {
        layout: vertical;
    }

    #status-bar {
        dock: bottom;
        height: 3;
        padding: 0 1;
        background: $surface;
        border-top: solid $primary;
    }

    #filter-input {
        dock: bottom;
        display: none;
    }

    #filter-input.visible {
        display: block;
    }


    TabbedContent {
        height: 1fr;
    }

    DataTable {
        height: 1fr;
    }

    #log-resize {
        display: none;
    }

    #hotspots-panel {
        display: none;
    }

    #hotspots-panel.visible {
        display: block;
    }

    #source-panel {
        display: none;
    }

    #source-panel.visible {
        display: block;
    }

    #gantt-panel {
        display: none;
    }

    #gantt-panel.visible {
        display: block;
    }

    #log-panel {
        display: none;
    }

    #microarch-panel {
        display: none;
    }

    #microarch-panel.visible {
        display: block;
    }

    #assembly-panel {
        display: none;
    }

    #assembly-panel.visible {
        display: block;
    }

    #timeline-panel {
        display: none;
    }

    #timeline-panel.visible {
        display: block;
    }

    """

    BINDINGS = [
        # Shown in footer
        Binding("q", "quit", "Quit", priority=True),
        Binding("slash", "filter", "Filter"),
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("t", "add_connection", "Connect", priority=True),
        # Available via key + command palette (ctrl+p)
        Binding("r", "refresh", "Refresh", show=False),
        Binding("c", "connect", "Retry connection", show=False),
        Binding("e", "expand_all", "Expand all", show=False),
        Binding("w", "collapse_all", "Collapse all", show=False),
        Binding("escape", "clear_filter", "Clear filter", show=False),
        Binding("x", "export_json", "Export JSON", show=False),
        Binding("g", "toggle_timeline", "Timeline", show=False),
        Binding("n", "search_next", "Next match", show=False),
        Binding("N", "search_prev", "Prev match", show=False),
        Binding("h", "toggle_hotpath", "Hot path", show=False),
        Binding("y", "copy_node", "Copy", show=False),
        Binding("d", "diff_snapshot", "Diff", show=False),
        Binding("f", "toggle_flamegraph", "Flame", show=False),
        Binding("a", "toggle_flat", "Flat", show=False),
        Binding("o", "toggle_roofline", "Roofline", show=False),
        Binding("b", "toggle_bookmark", "Bookmark", show=False),
        Binding("B", "next_bookmark", "Next bookmark", show=False),
        Binding("S", "save_session", "Save session", show=False),
        Binding("ctrl+s", "save_all_sessions", "Save all sessions", show=False),
        Binding("T", "cycle_color_theme", "Color theme", show=False),
        Binding("H", "toggle_hotspots", "Hotspots", show=False),
        Binding("u", "toggle_bottom_up", "Bottom-up", show=False),
        Binding("V", "toggle_source", "Source", show=False),
        Binding("C", "compare_sessions", "Compare", show=False),
        Binding("D", "toggle_listen", "Listen mode", show=False),
        Binding("R", "toggle_record", "Record", show=False),
        Binding("G", "toggle_gantt", "Gantt", show=False),
        Binding("M", "toggle_microarch", "MicroArch", show=False),
        Binding("A", "toggle_assembly", "Assembly", show=False),
        Binding("L", "toggle_logs", "Logs", show=False),
        Binding("shift+l", "cycle_log_level", "Log Level", show=False),
        Binding("plus_sign", "speed_up", "Speed+", show=False),
        Binding("hyphen_minus", "speed_down", "Speed-", show=False),
        Binding("question_mark", "show_help", "Help"),
    ]

    def __init__(
        self,
        endpoints: list[tuple[str, int]] | None = None,
        record_path: str | None = None,
        replay_path: str | None = None,
        replay_speed: float = 1.0,
        mdns: bool = True,
        load_paths: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._endpoints: list[tuple[str, int]] = endpoints or []
        self._clients: list[ProfileClient] = []
        self._name_filter = ""
        self._paused: bool = False
        # Multi-session state
        self._sessions: dict[str, SessionState] = {}
        self._active_session_id: str | None = None
        self._session_counter: int = 0
        # Recording
        self._record_path: str | None = record_path
        self._record_file: io.TextIOWrapper | None = None
        self._recording: bool = record_path is not None
        # Replay
        self._replay_path: str | None = replay_path
        self._replay_speed: float = max(0.1, replay_speed)
        # mDNS/Bonjour discovery (disabled by default when loading saved sessions)
        self._mdns_enabled: bool = mdns and HAS_ZEROCONF and not load_paths
        self._zeroconf: Any = None
        self._mdns_browser: Any = None
        self._known_endpoints: set[tuple[str, int]] = set()  # all endpoints ever started
        # Saved sessions to load
        self._load_paths: list[str] = load_paths or []

    def compose(self) -> ComposeResult:
        yield Header()
        yield HotspotsPanel(id="hotspots-panel")
        yield TabbedContent(id="session-tabs")
        yield FlameGraphPanel(id="flamegraph-panel")
        if HAS_PLOTEXT:
            yield TimelinePanel(id="timeline-panel")
            yield RooflinePanel(id="roofline-panel")
        yield GanttPanel(id="gantt-panel")
        yield MicroArchPanel(id="microarch-panel")
        yield AssemblyPanel(id="assembly-panel")
        yield SourcePanel(id="source-panel")
        yield ResizeHandle("detail-panel", id="detail-resize")
        yield DetailPanel(id="detail-panel")
        yield ResizeHandle("log-panel", id="log-resize")
        yield LogPanel(id="log-panel")
        yield Input(placeholder="Filter by name (regex supported)...", id="filter-input")
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        # Hide optional panels initially
        try:
            self.query_one("#flamegraph-panel", FlameGraphPanel).display = False
        except Exception:
            pass
        if HAS_PLOTEXT:
            try:
                self.query_one("#roofline-panel", RooflinePanel).display = False
            except Exception:
                pass
        # Open recording file if --record was given on the command line
        if self._record_path and self._recording:
            self._record_file = open(self._record_path, "w")
            self.notify(f"Recording to {self._record_path}")
        # Load saved session files
        for path in self._load_paths:
            try:
                self._load_session_file(path)
            except Exception as exc:
                self.notify(f"Failed to load {path}: {exc}", severity="error")
        # Choose replay or live connection loop
        if self._replay_path:
            self.run_worker(self._replay_loop(), exclusive=True, name="replay")
        else:
            # Connect to explicitly requested endpoints
            for host, port in self._endpoints:
                self._start_client(host, port)
            # Start mDNS browser to auto-discover profiler servers
            if self._mdns_enabled:
                self._start_mdns_browser()
            elif not self._endpoints and not self._load_paths:
                # No mDNS, no explicit endpoints, no loaded files — fall back to default
                self._start_client("127.0.0.1", 19216)

    def _create_session(self, meta: ProfileMeta | None = None) -> SessionState:
        """Create a new session and add its tab to the outer TabbedContent."""
        self._session_counter += 1
        sid = f"s{self._session_counter}"
        if meta and (meta.executable or meta.start_time):
            exe = meta.executable or "unknown"
            t = meta.start_time or "?"
            label = f"{exe} ({t})"
        else:
            label = f"Session {self._session_counter}"

        session = SessionState(session_id=sid, label=label, meta=meta)
        self._sessions[sid] = session
        self._active_session_id = sid

        # Add outer tab with inner TabbedContent for threads
        session_tabs = self.query_one("#session-tabs", TabbedContent)
        inner_tabs = TabbedContent(id=f"threads-{sid}")
        pane = TabPane(label, id=f"session-tab-{sid}")
        pane.compose_add_child(inner_tabs)
        session_tabs.add_pane(pane)
        # Activate the new session tab
        session_tabs.active = f"session-tab-{sid}"

        return session

    def _load_session_file(self, path: str) -> None:
        """Load a saved session JSON file (single or multi-session)."""
        with open(path, "r") as f:
            data = json.load(f)

        # Multi-session file has a "sessions" key
        if "sessions" in data:
            for sdata in data["sessions"]:
                self._load_session_data(sdata, path)
        else:
            # Single session file
            self._load_session_data(data, path)

    def _load_session_data(self, data: dict[str, Any], source_path: str) -> None:
        """Load a single session from a parsed dict."""
        meta_data = data.get("meta", {})
        meta = ProfileMeta(
            pid=meta_data.get("pid", 0),
            hostname=meta_data.get("hostname", ""),
            executable=meta_data.get("executable", ""),
            start_time=meta_data.get("start_time", ""),
            counters=meta_data.get("counters", []),
            executable_path=meta_data.get("executable_path", ""),
        )
        session = self._create_session(meta)
        label = data.get("label")
        if label:
            session.label = label
            # Update the tab label
            try:
                pane = self.query_one(f"#session-tab-{session.session_id}", TabPane)
                pane.label = label
            except Exception:
                pass

        # Restore bookmarks
        for bm in data.get("bookmarks", []):
            session.bookmarks.add(bm)

        # Restore snapshot
        snap_data = data.get("snapshot")
        if snap_data:
            session.latest_snapshot = parse_snapshot(snap_data)

        # Restore node history
        for name, values in data.get("node_history", {}).items():
            session.node_history[name] = deque(values, maxlen=20)

        # Populate display
        self._update_display(session)
        self.notify(f"Loaded session from {source_path}")

    def _start_client(self, host: str, port: int) -> ProfileClient:
        """Create a ProfileClient and start its connection loop worker."""
        client = ProfileClient(host, port)
        self._clients.append(client)
        self._known_endpoints.add((host, port))
        self.run_worker(self._connection_loop(client), exclusive=False, name=f"net-{client.endpoint}")
        return client

    def _start_mdns_browser(self) -> None:
        """Start browsing for _einsums-profile._tcp services via mDNS/Bonjour."""
        if not HAS_ZEROCONF:
            return
        self._zeroconf = Zeroconf()

        def on_service_state_change(zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange) -> None:
            if state_change in (ServiceStateChange.Added, ServiceStateChange.Updated):
                info = zeroconf.get_service_info(service_type, name)
                if info is None:
                    return
                # Extract host and port
                addresses = info.parsed_addresses()
                if not addresses:
                    return
                host = addresses[0]
                port = info.port
                # Extract TXT record info for the notification
                props = {k.decode(): v.decode() if v else "" for k, v in info.properties.items()}
                exe = props.get("exe", "unknown")
                # Schedule connection on the main event loop (zeroconf callback is on a background thread)
                self.call_from_thread(self._on_mdns_service_found, host, port, exe)

        self._mdns_browser = ServiceBrowser(self._zeroconf, "_einsums-profile._tcp.local.", handlers=[on_service_state_change])
        self.notify("mDNS: browsing for profiler servers...")

    def _on_mdns_service_found(self, host: str, port: int, exe: str) -> None:
        """Called on the main thread when an mDNS service is discovered."""
        # Normalize IPv6 localhost to IPv4
        if host in ("::1", "0:0:0:0:0:0:0:1"):
            host = "127.0.0.1"
        # Check if we already have a connection to the same port (any address).
        # The C++ server binds to 127.0.0.1 but Bonjour advertises on all
        # interfaces, so we may see the same service on multiple addresses.
        for h, p in self._known_endpoints:
            if p == port:
                return
        # Prefer connecting via localhost when the server is local
        if host != "127.0.0.1":
            # Check if this is the local machine (same host advertised on a LAN interface)
            try:
                local_addrs = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None)}
            except OSError:
                local_addrs = set()
            if host in local_addrs:
                host = "127.0.0.1"
        self.notify(f"mDNS: discovered {exe} at {host}:{port}")
        self._start_client(host, port)

    def _stop_mdns_browser(self) -> None:
        """Stop the mDNS browser and close zeroconf."""
        try:
            if self._mdns_browser is not None:
                self._mdns_browser.cancel()
                self._mdns_browser = None
            if self._zeroconf is not None:
                self._zeroconf.close()
                self._zeroconf = None
        except Exception:
            pass

    def _get_active_session(self) -> SessionState | None:
        """Get the session state for the currently active session tab."""
        if self._active_session_id and self._active_session_id in self._sessions:
            return self._sessions[self._active_session_id]
        return None

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Track which session is active when outer tabs change."""
        pane_id = event.pane.id or ""
        if pane_id.startswith("session-tab-"):
            sid = pane_id.removeprefix("session-tab-")
            if sid in self._sessions:
                self._active_session_id = sid
                self._refresh_detail_panel()

    async def _connection_loop(self, client: ProfileClient) -> None:
        """Background worker: connect, read, and update UI for one client."""
        current_session: SessionState | None = None
        ep = client.endpoint

        while True:
            if not client.connected:
                self._update_status(f"Connecting to {ep}...")
                ok = await client.connect()
                if not ok:
                    if self._mdns_enabled:
                        # mDNS: server may still be starting up; retry a few times then give up
                        if client._reconnect_delay > 4.0:
                            if client in self._clients:
                                self._clients.remove(client)
                            self._known_endpoints.discard((client.host, client.port))
                            self._update_status(f"{ep}: connect failed (waiting for mDNS)")
                            return
                        # Brief wait before retry
                        self._update_status(f"{ep}: connecting... (attempt in {client._reconnect_delay:.0f}s)")
                        await asyncio.sleep(client._reconnect_delay)
                        client._reconnect_delay = min(client._reconnect_delay * 2, 8.0)
                        continue
                    delay = client._reconnect_delay
                    remaining = int(delay)
                    while remaining > 0:
                        self._update_status(
                            f"{ep}: disconnected -- retrying in {remaining}s (press 'c' to retry now)"
                        )
                        try:
                            await asyncio.wait_for(client.retry_event.wait(), timeout=1)
                            client.retry_event.clear()
                            client._reconnect_delay = 1.0
                            break
                        except asyncio.TimeoutError:
                            remaining -= 1
                    else:
                        client._reconnect_delay = min(delay * 2, 30.0)
                    continue
                self._update_status(f"Connected to {ep}")
                # New connection → new session will be created when we get the meta message
                current_session = None

            messages = await client.read_messages()
            if not client.connected:
                await client.disconnect()
                if client in self._clients:
                    self._clients.remove(client)
                self._known_endpoints.discard((client.host, client.port))
                if self._mdns_enabled:
                    self._update_status(f"{ep}: disconnected (waiting for mDNS)")
                else:
                    self._update_status(f"{ep}: disconnected")
                return

            if messages and self._record_file:
                wall = time.monotonic()
                for msg in messages:
                    record = {"_ts": wall, "_msg": msg}
                    self._record_file.write(json.dumps(record) + "\n")
                self._record_file.flush()

            # Check for meta message → create new session
            for msg in messages:
                if msg.get("type") == "meta":
                    meta = ProfileMeta(
                        pid=msg.get("pid", 0),
                        hostname=msg.get("hostname", ""),
                        executable=msg.get("executable", ""),
                        start_time=msg.get("start_time", ""),
                        counters=msg.get("counters", []),
                        executable_path=msg.get("executable_path", ""),
                    )
                    client.meta = meta
                    current_session = self._create_session(meta)
                    current_session.client = client

            if client.process_messages(messages):
                # Only update display if we have a session (i.e. received a meta message)
                if current_session and client.latest_snapshot:
                    current_session.latest_snapshot = client.latest_snapshot
                    if client.latest_timeline:
                        current_session.timeline_events = client.latest_timeline
                    self._update_display(current_session)

    async def _replay_loop(self) -> None:
        """Background worker: replay a recorded .jsonl file."""
        path = self._replay_path
        if not path:
            return

        self._update_status(f"Loading {path}...")

        # Load all recorded entries
        entries: list[tuple[float, dict]] = []
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    ts = record.get("_ts", 0.0)
                    msg = record.get("_msg", {})
                    entries.append((ts, msg))
        except (OSError, json.JSONDecodeError) as exc:
            self._update_status(f"Replay error: {exc}")
            return

        if not entries:
            self._update_status("Replay file is empty")
            return

        total = len(entries)
        self._update_status(f"Replaying {total} messages from {path}")

        base_ts = entries[0][0]
        idx = 0
        current_session: SessionState | None = None
        # Dummy client for replay state
        replay_client = ProfileClient("replay", 0)

        while idx < total:
            # Handle pause
            while self._paused:
                await asyncio.sleep(0.1)

            ts, msg = entries[idx]

            # Wait for the appropriate delay (scaled by speed)
            if idx > 0:
                prev_ts = entries[idx - 1][0]
                delay = (ts - prev_ts) / self._replay_speed
                if delay > 0:
                    await asyncio.sleep(delay)

            # Check for meta → new session
            if msg.get("type") == "meta":
                meta = ProfileMeta(
                    pid=msg.get("pid", 0),
                    hostname=msg.get("hostname", ""),
                    executable=msg.get("executable", ""),
                    start_time=msg.get("start_time", ""),
                    counters=msg.get("counters", []),
                    executable_path=msg.get("executable_path", ""),
                )
                replay_client.meta = meta
                current_session = self._create_session(meta)

            # Process the message
            replay_client.process_messages([msg])
            # Only update display if we have a session (i.e. received a meta message)
            if current_session is not None:
                if replay_client.latest_snapshot:
                    current_session.latest_snapshot = replay_client.latest_snapshot
                self._update_display(current_session)

            progress = idx + 1
            elapsed = ts - base_ts
            speed_str = f"{self._replay_speed:.1f}x"
            self._update_status(
                f"Replay [{progress}/{total}] t={elapsed:.1f}s speed={speed_str}"
            )

            idx += 1

        self._update_status(f"Replay complete ({total} messages)")

    def _update_status(self, text: str) -> None:
        session = self._get_active_session()
        meta = session.meta if session else None
        pid_str = f" (PID {meta.pid})" if meta and meta.pid else ""
        snap = session.latest_snapshot if session else None
        dropped = snap.dropped if snap else 0
        dropped_str = f"  dropped={dropped}" if dropped > 0 else ""
        paused_str = "PAUSED  " if self._paused else ""
        diff_str = "  [DIFF]" if session and session.baseline_snapshot is not None else ""
        rec_str = "  [REC]" if self._recording else ""
        listen_str = "" if self._mdns_enabled else "  [NO LISTEN]"
        num_sessions = len(self._sessions)
        sessions_str = f"  [{num_sessions} session{'s' if num_sessions != 1 else ''}]" if num_sessions > 1 else ""
        num_clients = len(self._clients)
        clients_str = f"  [{num_clients} server{'s' if num_clients != 1 else ''}]" if num_clients > 1 else ""
        full = f"{paused_str}{text}{pid_str}{dropped_str}{diff_str}{rec_str}{listen_str}{sessions_str}{clients_str}"
        try:
            bar = self.query_one("#status-bar", StatusBar)
            bar.status_text = full
        except Exception:
            pass

    def _update_node_history(self, session: SessionState, snap: ProfileSnapshot) -> None:
        """Walk all nodes in snapshot and record exclusive_ms history."""
        node_values: dict[str, float] = {}
        for tdata in snap.threads.values():
            _collect_node_exclusive_ms(tdata.children, node_values)
        for name, excl_ms in node_values.items():
            if name not in session.node_history:
                session.node_history[name] = deque(maxlen=20)
            session.node_history[name].append(excl_ms)

    def _update_display(self, session: SessionState | None = None) -> None:
        """Update the tree display from the latest snapshot."""
        if self._paused:
            return

        if session is None:
            session = self._get_active_session()
        if session is None:
            return

        snap = session.latest_snapshot
        if not snap:
            return

        # Update per-node history for sparklines
        self._update_node_history(session, snap)

        # Get inner thread tabs for this session
        try:
            thread_tabs = self.query_one(f"#threads-{session.session_id}", TabbedContent)
        except Exception:
            return

        for tid, tdata in snap.threads.items():
            view_id = f"{session.session_id}-{tid}"
            if view_id not in session.thread_views:
                tab_label = tdata.thread_name if tdata.thread_name else f"Thread {tid}"
                view = ThreadTreeView(tid, id=f"view-{view_id}")
                view._bookmarks = session.bookmarks  # share bookmark set
                pane = TabPane(tab_label, id=f"tab-{view_id}")
                pane.compose_add_child(view)
                thread_tabs.add_pane(pane)
                session.thread_views[view_id] = view
            else:
                # Update tab label if thread name becomes available
                if tdata.thread_name:
                    try:
                        pane = self.query_one(f"#tab-{view_id}", TabPane)
                        current_label = str(pane.label) if hasattr(pane, 'label') else ""
                        if current_label != tdata.thread_name:
                            pane.label = tdata.thread_name
                    except Exception:
                        pass

            session.thread_views[view_id].update_data(tdata.children, self._name_filter)

        # Feed hotspots panel if visible
        try:
            hotspots = self.query_one("#hotspots-panel", HotspotsPanel)
            if hotspots.display:
                hotspots.update_hotspots(self._sessions)
        except Exception:
            pass

        # Feed Gantt chart if visible
        try:
            gantt = self.query_one("#gantt-panel", GanttPanel)
            if gantt.display and session.timeline_events:
                gantt.update_events(session.timeline_events)
        except Exception:
            pass

        # Feed flamegraph panel (use the active thread's data)
        try:
            fg = self.query_one("#flamegraph-panel", FlameGraphPanel)
            if fg.display:
                view = self._get_active_tree_view()
                if view and view._data:
                    fg.update_data(view._data)
        except Exception:
            pass

        # Feed timeline panel
        if HAS_PLOTEXT:
            try:
                timeline = self.query_one("#timeline-panel", TimelinePanel)
                timeline.record(snap)
            except Exception:
                pass

            # Feed roofline panel
            try:
                roof = self.query_one("#roofline-panel", RooflinePanel)
                if roof.display:
                    view = self._get_active_tree_view()
                    if view and view._data:
                        roof.update_data(view._data)
            except Exception:
                pass

        # Feed log panel if visible
        try:
            log_panel = self.query_one("#log-panel", LogPanel)
            if log_panel.display and session.client:
                log_panel.update_logs(session.client.log_entries)
        except Exception:
            pass

    def _get_active_tree_view(self) -> ThreadTreeView | None:
        """Get the ThreadTreeView in the currently active tab."""
        session = self._get_active_session()
        if not session:
            return None
        for view in session.thread_views.values():
            try:
                table = view.query_one(ProfileDataTable)
                if table.is_attached and table.display:
                    return view
            except Exception:
                pass
        return None

    def _refresh_detail_panel(self) -> None:
        """Re-populate the detail panel for the current cursor position."""
        view = self._get_active_tree_view()
        if not view:
            return
        session = self._get_active_session()
        panel = self.query_one("#detail-panel", DetailPanel)
        try:
            table = view.query_one(ProfileDataTable)
            node = view.get_node_at_row(table.cursor_row)
            if node:
                pct = (node.exclusive_ms / view._total_ms * 100.0) if view._total_ms > 0 else 0.0
                sparkline = ""
                node_history = session.node_history if session else {}
                if node.name in node_history:
                    sparkline = _make_sparkline(list(node_history[node.name]))
                bm = session.bookmarks if session else set()
                panel.update_node(node, pct, baseline_snap=session.baseline_snapshot if session else None, sparkline=sparkline, is_bookmarked=node.name in bm)
                try:
                    microarch_panel = self.query_one("#microarch-panel", MicroArchPanel)
                    if microarch_panel.display:
                        microarch_panel.update_node(node)
                except Exception:
                    pass
        except Exception:
            pass

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update detail panel when a row is highlighted."""
        view = self._get_active_tree_view()
        session = self._get_active_session()
        panel = self.query_one("#detail-panel", DetailPanel)
        if view and event.cursor_row is not None:
            node = view.get_node_at_row(event.cursor_row)
            if node:
                pct = (node.exclusive_ms / view._total_ms * 100.0) if view._total_ms > 0 else 0.0
                sparkline = ""
                node_history = session.node_history if session else {}
                if node.name in node_history:
                    sparkline = _make_sparkline(list(node_history[node.name]))
                baseline = session.baseline_snapshot if session else None
                bm = session.bookmarks if session else set()
                panel.update_node(node, pct, baseline_snap=baseline, sparkline=sparkline, is_bookmarked=node.name in bm)
                # Update source panel if visible
                try:
                    source = self.query_one("#source-panel", SourcePanel)
                    if source.display:
                        source.update_node(node)
                except Exception:
                    pass
                # Update micro-architecture panel if visible
                try:
                    microarch_panel = self.query_one("#microarch-panel", MicroArchPanel)
                    if microarch_panel.display:
                        microarch_panel.update_node(node)
                except Exception:
                    pass
                return
        panel.update_node(None)

    def on_profile_data_table_toggle_request(self, event: ProfileDataTable.ToggleRequest) -> None:
        """Toggle expand/collapse when space/enter/click on a row."""
        view = self._get_active_tree_view()
        if view:
            view.toggle_row(event.row)

    def action_toggle_node(self) -> None:
        """Toggle expand/collapse on the currently selected row."""
        view = self._get_active_tree_view()
        if not view:
            return
        try:
            table = view.query_one(ProfileDataTable)
            view.toggle_row(table.cursor_row)
        except Exception:
            pass

    def action_show_help(self) -> None:
        """Show the help dialog."""
        self.push_screen(HelpDialog())

    def action_quit(self) -> None:
        self._stop_mdns_browser()
        self.exit()

    def action_connect(self) -> None:
        for client in self._clients:
            client.retry_event.set()

    def action_add_connection(self) -> None:
        """Show modal dialog to add a new server connection."""
        self.push_screen(ConnectDialog(), self._on_connect_dialog_result)

    def _on_connect_dialog_result(self, value: str | None) -> None:
        """Handle the result from the connect dialog."""
        if not value:
            return
        # Parse host:port
        if ":" in value:
            host, port_str = value.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                self.notify(f"Invalid port: {port_str}", severity="error")
                return
        else:
            try:
                port = int(value)
                host = "127.0.0.1"
            except ValueError:
                self.notify(f"Expected host:port or port number, got: {value}", severity="error")
                return
        # Check for duplicate
        for c in self._clients:
            if c.host == host and c.port == port:
                self.notify(f"Already connected to {host}:{port}", severity="warning")
                return
        self._start_client(host, port)
        self.notify(f"Connecting to {host}:{port}...")

    def action_refresh(self) -> None:
        self._update_display()

    def action_expand_all(self) -> None:
        view = self._get_active_tree_view()
        if view:
            view.expand_all()

    def action_collapse_all(self) -> None:
        view = self._get_active_tree_view()
        if view:
            view.collapse_all()

    def action_filter(self) -> None:
        inp = self.query_one("#filter-input", Input)
        inp.add_class("visible")
        inp.focus()

    def action_clear_filter(self) -> None:
        self._name_filter = ""
        self.query_one("#filter-input", Input).remove_class("visible")
        self._update_display()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter-input":
            self._name_filter = event.value
            self.query_one("#filter-input", Input).remove_class("visible")
            self._update_display()

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            self._update_status("PAUSED")
        else:
            self._update_status("Connected")
            self._update_display()

    def action_cycle_sort(self) -> None:
        view = self._get_active_tree_view()
        if view:
            view.cycle_sort()
            self.notify(f"Sort: {view._sort_key}")

    def action_toggle_timeline(self) -> None:
        if not HAS_PLOTEXT:
            self.notify("Install textual-plotext for timeline graphs: pip install textual-plotext", severity="warning")
            return
        try:
            panel = self.query_one("#timeline-panel")
            panel.toggle_class("visible")
        except Exception:
            pass

    def action_search_next(self) -> None:
        """Move cursor to the next row matching the current filter."""
        if not self._name_filter:
            self.notify("No active filter. Press / to set one.", severity="warning")
            return
        view = self._get_active_tree_view()
        if not view:
            return
        matches = view.find_matching_rows(self._name_filter)
        if not matches:
            self.notify("No matches found", severity="warning")
            return
        try:
            table = view.query_one(ProfileDataTable)
            current = table.cursor_row
            # Find next match after current position
            for idx in matches:
                if idx > current:
                    table.move_cursor(row=idx)
                    return
            # Wrap around to first match
            table.move_cursor(row=matches[0])
        except Exception:
            pass

    def action_search_prev(self) -> None:
        """Move cursor to the previous row matching the current filter."""
        if not self._name_filter:
            self.notify("No active filter. Press / to set one.", severity="warning")
            return
        view = self._get_active_tree_view()
        if not view:
            return
        matches = view.find_matching_rows(self._name_filter)
        if not matches:
            self.notify("No matches found", severity="warning")
            return
        try:
            table = view.query_one(ProfileDataTable)
            current = table.cursor_row
            # Find previous match before current position
            for idx in reversed(matches):
                if idx < current:
                    table.move_cursor(row=idx)
                    return
            # Wrap around to last match
            table.move_cursor(row=matches[-1])
        except Exception:
            pass

    def action_toggle_hotpath(self) -> None:
        """Toggle hot-path highlighting mode."""
        view = self._get_active_tree_view()
        if not view:
            return
        view._hotpath_mode = not view._hotpath_mode
        view._refresh_table()
        state = "ON" if view._hotpath_mode else "OFF"
        self.notify(f"Hot-path mode: {state}")

    def action_copy_node(self) -> None:
        """Copy the currently selected node's details to the clipboard."""
        view = self._get_active_tree_view()
        if not view:
            return
        try:
            table = view.query_one(ProfileDataTable)
            node = view.get_node_at_row(table.cursor_row)
            if node is None:
                self.notify("No node selected", severity="warning")
                return
            mean_ms = node.exclusive_ms / node.call_count if node.call_count > 0 else 0.0
            lines = [
                f"Name: {node.name}",
                f"Exclusive: {node.exclusive_ms:.3f} ms",
                f"Inclusive: {node.inclusive_ms:.3f} ms",
                f"Calls: {node.call_count}",
                f"Mean: {mean_ms:.3f} ms",
            ]
            if node.stddev_ms > 0:
                lines.append(f"Stddev: {node.stddev_ms:.3f} ms")
            if node.file:
                loc = node.file
                if node.line > 0:
                    loc += f":{node.line}"
                lines.append(f"Location: {loc}")
            if node.function:
                lines.append(f"Function: {node.function}")
            if node.annotations:
                lines.append("Annotations:")
                for k, v in node.annotations.items():
                    if isinstance(v, dict):
                        avg = v.get("avg", "")
                        vmin = v.get("min", "")
                        vmax = v.get("max", "")
                        lines.append(f"  {k} = avg:{avg}  min:{vmin}  max:{vmax}")
                    else:
                        lines.append(f"  {k} = {v}")
            text = "\n".join(lines)
            self.app.copy_to_clipboard(text)
            self.notify("Copied to clipboard")
        except Exception:
            self.notify("Failed to copy", severity="error")

    def action_diff_snapshot(self) -> None:
        """Toggle diff/snapshot mode. First press captures baseline, second clears it."""
        session = self._get_active_session()
        if not session:
            return
        if session.baseline_snapshot is not None:
            session.baseline_snapshot = None
            self.notify("Baseline cleared")
        else:
            snap = session.latest_snapshot
            if snap is None:
                self.notify("No snapshot available to use as baseline", severity="warning")
                return
            session.baseline_snapshot = copy.deepcopy(snap)
            self.notify("Baseline captured")
        self._update_status("Connected" if not self._paused else "PAUSED")
        self._refresh_detail_panel()

    def action_toggle_flamegraph(self) -> None:
        """Toggle the flamegraph icicle view."""
        try:
            panel = self.query_one("#flamegraph-panel", FlameGraphPanel)
            if not panel.display:
                panel.display = True
                # Populate with current thread data
                view = self._get_active_tree_view()
                if view and view._data:
                    panel.update_data(view._data)
                self.notify("Flamegraph ON — click bars to zoom, Esc to zoom out")
            else:
                panel.display = False
                panel.zoom_reset()
        except Exception:
            pass

    def on_flame_graph_panel_span_clicked(self, event: FlameGraphPanel.SpanClicked) -> None:
        """Update detail panel when a flamegraph span is clicked."""
        panel = self.query_one("#detail-panel", DetailPanel)
        node = event.profile_node
        session = self._get_active_session()
        view = self._get_active_tree_view()
        total = view._total_ms if view else 1.0
        pct = (node.exclusive_ms / total * 100.0) if total > 0 else 0.0
        sparkline = ""
        node_history = session.node_history if session else {}
        if node.name in node_history:
            sparkline = _make_sparkline(list(node_history[node.name]))
        baseline = session.baseline_snapshot if session else None
        bm = session.bookmarks if session else set()
        panel.update_node(node, pct, baseline_snap=baseline, sparkline=sparkline, is_bookmarked=node.name in bm)

    def action_toggle_flat(self) -> None:
        """Toggle between tree and flat (aggregated) profile view."""
        view = self._get_active_tree_view()
        if not view:
            return
        view.toggle_flat_mode()
        state = "FLAT" if view._flat_mode else "TREE"
        self.notify(f"View mode: {state}")

    def action_toggle_roofline(self) -> None:
        """Toggle the roofline model plot."""
        if not HAS_PLOTEXT:
            self.notify("Install textual-plotext for roofline plot: pip install textual-plotext", severity="warning")
            return
        try:
            panel = self.query_one("#roofline-panel", RooflinePanel)
            if not panel.display:
                panel.display = True
                # Populate with current thread data
                view = self._get_active_tree_view()
                if view and view._data:
                    panel.update_data(view._data)
                    if not panel._points:
                        self.notify(
                            "No roofline data — nodes need 'flops' and 'bytes_read'/'bytes_written' annotations",
                            severity="warning",
                        )
                self.notify("Roofline ON")
            else:
                panel.display = False
        except Exception:
            pass

    def action_toggle_listen(self) -> None:
        """Toggle listening for new server connections (mDNS + retry loops)."""
        self._mdns_enabled = not self._mdns_enabled
        if self._mdns_enabled:
            if HAS_ZEROCONF and self._zeroconf is None:
                self._start_mdns_browser()
            self.notify("Listen mode: ON — accepting new connections")
        else:
            self._stop_mdns_browser()
            self._known_endpoints.clear()
            self.notify("Listen mode: OFF — no new connections")

    def action_toggle_record(self) -> None:
        """Toggle recording of incoming snapshots to a file."""
        if self._replay_path:
            self.notify("Cannot record during replay", severity="warning")
            return
        if self._recording:
            # Stop recording
            self._recording = False
            if self._record_file:
                self._record_file.close()
                self._record_file = None
            self.notify(f"Recording stopped: {self._record_path}")
        else:
            # Start recording
            if not self._record_path:
                self._record_path = "einsums_profile_recording.jsonl"
            self._record_file = open(self._record_path, "w")
            self._recording = True
            self.notify(f"Recording to {self._record_path}")
        self._update_status("Connected" if not self._paused else "PAUSED")

    def action_speed_up(self) -> None:
        """Increase replay speed (only during replay)."""
        if not self._replay_path:
            return
        self._replay_speed = min(self._replay_speed * 2.0, 64.0)
        self.notify(f"Replay speed: {self._replay_speed:.1f}x")

    def action_speed_down(self) -> None:
        """Decrease replay speed (only during replay)."""
        if not self._replay_path:
            return
        self._replay_speed = max(self._replay_speed / 2.0, 0.1)
        self.notify(f"Replay speed: {self._replay_speed:.1f}x")

    def action_toggle_bookmark(self) -> None:
        """Toggle bookmark on the currently selected node."""
        session = self._get_active_session()
        if not session:
            return
        view = self._get_active_tree_view()
        if not view:
            return
        try:
            table = view.query_one(ProfileDataTable)
            node = view.get_node_at_row(table.cursor_row)
            if node is None:
                return
            if node.name in session.bookmarks:
                session.bookmarks.discard(node.name)
                self.notify(f"Unbookmarked: {node.name}")
            else:
                session.bookmarks.add(node.name)
                self.notify(f"Bookmarked: {node.name}")
            # Refresh all thread views to show/hide bookmark flag
            for v in session.thread_views.values():
                v._refresh_table()
        except Exception:
            pass

    def action_next_bookmark(self) -> None:
        """Jump to the next bookmarked node in the current thread view."""
        session = self._get_active_session()
        if not session or not session.bookmarks:
            self.notify("No bookmarks set (press b to bookmark)", severity="warning")
            return
        view = self._get_active_tree_view()
        if not view:
            return
        matches = view.find_bookmarked_rows()
        if not matches:
            self.notify("No bookmarked nodes visible in this thread", severity="warning")
            return
        try:
            table = view.query_one(ProfileDataTable)
            current = table.cursor_row
            for idx in matches:
                if idx > current:
                    table.move_cursor(row=idx)
                    return
            # Wrap around
            table.move_cursor(row=matches[0])
        except Exception:
            pass

    def action_toggle_hotspots(self) -> None:
        """Toggle the top-N hotspots panel."""
        try:
            panel = self.query_one("#hotspots-panel", HotspotsPanel)
            if not panel.display:
                panel.display = True
                panel.update_hotspots(self._sessions)
                self.notify("Hotspots ON")
            else:
                panel.display = False
        except Exception:
            pass

    def action_toggle_bottom_up(self) -> None:
        """Toggle bottom-up view mode."""
        view = self._get_active_tree_view()
        if not view:
            return
        view.toggle_bottom_up_mode()
        state = "BOTTOM-UP" if view._bottom_up_mode else "TREE"
        self.notify(f"View mode: {state}")

    def action_toggle_source(self) -> None:
        """Toggle the source code panel."""
        try:
            panel = self.query_one("#source-panel", SourcePanel)
            if not panel.display:
                panel.display = True
                # Populate with current selection
                view = self._get_active_tree_view()
                if view:
                    try:
                        table = view.query_one(ProfileDataTable)
                        node = view.get_node_at_row(table.cursor_row)
                        panel.update_node(node)
                    except Exception:
                        pass
                self.notify("Source view ON")
            else:
                panel.display = False
        except Exception:
            pass

    def action_toggle_gantt(self) -> None:
        """Toggle the thread Gantt chart panel."""
        try:
            panel = self.query_one("#gantt-panel", GanttPanel)
            if not panel.display:
                panel.display = True
                session = self._get_active_session()
                if session and session.timeline_events:
                    panel.update_events(session.timeline_events)
                self.notify("Gantt chart ON")
            else:
                panel.display = False
        except Exception:
            pass

    def action_toggle_microarch(self) -> None:
        """Toggle micro-architecture stall analysis panel."""
        try:
            panel = self.query_one("#microarch-panel", MicroArchPanel)
            if not panel.display:
                panel.display = True
                # Populate with current selection
                view = self._get_active_tree_view()
                if view:
                    try:
                        table = view.query_one(ProfileDataTable)
                        node = view.get_node_at_row(table.cursor_row)
                        panel.update_node(node)
                    except Exception:
                        pass
                self.notify("MicroArch analysis ON")
            else:
                panel.display = False
        except Exception:
            pass

    def action_toggle_assembly(self) -> None:
        """Toggle the Assembly/IR panel for JIT kernels or executable functions."""
        try:
            panel = self.query_one("#assembly-panel", AssemblyPanel)
            if not panel.display:
                panel.display = True
                # Populate with current selection
                view = self._get_active_tree_view()
                if view:
                    try:
                        table = view.query_one(ProfileDataTable)
                        node = view.get_node_at_row(table.cursor_row)
                        if node:
                            self._update_assembly_for_node(node)
                    except Exception:
                        pass
                self.notify("Assembly panel ON")
            else:
                panel.display = False
        except Exception:
            pass

    def action_toggle_logs(self) -> None:
        """Toggle the log message panel."""
        try:
            panel = self.query_one("#log-panel", LogPanel)
            panel.display = not panel.display
            try:
                self.query_one("#log-resize", ResizeHandle).display = panel.display
            except Exception:
                pass
            if panel.display:
                session = self._get_active_session()
                if session and session.client:
                    panel.update_logs(session.client.log_entries)
                self.notify("Log panel ON")
        except Exception:
            pass

    def action_cycle_log_level(self) -> None:
        """Cycle the minimum log level displayed."""
        try:
            panel = self.query_one("#log-panel", LogPanel)
            new_level = panel.cycle_level()
            self.notify(f"Log level: {LOG_LEVEL_NAMES.get(new_level, '?')}+")
            session = self._get_active_session()
            if session and session.client:
                panel.update_logs(session.client.log_entries)
        except Exception:
            pass

    def _update_assembly_for_node(self, node: ProfileNode) -> None:
        """Update the assembly panel for the given node."""
        try:
            panel = self.query_one("#assembly-panel", AssemblyPanel)
            if not panel.display:
                return
        except Exception:
            return

        # Check for jit_kernel_hash annotation (stored as string to preserve unsigned size_t)
        jit_hash = None
        annotations = node.annotations or {}
        hash_val = annotations.get("jit_kernel_hash")
        if hash_val is not None:
            if isinstance(hash_val, dict):
                # Could be numeric annotation with last/avg/min/max or string annotation with last
                raw = hash_val.get("last", 0)
                try:
                    jit_hash = int(str(raw).strip('"'))
                except (ValueError, TypeError):
                    pass
            elif isinstance(hash_val, str):
                try:
                    jit_hash = int(hash_val)
                except (ValueError, TypeError):
                    pass
            else:
                try:
                    jit_hash = int(hash_val)
                except (ValueError, TypeError):
                    pass

        if jit_hash:
            # Find a connected client to send the request to
            client = None
            for c in self._clients:
                if c.connected:
                    client = c
                    break
            self.run_worker(
                panel.update_for_jit_kernel(jit_hash, client),
                exclusive=False,
                name="asm-fetch",
            )
        elif node.function:
            # Try executable disassembly
            exe_path = ""
            session = self._get_active_session()
            if session and session.meta:
                exe_path = session.meta.executable_path
            panel.update_for_executable_func(node.function, exe_path)
        else:
            panel.update_not_jit()

    def action_compare_sessions(self) -> None:
        """Open side-by-side session comparison."""
        sessions_with_data = [s for s in self._sessions.values() if s.latest_snapshot]
        if len(sessions_with_data) < 2:
            self.notify("Need at least 2 sessions to compare (load multiple with --load)", severity="warning")
            return
        if len(sessions_with_data) == 2:
            self.push_screen(CompareScreen(sessions_with_data[0], sessions_with_data[1]))
        else:
            choices = [(s.session_id, s.label) for s in sessions_with_data]
            self.push_screen(
                SessionPickerDialog(sessions=choices),
                self._on_session_picker_result,
            )

    def _on_session_picker_result(self, result: tuple[str, str] | None) -> None:
        if not result:
            return
        sid_a, sid_b = result
        sa = self._sessions.get(sid_a)
        sb = self._sessions.get(sid_b)
        if sa and sb:
            self.push_screen(CompareScreen(sa, sb))

    def action_cycle_color_theme(self) -> None:
        """Cycle color theme for tree rows."""
        themes = ["none", "heat"]
        view = self._get_active_tree_view()
        if not view:
            return
        idx = themes.index(view._color_theme) if view._color_theme in themes else 0
        new_theme = themes[(idx + 1) % len(themes)]
        # Apply to all views in the active session
        session = self._get_active_session()
        if session:
            for v in session.thread_views.values():
                v._color_theme = new_theme
                v._refresh_table()
        self.notify(f"Color theme: {new_theme}")

    def _session_to_dict(self, session: SessionState) -> dict[str, Any]:
        """Serialize a session's profiling data to a dict."""
        data: dict[str, Any] = {"session_id": session.session_id, "label": session.label}
        if session.meta:
            data["meta"] = {
                "pid": session.meta.pid,
                "hostname": session.meta.hostname,
                "executable": session.meta.executable,
                "start_time": session.meta.start_time,
                "counters": session.meta.counters,
                "executable_path": session.meta.executable_path,
            }
        if session.bookmarks:
            data["bookmarks"] = sorted(session.bookmarks)
        snap = session.latest_snapshot
        if snap:
            data["snapshot"] = {"seq": snap.seq, "dropped": snap.dropped, "threads": {}}
            for tid, tdata in snap.threads.items():
                data["snapshot"]["threads"][tid] = {
                    "name": tdata.thread_name,
                    "children": [self._node_to_dict(n) for n in tdata.children],
                }
        # Include node history for sparklines
        if session.node_history:
            data["node_history"] = {
                name: list(values) for name, values in session.node_history.items()
            }
        return data

    def action_save_session(self) -> None:
        """Save the current session's profiling data to a JSON file."""
        session = self._get_active_session()
        if not session or not session.latest_snapshot:
            self.notify("No session data to save", severity="warning")
            return
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in session.label)
        default_name = f"einsums_session_{safe_name}.json"
        self.push_screen(
            SaveDialog(title="Save current session", default_name=default_name),
            lambda path: self._do_save_session(session, path),
        )

    def _do_save_session(self, session: SessionState, path: str | None) -> None:
        if not path:
            return
        if os.path.exists(path):
            self.push_screen(
                ConfirmDialog(f"'{path}' already exists. Overwrite?"),
                lambda ok: self._write_session_file(session, path) if ok else None,
            )
        else:
            self._write_session_file(session, path)

    def _write_session_file(self, session: SessionState, path: str) -> None:
        data = self._session_to_dict(session)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.notify(f"Session saved to {path}")

    def action_save_all_sessions(self) -> None:
        """Save selected sessions' profiling data to a single JSON file."""
        if not self._sessions:
            self.notify("No sessions to save", severity="warning")
            return
        sessions_with_data = [
            s for s in self._sessions.values() if s.latest_snapshot
        ]
        if not sessions_with_data:
            self.notify("No session data to save", severity="warning")
            return
        session_choices = [(s.session_id, s.label) for s in sessions_with_data]
        self.push_screen(
            SaveAllDialog(sessions=session_choices),
            self._on_save_all_dialog_result,
        )

    def _on_save_all_dialog_result(self, result: tuple[list[str], str] | None) -> None:
        if not result:
            return
        selected_ids, path = result
        sessions = [
            s for s in self._sessions.values()
            if s.session_id in selected_ids and s.latest_snapshot
        ]
        if not sessions:
            self.notify("No sessions selected", severity="warning")
            return
        if os.path.exists(path):
            self.push_screen(
                ConfirmDialog(f"'{path}' already exists. Overwrite?"),
                lambda ok: self._write_all_sessions_file(sessions, path) if ok else None,
            )
        else:
            self._write_all_sessions_file(sessions, path)

    def _write_all_sessions_file(self, sessions: list[SessionState], path: str) -> None:
        data = {
            "sessions": [self._session_to_dict(s) for s in sessions],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.notify(f"Saved {len(sessions)} session(s) to {path}")

    def action_export_json(self) -> None:
        session = self._get_active_session()
        snap = session.latest_snapshot if session else None
        if not snap:
            self.notify("No data to export", severity="warning")
            return

        # Export JSON
        json_path = "einsums_profile_snapshot.json"
        data = {"seq": snap.seq, "dropped": snap.dropped, "threads": {}}
        for tid, tdata in snap.threads.items():
            data["threads"][tid] = {
                "name": tdata.thread_name,
                "children": [self._node_to_dict(n) for n in tdata.children]
            }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Export CSV alongside
        csv_path = "einsums_profile_snapshot.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "thread", "depth", "name", "call_count", "exclusive_ms",
                "inclusive_ms", "mean_ms", "stddev_ms", "anomaly",
                "mem_alloc_bytes", "mem_peak_bytes",
            ])
            for tid, tdata in snap.threads.items():
                thread_label = tdata.thread_name if tdata.thread_name else tid
                all_nodes = _collect_all_nodes(tdata.children)
                for node, depth in all_nodes:
                    mean_ms = node.exclusive_ms / node.call_count if node.call_count > 0 else 0.0
                    is_anom = _is_node_anomaly(node)
                    writer.writerow([
                        thread_label,
                        depth,
                        node.name,
                        node.call_count,
                        f"{node.exclusive_ms:.3f}",
                        f"{node.inclusive_ms:.3f}",
                        f"{mean_ms:.3f}",
                        f"{node.stddev_ms:.3f}",
                        "!" if is_anom else "",
                        node.mem_alloc_bytes,
                        node.mem_peak_bytes,
                    ])

        self.notify(f"Exported to {json_path} and {csv_path}")

    def _node_to_dict(self, node: ProfileNode) -> dict[str, Any]:
        """Recursively convert a ProfileNode to a serializable dict."""
        d: dict[str, Any] = {
            "name": node.name,
            "call_count": node.call_count,
            "exclusive_ms": node.exclusive_ms,
            "inclusive_ms": node.inclusive_ms,
            "exclusive_min_ms": node.exclusive_min_ms,
            "exclusive_max_ms": node.exclusive_max_ms,
            "stddev_ms": node.stddev_ms,
            "file": node.file,
            "line": node.line,
            "function": node.function,
            "annotations": node.annotations,
            "counters": node.counters,
            "children": [self._node_to_dict(c) for c in node.children],
        }
        if node.mem_alloc_bytes > 0 or node.mem_free_bytes > 0:
            d["memory"] = {
                "alloc_count": node.mem_alloc_count,
                "free_count": node.mem_free_count,
                "alloc_bytes": node.mem_alloc_bytes,
                "free_bytes": node.mem_free_bytes,
                "current_bytes": node.mem_current_bytes,
                "peak_bytes": node.mem_peak_bytes,
            }
        return d


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Einsums Profile Viewer -- real-time TUI"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Profiler server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, nargs="+", default=None,
        help="Profiler server port(s). Specify multiple to connect to several servers.",
    )
    parser.add_argument(
        "--connect", metavar="HOST:PORT", nargs="+", default=[],
        help="Additional host:port endpoints to connect to (e.g. 10.0.0.5:19216)",
    )
    parser.add_argument(
        "--record", metavar="FILE", default=None,
        help="Record incoming snapshots to a .jsonl file for later replay",
    )
    parser.add_argument(
        "--replay", metavar="FILE", default=None,
        help="Replay a previously recorded .jsonl file instead of connecting live",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier for --replay (default: 1.0)",
    )
    parser.add_argument(
        "--load", metavar="FILE", nargs="+", default=[],
        help="Load saved session JSON file(s) (from Save session / Save all sessions)",
    )
    parser.add_argument(
        "--no-mdns", action="store_true", default=False,
        help="Disable mDNS/Bonjour auto-discovery of profiler servers",
    )
    args = parser.parse_args()

    # Build endpoint list (empty = let mDNS handle discovery)
    endpoints: list[tuple[str, int]] = []
    if args.port:
        for p in args.port:
            endpoints.append((args.host, p))
    for spec in args.connect:
        if ":" in spec:
            h, ps = spec.rsplit(":", 1)
            endpoints.append((h, int(ps)))
        else:
            endpoints.append(("127.0.0.1", int(spec)))

    app = ProfileViewerApp(
        endpoints=endpoints,
        record_path=args.record,
        replay_path=args.replay,
        replay_speed=args.speed,
        mdns=not args.no_mdns,
        load_paths=args.load,
    )
    app.run()


if __name__ == "__main__":
    main()
