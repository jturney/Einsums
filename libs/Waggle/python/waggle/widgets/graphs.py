# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Graphical panels: flame graph, thread Gantt chart, and the plotext timeline and roofline."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from ..analysis import RooflinePoint, roofline_points, sum_exclusive, sum_mem_current
from ..format import flame_color
from ..model import ProfileNode, ProfileSnapshot, TimelineEvent

try:
    from textual_plotext import PlotextPlot

    HAVE_PLOTEXT = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_PLOTEXT = False


def _join(lines: list[Text]) -> Text:
    return Text("\n").join(lines)


@dataclass
class _Span:
    start: int
    end: int
    node: ProfileNode
    depth: int


class FlameGraph(Widget):
    """Icicle chart of the active thread: roots on top, width proportional to inclusive time.

    Click a bar to zoom into it; Escape or Backspace zooms back out.
    """

    can_focus = True
    MAX_DEPTH = 20

    class SpanClicked(Message):
        def __init__(self, node: ProfileNode) -> None:
            super().__init__()
            self.node = node

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._roots: list[ProfileNode] = []
        self._shown: list[ProfileNode] = []
        self._zoom: list[list[ProfileNode]] = []
        self._spans: list[_Span] = []

    def set_roots(self, roots: list[ProfileNode]) -> None:
        self._roots = roots
        if not self._zoom:
            self._shown = roots
        self._relayout()

    def zoom_in(self, node: ProfileNode) -> None:
        if node.children:
            self._zoom.append(self._shown)
            self._shown = [node]
            self._relayout()

    def zoom_out(self) -> None:
        if self._zoom:
            self._shown = self._zoom.pop()
            self._relayout()

    def zoom_reset(self) -> None:
        self._zoom.clear()
        self._shown = self._roots
        self._relayout()

    def on_resize(self) -> None:
        self._relayout()

    def _relayout(self) -> None:
        self._spans = []
        total = sum(n.inclusive_ms for n in self._shown)
        if total > 0:
            self._layout(self._shown, 0, 0, self.size.width or 120, total)
        self.refresh()

    def _layout(self, nodes: list[ProfileNode], depth: int, x0: int, x1: int, total: float) -> None:
        if depth >= self.MAX_DEPTH or x1 <= x0 or total <= 0:
            return
        x = float(x0)
        for node in nodes:
            width = node.inclusive_ms / total * (x1 - x0)
            start, end = int(x + 0.5), int(x + width + 0.5)
            if end <= start and width > 0.3:
                end = start + 1
            if end > start:
                self._spans.append(_Span(start, end, node, depth))
                self._layout(node.children, depth + 1, start, end, sum(c.inclusive_ms for c in node.children))
            x += width

    def render(self) -> Text:
        if not self._spans:
            return Text("No data yet", style="dim")
        width = self.size.width or 120
        top = 1 if self._zoom else 0
        depth_limit = min(max(s.depth for s in self._spans) + 1, max(1, (self.size.height or 20) - top))
        lines: list[Text] = []
        if self._zoom:
            lines.append(Text(f" Zoomed {len(self._zoom)} level(s) — Esc to zoom out ", style="bold reverse"))
        for depth in range(depth_limit):
            line, pos = Text(), 0
            for span in sorted((s for s in self._spans if s.depth == depth), key=lambda s: s.start):
                line.append(" " * (span.start - pos))
                bar = span.end - span.start
                label, ms = span.node.name, f" {span.node.inclusive_ms:.1f}ms"
                if bar >= len(label) + len(ms) + 2:
                    text = f" {label}{ms} "
                elif bar >= 4:
                    text = f" {label[: bar - 2]} "
                else:
                    text = "█" * bar
                line.append(text.ljust(bar)[:bar], style=f"black on {flame_color(span.node.name)}")
                pos = span.end
            line.append(" " * max(0, width - pos))
            lines.append(line)
        return _join(lines)

    def on_click(self, event: Any) -> None:
        row = event.y - (1 if self._zoom else 0)
        for span in self._spans:
            if span.depth == row and span.start <= event.x < span.end:
                self.post_message(self.SpanClicked(span.node))
                self.zoom_in(span.node)
                return

    def key_escape(self) -> None:
        self.zoom_out()

    def key_backspace(self) -> None:
        self.zoom_out()


class GanttChart(Widget):
    """Recent zone spans per thread, from the server's ``timeline`` messages."""

    LABEL_WIDTH = 14

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._events: list[TimelineEvent] = []

    def set_events(self, events: list[TimelineEvent]) -> None:
        self._events = events
        self.refresh()

    def render(self) -> Text:
        if not self._events:
            return Text("No timeline data (it streams only from a live connection)", style="dim")
        t0 = min(e.start_ms for e in self._events)
        span = max(e.end_ms for e in self._events) - t0
        if span <= 0:
            return Text("No time range", style="dim")
        chart = max(10, (self.size.width or 120) - self.LABEL_WIDTH - 1)

        header = Text(f"{'thread':<{self.LABEL_WIDTH}}", style="bold")
        step = max(1, chart // 5)
        for col in range(0, chart, step):
            header.append(f"{t0 + col / chart * span:.0f}ms".ljust(step)[: chart - col])
        lines = [header]

        threads: dict[str, list[TimelineEvent]] = {}
        for event in self._events:
            threads.setdefault(event.thread_id, []).append(event)
        for tid, events in sorted(threads.items())[: max(1, (self.size.height or 20) - 1)]:
            # The innermost zone wins a cell: later (nested) events overwrite earlier ones.
            owner: list[TimelineEvent | None] = [None] * chart
            for event in sorted(events, key=lambda e: (e.start_ms, -(e.end_ms - e.start_ms))):
                lo = max(0, min(chart - 1, int((event.start_ms - t0) / span * chart)))
                hi = max(lo + 1, min(chart, int((event.end_ms - t0) / span * chart)))
                owner[lo:hi] = [event] * (hi - lo)
            line = Text(f"T{tid}"[: self.LABEL_WIDTH - 1].ljust(self.LABEL_WIDTH), style="bold")
            col = 0
            while col < chart:
                event = owner[col]
                end = col
                while end < chart and owner[end] is event:
                    end += 1
                if event is None:
                    line.append(" " * (end - col))
                else:
                    width = end - col
                    text = f" {event.name} ".ljust(width)[:width] if len(event.name) + 2 <= width else "█" * width
                    style = f"black on {flame_color(event.name)}" if text[0] != "█" else flame_color(event.name)
                    line.append(text, style=style)
                col = end
            lines.append(line)
        return _join(lines)


class _MissingPlotext(Static):
    def __init__(self, what: str, **kwargs: Any) -> None:
        super().__init__(f"The {what} needs textual-plotext (conda install -c conda-forge textual-plotext)", **kwargs)


class TimelinePlot(Widget):
    """Profiled CPU time per wall second, and live tracked memory, over the last two minutes."""

    HISTORY = 240  # samples, one per snapshot shown

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._t: deque[float] = deque(maxlen=self.HISTORY)
        self._cpu: deque[float] = deque(maxlen=self.HISTORY)
        self._mem: deque[float] = deque(maxlen=self.HISTORY)
        self._start = time.monotonic()
        self._last: tuple[float, float] | None = None  # (wall, total exclusive ms)

    def compose(self) -> ComposeResult:
        if not HAVE_PLOTEXT:
            yield _MissingPlotext("timeline")
            return
        with Horizontal():
            yield PlotextPlot(id="cpu-plot")
            yield PlotextPlot(id="mem-plot")

    def record(self, snap: ProfileSnapshot) -> None:
        now = time.monotonic()
        roots = snap.all_roots()
        total = sum_exclusive(roots)
        cpu = 0.0
        if self._last is not None and now > self._last[0]:
            cpu = (total - self._last[1]) / ((now - self._last[0]) * 1000.0) * 100.0
        self._last = (now, total)
        self._t.append(now - self._start)
        self._cpu.append(max(0.0, cpu))
        self._mem.append(sum_mem_current(roots) / (1 << 20))
        if self.display and HAVE_PLOTEXT:
            self._plot()

    def _plot(self) -> None:
        if len(self._t) < 2:
            return
        for plot_id, title, unit, values in (
            ("#cpu-plot", "Profiled CPU (% of one core)", "%", self._cpu),
            ("#mem-plot", "Live tracked memory", "MiB", self._mem),
        ):
            widget = self.query_one(plot_id, PlotextPlot)
            plt = widget.plt
            plt.clear_data()
            plt.clear_figure()
            plt.title(title)
            plt.xlabel("time (s)")
            plt.ylabel(unit)
            plt.ylim(0, max(max(values), 1.0))
            plt.plot(list(self._t), list(values), marker="braille")
            widget.refresh()


class RooflinePlot(Widget):
    """Zones annotated with ``flops`` and ``bytes_read``/``bytes_written`` against a roofline."""

    def __init__(self, peak_gflops: float = 200.0, peak_gb_per_s: float = 50.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.peak_gflops = peak_gflops
        self.peak_gb_per_s = peak_gb_per_s
        self.points: list[RooflinePoint] = []

    def compose(self) -> ComposeResult:
        if not HAVE_PLOTEXT:
            yield _MissingPlotext("roofline")
            return
        yield PlotextPlot(id="roofline-plot")

    def set_roots(self, roots: list[ProfileNode]) -> None:
        self.points = roofline_points(roots)
        if HAVE_PLOTEXT and self.points:
            self._plot()

    def _plot(self) -> None:
        # Log-log, but with the logarithms taken here: plotext's own log scale transforms the
        # data in place at every render, so the second redraw takes the log of a log and fails.
        widget = self.query_one("#roofline-plot", PlotextPlot)
        plt = widget.plt
        plt.clear_data()
        plt.clear_figure()
        plt.title("Roofline")
        plt.xlabel("arithmetic intensity (FLOP/byte)")
        plt.ylabel("GFLOP/s")
        ais = [p.intensity for p in self.points]
        gfs = [p.gflops for p in self.points]
        x0, x1 = math.log10(max(min(ais) * 0.5, 1e-3)), math.log10(max(ais) * 2.0)
        y0 = math.log10(max(min(gfs) * 0.5, 1e-3))
        y1 = math.log10(max(max(gfs) * 2.0, self.peak_gflops * 1.2))
        plt.xlim(x0, x1)
        plt.ylim(y0, y1)
        for axis, lo, hi in ((plt.xticks, x0, x1), (plt.yticks, y0, y1)):
            decades = list(range(math.floor(lo), math.ceil(hi) + 1))
            axis(decades, [f"{10.0**d:g}" for d in decades])
        xs = [x0 + (x1 - x0) * i / 100 for i in range(101)]
        roof = [math.log10(min(self.peak_gflops, self.peak_gb_per_s * 10.0**x)) for x in xs]
        plt.plot(xs, roof, marker="braille", label=f"roof ({self.peak_gflops:.0f} GFLOP/s, {self.peak_gb_per_s:.0f} GB/s)")
        plt.scatter([math.log10(v) for v in ais], [math.log10(v) for v in gfs], marker="dot")
        for p in sorted(self.points, key=lambda p: p.gflops, reverse=True)[:5]:
            plt.text(p.name[:20], math.log10(p.intensity), math.log10(p.gflops))
        widget.refresh()
