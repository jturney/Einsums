#!/usr/bin/env python3
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Textual TUI for viewing Einsums PackedGemm benchmark results.

Supports output from both BenchmarkContraction_test and BenchmarkSortGemm_test.

Usage:
    # Pipe benchmark output directly:
    ./BenchmarkContraction_test 2>&1 | python bench_viewer.py

    # Or the Sort+GEMM benchmark:
    ./BenchmarkSortGemm_test 2>&1 | python bench_viewer.py

    # Combine both:
    { ./BenchmarkContraction_test; ./BenchmarkSortGemm_test; } 2>&1 | python bench_viewer.py

    # Or load from a saved file:
    python bench_viewer.py results.txt

    # Or run the benchmark and view (conda env with textual):
    python bench_viewer.py --run /path/to/BenchmarkContraction_test
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from devtools.benchmarks.parser import (
        BenchmarkResult, BreakdownResult, CacheHitResult, parse_output,
    )
except ImportError:
    # Fallback: allow running standalone without the benchmarks package installed
    _USE_LOCAL_PARSER = True
else:
    _USE_LOCAL_PARSER = False

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """A single benchmark timing result."""
    label: str
    n: int
    t_generic: float  # us
    t_blas_packed: float | None = None
    t_mlir: float | None = None
    t_einsum: float | None = None
    t_sort_gemm: float | None = None
    einsum_alg: str = ""
    speedup_blas: float | None = None
    speedup_mlir: float | None = None
    speedup_einsum: float | None = None
    speedup_sort_gemm: float | None = None


@dataclass
class BreakdownResult:
    """Packing vs micro-kernel breakdown."""
    n: int
    t_pack_only: float
    t_kernel_only: float
    t_total: float
    mr: int
    nr: int
    tiles: int


@dataclass
class CacheHitResult:
    """Cache-hit overhead measurement."""
    n: int
    t_generic: float
    t_mlir: float
    t_einsum: float
    einsum_alg: str
    speedup_mlir: float
    speedup_einsum: float


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Standard line:
# [Rank-2 N=64] Generic: 123.45 us  BLAS packed: 45.67 us  MLIR: 56.78 us  einsum (BLAS-GEMM): 34.56 us  BLAS/gen: 2.70x  MLIR/gen: 2.17x  einsum/gen: 3.57x
_RE_STANDARD = re.compile(
    r"\[(.+?)\s+N=(\d+)\]\s+"
    r"Generic:\s+([\d.]+)\s+.*?s"
    r"\s+BLAS packed:\s+([\d.]+)\s+.*?s"
    r"\s+MLIR:\s+([\d.]+)\s+.*?s"
    r"\s+einsum\s+\(([^)]+)\):\s+([\d.]+)\s+.*?s"
    r"\s+BLAS/gen:\s+([\d.]+)x"
    r"\s+MLIR/gen:\s+([\d.]+)x"
    r"\s+einsum/gen:\s+([\d.]+)x"
)

# Non-GEMM line (only Generic + einsum, no BLAS packed / MLIR columns):
# [Hadamard C[i]+=A[i,i]*B[i] N=256] Generic: 12.34 us  einsum (generic/MLIR): 5.67 us  einsum/gen: 2.18x
_RE_NONGEMM = re.compile(
    r"\[(.+?)\s+N=(\d+)\]\s+"
    r"Generic:\s+([\d.]+)\s+.*?s"
    r"\s+einsum\s+\(([^)]+)\):\s+([\d.]+)\s+.*?s"
    r"\s+einsum/gen:\s+([\d.]+)x"
)

# Large-K line (has M, N, K in the label):
# [Large-K ... M=64 N=64 K=2048] Generic: ... BLAS packed: ... MLIR: ... einsum (...): ... BLAS/gen: ... MLIR/gen: ... einsum/gen: ...
_RE_LARGE_K = re.compile(
    r"\[(.+?)\s+M=(\d+)\s+N=(\d+)\s+K=(\d+)\]\s+"
    r"Generic:\s+([\d.]+)\s+.*?s"
    r"\s+BLAS packed:\s+([\d.]+)\s+.*?s"
    r"\s+MLIR:\s+([\d.]+)\s+.*?s"
    r"\s+einsum\s+\(([^)]+)\):\s+([\d.]+)\s+.*?s"
    r"\s+BLAS/gen:\s+([\d.]+)x"
    r"\s+MLIR/gen:\s+([\d.]+)x"
    r"\s+einsum/gen:\s+([\d.]+)x"
)

# Batch line (has B and N in label):
# [Batch ... B=16 N=32] Generic: ... einsum (...): ... einsum/gen: ...
_RE_BATCH = re.compile(
    r"\[(.+?)\s+B=(\d+)\s+N=(\d+)\]\s+"
    r"Generic:\s+([\d.]+)\s+.*?s"
    r"\s+einsum\s+\(([^)]+)\):\s+([\d.]+)\s+.*?s"
    r"\s+einsum/gen:\s+([\d.]+)x"
)

# Breakdown line:
# [Breakdown N=128] pack_only: 45.32 us  kernel_only: 12.45 us  total: 78.90 us  (MR=16 NR=6 tiles=128)
_RE_BREAKDOWN = re.compile(
    r"\[Breakdown\s+N=(\d+)\]\s+"
    r"pack_only:\s+([\d.]+)\s+.*?s"
    r"\s+kernel_only:\s+([\d.]+)\s+.*?s"
    r"\s+total:\s+([\d.]+)\s+.*?s"
    r"\s+\(MR=(\d+)\s+NR=(\d+)\s+tiles=(\d+)\)"
)

# Sort+GEMM multi-line block (from BenchmarkSortGemm.cpp print_result):
# [Rank-3 C[i,l]+=A[i,k,j]*B[l,j,k] N=32]
#   Generic:       1234.56 µs
#   MLIR:           567.89 µs  (2.17x vs generic)
#   Sort+GEMM:      345.67 µs  (3.57x vs generic)
#   einsum(BLAS-GEMM):  234.56 µs  (5.27x vs generic)
_RE_SORT_HEADER = re.compile(r"^\[(.+?)\s+N=(\d+)\]\s*$")
_RE_SORT_GENERIC = re.compile(r"^\s+Generic:\s+([\d.]+)\s+.*?s\s*$")
_RE_SORT_MLIR = re.compile(r"^\s+MLIR:\s+([\d.]+)\s+.*?s\s+\(([\d.]+)x\s+vs\s+generic\)")
_RE_SORT_SORTGEMM = re.compile(r"^\s+Sort\+GEMM:\s+([\d.]+)\s+.*?s\s+\(([\d.]+)x\s+vs\s+generic\)")
_RE_SORT_EINSUM = re.compile(r"^\s+einsum\(([^)]+)\):\s+([\d.]+)\s+.*?s\s+\(([\d.]+)x\s+vs\s+generic\)")

# Cache hit line:
# [Cache hit N=32] Generic OMP: 234.56 us  MLIR direct: 89.12 us  einsum (BLAS-GEMM): 78.34 us  MLIR/generic: 2.63x  einsum/generic: 2.99x
_RE_CACHE = re.compile(
    r"\[Cache hit\s+N=(\d+)\]\s+"
    r"Generic OMP:\s+([\d.]+)\s+.*?s"
    r"\s+MLIR direct:\s+([\d.]+)\s+.*?s"
    r"\s+einsum\s+\(([^)]+)\):\s+([\d.]+)\s+.*?s"
    r"\s+MLIR/generic:\s+([\d.]+)x"
    r"\s+einsum/generic:\s+([\d.]+)x"
)


def parse_output(text: str) -> tuple[list[BenchmarkResult], list[BreakdownResult], list[CacheHitResult]]:
    """Parse benchmark stdout text into structured results."""
    benchmarks: list[BenchmarkResult] = []
    breakdowns: list[BreakdownResult] = []
    cache_hits: list[CacheHitResult] = []

    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        # --- Multi-line Sort+GEMM block ---
        # Starts with "[label N=X]" on its own line, followed by indented timing lines.
        m_hdr = _RE_SORT_HEADER.match(line)
        if m_hdr:
            label = m_hdr.group(1).strip()
            n = int(m_hdr.group(2))
            t_gen = t_ml = t_sg = t_es = None
            sp_ml = sp_sg = sp_es = None
            es_alg = ""
            # Consume following indented lines.
            j = idx + 1
            while j < len(lines):
                sub = lines[j]
                mg = _RE_SORT_GENERIC.match(sub)
                if mg:
                    t_gen = float(mg.group(1))
                    j += 1
                    continue
                mm = _RE_SORT_MLIR.match(sub)
                if mm:
                    t_ml = float(mm.group(1))
                    sp_ml = float(mm.group(2))
                    j += 1
                    continue
                ms = _RE_SORT_SORTGEMM.match(sub)
                if ms:
                    t_sg = float(ms.group(1))
                    sp_sg = float(ms.group(2))
                    j += 1
                    continue
                me = _RE_SORT_EINSUM.match(sub)
                if me:
                    es_alg = me.group(1)
                    t_es = float(me.group(2))
                    sp_es = float(me.group(3))
                    j += 1
                    continue
                break  # Non-matching line: end of block.
            if t_gen is not None:
                benchmarks.append(BenchmarkResult(
                    label=label,
                    n=n,
                    t_generic=t_gen,
                    t_mlir=t_ml,
                    t_sort_gemm=t_sg,
                    t_einsum=t_es,
                    einsum_alg=es_alg,
                    speedup_mlir=sp_ml,
                    speedup_sort_gemm=sp_sg,
                    speedup_einsum=sp_es,
                ))
                idx = j
                continue

        # --- Single-line formats ---

        # Try full standard line first (Generic + BLAS packed + MLIR + einsum)
        m = _RE_STANDARD.search(line)
        if m:
            benchmarks.append(BenchmarkResult(
                label=m.group(1).strip(),
                n=int(m.group(2)),
                t_generic=float(m.group(3)),
                t_blas_packed=float(m.group(4)),
                t_mlir=float(m.group(5)),
                einsum_alg=m.group(6),
                t_einsum=float(m.group(7)),
                speedup_blas=float(m.group(8)),
                speedup_mlir=float(m.group(9)),
                speedup_einsum=float(m.group(10)),
            ))
            idx += 1
            continue

        # Large-K line (M=... N=... K=...)
        m = _RE_LARGE_K.search(line)
        if m:
            benchmarks.append(BenchmarkResult(
                label=m.group(1).strip(),
                n=int(m.group(3)),  # use N from M/N/K
                t_generic=float(m.group(5)),
                t_blas_packed=float(m.group(6)),
                t_mlir=float(m.group(7)),
                einsum_alg=m.group(8),
                t_einsum=float(m.group(9)),
                speedup_blas=float(m.group(10)),
                speedup_mlir=float(m.group(11)),
                speedup_einsum=float(m.group(12)),
            ))
            idx += 1
            continue

        # Batch line (B=... N=...)
        m = _RE_BATCH.search(line)
        if m:
            benchmarks.append(BenchmarkResult(
                label=m.group(1).strip(),
                n=int(m.group(3)),  # use N
                t_generic=float(m.group(4)),
                einsum_alg=m.group(5),
                t_einsum=float(m.group(6)),
                speedup_einsum=float(m.group(7)),
            ))
            idx += 1
            continue

        # Non-GEMM line (Generic + einsum only)
        m = _RE_NONGEMM.search(line)
        if m:
            benchmarks.append(BenchmarkResult(
                label=m.group(1).strip(),
                n=int(m.group(2)),
                t_generic=float(m.group(3)),
                einsum_alg=m.group(4),
                t_einsum=float(m.group(5)),
                speedup_einsum=float(m.group(6)),
            ))
            idx += 1
            continue

        m = _RE_BREAKDOWN.search(line)
        if m:
            breakdowns.append(BreakdownResult(
                n=int(m.group(1)),
                t_pack_only=float(m.group(2)),
                t_kernel_only=float(m.group(3)),
                t_total=float(m.group(4)),
                mr=int(m.group(5)),
                nr=int(m.group(6)),
                tiles=int(m.group(7)),
            ))
            idx += 1
            continue

        m = _RE_CACHE.search(line)
        if m:
            cache_hits.append(CacheHitResult(
                n=int(m.group(1)),
                t_generic=float(m.group(2)),
                t_mlir=float(m.group(3)),
                einsum_alg=m.group(4),
                t_einsum=float(m.group(5)),
                speedup_mlir=float(m.group(6)),
                speedup_einsum=float(m.group(7)),
            ))
            idx += 1
            continue

        idx += 1

    return benchmarks, breakdowns, cache_hits


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------

class BarChart(Static):
    """A simple horizontal bar chart rendered with block characters."""

    def __init__(
        self,
        data: list[tuple[str, float, str]],  # (label, value, color)
        max_val: float | None = None,
        unit: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._data = data
        self._max_val = max_val
        self._unit = unit

    def render(self) -> str:
        if not self._data:
            return "No data"

        max_val = self._max_val or max(v for _, v, _ in self._data)
        if max_val <= 0:
            return "No data"

        max_label = max(len(lab) for lab, _, _ in self._data)
        bar_width = 50
        lines = []

        for label, value, color in self._data:
            filled = int(bar_width * value / max_val) if max_val > 0 else 0
            filled = max(filled, 1) if value > 0 else 0
            bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
            lines.append(
                f"  [{color}]{label:<{max_label}}[/] {bar} {value:>10.2f} {self._unit}"
            )

        return "\n".join(lines)


class SpeedupChart(Static):
    """Horizontal bar chart for speedup ratios with a 1.0x reference line."""

    def __init__(
        self,
        data: list[tuple[str, float, str]],  # (label, speedup, color)
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._data = data

    def render(self) -> str:
        if not self._data:
            return "No data"

        max_val = max(max(v for _, v, _ in self._data), 1.5)
        max_label = max(len(lab) for lab, _, _ in self._data)
        bar_width = 50
        ref_pos = int(bar_width * 1.0 / max_val)

        lines = []
        # Reference line header
        ref_marker = " " * (max_label + 3) + " " * ref_pos + "\u2502 1.0x"
        lines.append(f"  [dim]{ref_marker}[/]")

        for label, value, color in self._data:
            filled = int(bar_width * value / max_val)
            filled = max(filled, 1) if value > 0 else 0

            # Build bar with reference marker
            bar_chars = list("\u2591" * bar_width)
            for idx in range(min(filled, bar_width)):
                bar_chars[idx] = "\u2588"
            if 0 <= ref_pos < bar_width:
                bar_chars[ref_pos] = "\u2502"

            bar = "".join(bar_chars)
            indicator = "[bold green]faster[/]" if value >= 1.0 else "[bold red]slower[/]"
            lines.append(
                f"  [{color}]{label:<{max_label}}[/] {bar} {value:>6.2f}x {indicator}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class BenchViewerApp(App):
    """Einsums PackedGemm Benchmark Viewer."""

    CSS = """
    Screen {
        background: $surface;
    }

    #title-bar {
        dock: top;
        height: 3;
        background: $primary;
        color: $text;
        content-align: center middle;
        text-style: bold;
        padding: 0 2;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1 2;
    }

    DataTable {
        height: auto;
        max-height: 20;
        margin: 1 0;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0 0 0;
    }

    .chart-container {
        margin: 1 0;
        padding: 1;
        border: solid $primary-lighten-2;
        height: auto;
    }

    .summary-box {
        margin: 1 0;
        padding: 1 2;
        border: solid $accent;
        height: auto;
        background: $surface-darken-1;
    }

    .breakdown-box {
        margin: 1 0;
        padding: 1 2;
        border: solid $warning;
        height: auto;
    }

    .cache-box {
        margin: 1 0;
        padding: 1 2;
        border: solid $success;
        height: auto;
    }

    .no-data {
        margin: 2;
        color: $text-muted;
        text-style: italic;
    }

    .fastest {
        color: $success;
        text-style: bold;
    }
    """

    TITLE = "Einsums MLIR Benchmark Viewer"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "toggle_dark", "Toggle dark"),
        Binding("1", "tab_overview", "Overview"),
        Binding("2", "tab_speedups", "Speedups"),
        Binding("3", "tab_breakdown", "Breakdown"),
        Binding("4", "tab_raw", "Raw output"),
    ]

    def __init__(
        self,
        benchmarks: list[BenchmarkResult],
        breakdowns: list[BreakdownResult],
        cache_hits: list[CacheHitResult],
        raw_text: str = "",
    ) -> None:
        super().__init__()
        self.benchmarks = benchmarks
        self.breakdowns = breakdowns
        self.cache_hits = cache_hits
        self.raw_text = raw_text

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent("Overview", "Speedups", "Breakdown", "Raw Output"):
            with TabPane("Overview", id="tab-overview"):
                yield from self._compose_overview()
            with TabPane("Speedups", id="tab-speedups"):
                yield from self._compose_speedups()
            with TabPane("Breakdown", id="tab-breakdown"):
                yield from self._compose_breakdown()
            with TabPane("Raw Output", id="tab-raw"):
                yield from self._compose_raw()
        yield Footer()

    # -- Overview tab --

    def _compose_overview(self) -> ComposeResult:
        if not self.benchmarks:
            yield Label("No benchmark data found.", classes="no-data")
            return

        yield Label("Contraction Timings (all times in us)", classes="section-title")

        table = DataTable(id="overview-table")
        yield table

        # Summary stats
        yield Label("Summary", classes="section-title")
        yield Static(self._summary_text(), classes="summary-box")

    def on_mount(self) -> None:
        """Populate tables after mount."""
        try:
            table = self.query_one("#overview-table", DataTable)
        except Exception:
            return

        table.add_columns(
            "Benchmark", "N", "Generic (us)", "BLAS Packed (us)",
            "MLIR (us)", "Sort+GEMM (us)", "einsum (us)", "Algorithm", "Best"
        )

        for b in self.benchmarks:
            times = {}
            if b.t_generic:
                times["Generic"] = b.t_generic
            if b.t_blas_packed:
                times["BLAS"] = b.t_blas_packed
            if b.t_mlir:
                times["MLIR"] = b.t_mlir
            if b.t_sort_gemm:
                times["Sort+GEMM"] = b.t_sort_gemm
            if b.t_einsum:
                times["einsum"] = b.t_einsum

            best = min(times, key=times.get) if times else "?"

            table.add_row(
                b.label,
                str(b.n),
                f"{b.t_generic:.2f}",
                f"{b.t_blas_packed:.2f}" if b.t_blas_packed else "-",
                f"{b.t_mlir:.2f}" if b.t_mlir else "-",
                f"{b.t_sort_gemm:.2f}" if b.t_sort_gemm else "-",
                f"{b.t_einsum:.2f}" if b.t_einsum else "-",
                b.einsum_alg or "-",
                best,
            )

        # Populate cache table if present
        try:
            ct = self.query_one("#cache-table", DataTable)
            ct.add_columns("N", "Generic (us)", "MLIR (us)", "einsum (us)", "Algorithm", "MLIR/gen", "einsum/gen")
            for c in self.cache_hits:
                ct.add_row(
                    str(c.n),
                    f"{c.t_generic:.2f}",
                    f"{c.t_mlir:.2f}",
                    f"{c.t_einsum:.2f}",
                    c.einsum_alg,
                    f"{c.speedup_mlir:.2f}x",
                    f"{c.speedup_einsum:.2f}x",
                )
        except Exception:
            pass

    def _summary_text(self) -> str:
        if not self.benchmarks:
            return "No data."

        lines = []
        total_benchmarks = len(self.benchmarks)
        mlir_wins = sum(
            1 for b in self.benchmarks
            if b.t_mlir and b.t_blas_packed and b.t_mlir < b.t_blas_packed
        )
        avg_mlir_speedup = 0.0
        count = 0
        for b in self.benchmarks:
            if b.speedup_mlir is not None:
                avg_mlir_speedup += b.speedup_mlir
                count += 1
        if count > 0:
            avg_mlir_speedup /= count

        avg_einsum_speedup = 0.0
        count2 = 0
        for b in self.benchmarks:
            if b.speedup_einsum is not None:
                avg_einsum_speedup += b.speedup_einsum
                count2 += 1
        if count2 > 0:
            avg_einsum_speedup /= count2

        avg_sort_speedup = 0.0
        count3 = 0
        for b in self.benchmarks:
            if b.speedup_sort_gemm is not None:
                avg_sort_speedup += b.speedup_sort_gemm
                count3 += 1
        if count3 > 0:
            avg_sort_speedup /= count3

        lines.append(f"  Total benchmarks:              {total_benchmarks}")
        lines.append(f"  MLIR faster than BLAS:          {mlir_wins}/{total_benchmarks}")
        lines.append(f"  Avg MLIR/generic speedup:       {avg_mlir_speedup:.2f}x")
        lines.append(f"  Avg Sort+GEMM/generic speedup:  {avg_sort_speedup:.2f}x")
        lines.append(f"  Avg einsum/generic speedup:     {avg_einsum_speedup:.2f}x")

        if self.cache_hits:
            c = self.cache_hits[0]
            lines.append(f"  Cache-hit MLIR overhead:    {c.t_mlir:.2f} us (N={c.n})")

        return "\n".join(lines)

    # -- Speedups tab --

    def _compose_speedups(self) -> ComposeResult:
        if not self.benchmarks:
            yield Label("No benchmark data found.", classes="no-data")
            return

        for b in self.benchmarks:
            yield Label(f"{b.label} N={b.n}", classes="section-title")

            # Timing bars
            chart_data: list[tuple[str, float, str]] = []
            chart_data.append(("Generic", b.t_generic, "red"))
            if b.t_blas_packed is not None:
                chart_data.append(("BLAS Packed", b.t_blas_packed, "yellow"))
            if b.t_mlir is not None:
                chart_data.append(("MLIR JIT", b.t_mlir, "cyan"))
            if b.t_sort_gemm is not None:
                chart_data.append(("Sort+GEMM", b.t_sort_gemm, "magenta"))
            if b.t_einsum is not None:
                chart_data.append((f"einsum ({b.einsum_alg})", b.t_einsum, "green"))

            yield BarChart(chart_data, unit="us", classes="chart-container")

            # Speedup bars
            speedup_data: list[tuple[str, float, str]] = []
            if b.speedup_blas is not None:
                speedup_data.append(("BLAS/generic", b.speedup_blas, "yellow"))
            if b.speedup_mlir is not None:
                speedup_data.append(("MLIR/generic", b.speedup_mlir, "cyan"))
            if b.speedup_sort_gemm is not None:
                speedup_data.append(("Sort+GEMM/generic", b.speedup_sort_gemm, "magenta"))
            if b.speedup_einsum is not None:
                speedup_data.append(("einsum/generic", b.speedup_einsum, "green"))

            if speedup_data:
                yield SpeedupChart(speedup_data, classes="chart-container")

        # Cache hits
        if self.cache_hits:
            yield Label("Cache Hit Overhead", classes="section-title")
            for c in self.cache_hits:
                chart_data = [
                    ("Generic OMP", c.t_generic, "red"),
                    ("MLIR direct", c.t_mlir, "cyan"),
                    (f"einsum ({c.einsum_alg})", c.t_einsum, "green"),
                ]
                yield BarChart(chart_data, unit="us", classes="cache-box")

    # -- Breakdown tab --

    def _compose_breakdown(self) -> ComposeResult:
        if not self.breakdowns:
            yield Label("No breakdown data found. Run the breakdown benchmark.", classes="no-data")
            return

        for bd in self.breakdowns:
            yield Label(f"Packing vs Micro-kernel Breakdown (N={bd.n})", classes="section-title")

            # Pie-like breakdown as bars
            chart_data = [
                ("Pack A+B", bd.t_pack_only, "yellow"),
                ("Micro-kernel", bd.t_kernel_only, "cyan"),
                ("Total (pack+kern+beta)", bd.t_total, "green"),
            ]
            yield BarChart(chart_data, unit="us", classes="breakdown-box")

            overhead = bd.t_total - bd.t_pack_only - bd.t_kernel_only
            pct_pack = 100.0 * bd.t_pack_only / bd.t_total if bd.t_total > 0 else 0
            pct_kern = 100.0 * bd.t_kernel_only / bd.t_total if bd.t_total > 0 else 0
            pct_over = 100.0 * overhead / bd.t_total if bd.t_total > 0 else 0

            info_lines = [
                f"  MR={bd.mr}  NR={bd.nr}  Total tiles={bd.tiles}",
                f"  Packing:      {bd.t_pack_only:>10.2f} us  ({pct_pack:.1f}%)",
                f"  Micro-kernel: {bd.t_kernel_only:>10.2f} us  ({pct_kern:.1f}%)",
                f"  Overhead:     {overhead:>10.2f} us  ({pct_over:.1f}%)  (beta scaling, loop control, etc.)",
            ]
            yield Static("\n".join(info_lines), classes="summary-box")

    # -- Raw output tab --

    def _compose_raw(self) -> ComposeResult:
        if self.raw_text:
            yield Label("Raw Benchmark Output", classes="section-title")
            yield Static(self.raw_text, classes="summary-box")
        else:
            yield Label("No raw output captured.", classes="no-data")

    # -- Actions --

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_tab_overview(self) -> None:
        self.query_one(TabbedContent).active = "tab-overview"

    def action_tab_speedups(self) -> None:
        self.query_one(TabbedContent).active = "tab-speedups"

    def action_tab_breakdown(self) -> None:
        self.query_one(TabbedContent).active = "tab-breakdown"

    def action_tab_raw(self) -> None:
        self.query_one(TabbedContent).active = "tab-raw"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Einsums PackedGemm benchmark viewer TUI"
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Path to a file containing benchmark output (reads stdin if omitted and not a tty)"
    )
    parser.add_argument(
        "--run", metavar="EXECUTABLE",
        help="Run the benchmark executable and display results"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Show the viewer with sample data"
    )
    args = parser.parse_args()

    raw_text = ""

    if args.demo:
        raw_text = _DEMO_DATA
    elif args.run:
        exe = Path(args.run)
        if not exe.exists():
            print(f"Error: executable not found: {exe}", file=sys.stderr)
            sys.exit(1)
        result = subprocess.run(
            [str(exe), "--einsums:no-install-signal-handlers", "--einsums:no-profiler-report"],
            capture_output=True, text=True
        )
        raw_text = result.stdout + result.stderr
    elif args.input:
        raw_text = Path(args.input).read_text()
    elif not sys.stdin.isatty():
        raw_text = sys.stdin.read()
    else:
        # No input: show demo
        print("No input provided. Use --demo for sample data, pipe benchmark output, or pass a file.")
        print("Usage: python bench_viewer.py [results.txt | --run EXECUTABLE | --demo]")
        sys.exit(1)

    benchmarks, breakdowns, cache_hits = parse_output(raw_text)

    if not benchmarks and not breakdowns and not cache_hits:
        print("Warning: no benchmark results parsed from input.", file=sys.stderr)
        print("Showing raw output only.", file=sys.stderr)

    app = BenchViewerApp(benchmarks, breakdowns, cache_hits, raw_text=raw_text)
    app.run()


_DEMO_DATA = """\
[Rank-2 N=64] Generic: 1523.45 \u00b5s  BLAS packed: 245.67 \u00b5s  MLIR: 198.34 \u00b5s  einsum (BLAS-GEMM): 34.56 \u00b5s  BLAS/gen: 6.20x  MLIR/gen: 7.68x  einsum/gen: 44.08x
[Rank-3 N=32] Generic: 8234.12 \u00b5s  BLAS packed: 1456.78 \u00b5s  MLIR: 1234.56 \u00b5s  einsum (BLAS-GEMM): 987.65 \u00b5s  BLAS/gen: 5.65x  MLIR/gen: 6.67x  einsum/gen: 8.34x
[Rank-3 N=64] Generic: 65432.10 \u00b5s  BLAS packed: 9876.54 \u00b5s  MLIR: 8765.43 \u00b5s  einsum (BLAS-GEMM): 7654.32 \u00b5s  BLAS/gen: 6.63x  MLIR/gen: 7.47x  einsum/gen: 8.55x
[Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k] N=8] Generic: 45.67 \u00b5s  BLAS packed: 23.45 \u00b5s  MLIR: 19.87 \u00b5s  einsum (generic/MLIR): 20.12 \u00b5s  BLAS/gen: 1.95x  MLIR/gen: 2.30x  einsum/gen: 2.27x
[Rank-2 N=256] Generic: 98765.43 \u00b5s  BLAS packed: 5432.10 \u00b5s  MLIR: 4567.89 \u00b5s  einsum (BLAS-GEMM): 2345.67 \u00b5s  BLAS/gen: 18.18x  MLIR/gen: 21.62x  einsum/gen: 42.10x
[Rank-3 N=128] Generic: 543210.00 \u00b5s  BLAS packed: 34567.89 \u00b5s  MLIR: 29876.54 \u00b5s  einsum (BLAS-GEMM): 28765.43 \u00b5s  BLAS/gen: 15.72x  MLIR/gen: 18.18x  einsum/gen: 18.88x
[Hadamard C[i]+=A[i,i]*B[i] N=256] Generic: 12.34 \u00b5s  einsum (generic/MLIR): 5.67 \u00b5s  einsum/gen: 2.18x
[Hadamard C[i,j]+=A[i,i,j]*B[j] N=64] Generic: 456.78 \u00b5s  einsum (generic/MLIR): 234.56 \u00b5s  einsum/gen: 1.95x
[Scalar s+=A[i,j]*B[i,j] N=128] Generic: 89.12 \u00b5s  einsum (generic/MLIR): 45.67 \u00b5s  einsum/gen: 1.95x
[Scalar s+=A[i,j,k]*B[i,j,k] N=32] Generic: 234.56 \u00b5s  einsum (generic/MLIR): 123.45 \u00b5s  einsum/gen: 1.90x
[Asymmetric C[i,j,k]+=A[i,l]*B[l,j,k] N=32] Generic: 5678.90 \u00b5s  einsum (generic/MLIR): 2345.67 \u00b5s  einsum/gen: 2.42x
[Batch C[b,i,j]+=A[b,i,k]*B[b,k,j] B=16 N=32] Generic: 3456.78 \u00b5s  einsum (generic/MLIR): 1234.56 \u00b5s  einsum/gen: 2.80x
[C[i,k]+=A[i,j,k,l]*B[j,l] N=16] Generic: 987.65 \u00b5s  einsum (generic/MLIR): 456.78 \u00b5s  einsum/gen: 2.16x
[Outer product C[i,j]+=A[i]*B[j] N=256] Generic: 345.67 \u00b5s  einsum (BLAS-GER): 12.34 \u00b5s  einsum/gen: 28.01x
[Tensor dot C[i,l]+=A[i,j,k]*B[k,j,l] N=32] Generic: 8765.43 \u00b5s  BLAS packed: 1567.89 \u00b5s  MLIR: 1345.67 \u00b5s  einsum (generic/MLIR): 1234.56 \u00b5s  BLAS/gen: 5.59x  MLIR/gen: 6.51x  einsum/gen: 7.10x
[Large-K C[i,j]+=A[i,k]*B[k,j] M=64 N=64 K=2048] Generic: 45678.90 \u00b5s  BLAS packed: 2345.67 \u00b5s  MLIR: 1987.65 \u00b5s  einsum (BLAS-GEMM): 1234.56 \u00b5s  BLAS/gen: 19.47x  MLIR/gen: 22.98x  einsum/gen: 37.00x
[Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k] N=16] Generic: 1234.56 \u00b5s  BLAS packed: 345.67 \u00b5s  MLIR: 298.76 \u00b5s  einsum (generic/MLIR): 312.45 \u00b5s  BLAS/gen: 3.57x  MLIR/gen: 4.13x  einsum/gen: 3.95x
[Breakdown N=128] pack_only: 4532.10 \u00b5s  kernel_only: 23456.78 \u00b5s  total: 29876.54 \u00b5s  (MR=16 NR=6 tiles=1536)
[Cache hit N=32] Generic OMP: 234.56 \u00b5s  MLIR direct: 89.12 \u00b5s  einsum (BLAS-GEMM): 12.34 \u00b5s  MLIR/generic: 2.63x  einsum/generic: 19.01x
[Rank-3 C[i,l]+=A[i,k,j]*B[l,j,k] N=32]
  Generic:      8234.12 \u00b5s
  MLIR:         1345.67 \u00b5s  (6.12x vs generic)
  Sort+GEMM:     987.65 \u00b5s  (8.34x vs generic)
  einsum(generic/MLIR): 1234.56 \u00b5s  (6.67x vs generic)
[Rank-3 C[i,l]+=A[i,k,j]*B[l,j,k] N=64]
  Generic:     65432.10 \u00b5s
  MLIR:         9876.54 \u00b5s  (6.63x vs generic)
  Sort+GEMM:    7654.32 \u00b5s  (8.55x vs generic)
  einsum(generic/MLIR): 8765.43 \u00b5s  (7.47x vs generic)
[Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k] N=8]
  Generic:        45.67 \u00b5s
  MLIR:           19.87 \u00b5s  (2.30x vs generic)
  Sort+GEMM:      23.45 \u00b5s  (1.95x vs generic)
  einsum(generic/MLIR): 20.12 \u00b5s  (2.27x vs generic)
[Rank-4 rect C[i,j]+=A[i,k,l,m]*B[j,m,k,l] (32x32x16x16x8) N=0]
  Generic:      2345.67 \u00b5s
  MLIR:          456.78 \u00b5s  (5.13x vs generic)
  Sort+GEMM:     345.67 \u00b5s  (6.79x vs generic)
  einsum(generic/MLIR): 498.76 \u00b5s  (4.70x vs generic)
[Rank-3 scrambled C[j,i]+=A[k,i,l]*B[l,j,k] N=32]
  Generic:      8765.43 \u00b5s
  MLIR:         1567.89 \u00b5s  (5.59x vs generic)
  Sort+GEMM:    1234.56 \u00b5s  (7.10x vs generic)
  einsum(generic/MLIR): 1345.67 \u00b5s  (6.51x vs generic)
"""


if __name__ == "__main__":
    main()
