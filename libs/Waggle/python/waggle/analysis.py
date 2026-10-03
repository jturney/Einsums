# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Views computed from a call tree: flattening, aggregation, bottom-up, hot path, roofline."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .model import ProfileNode, ProfileSnapshot


def walk(nodes: list[ProfileNode], depth: int = 0) -> Iterator[tuple[ProfileNode, int]]:
    """Every node under *nodes*, depth first, with its depth."""
    for node in nodes:
        yield node, depth
        yield from walk(node.children, depth + 1)


def sum_exclusive(nodes: list[ProfileNode]) -> float:
    return sum(node.exclusive_ms for node, _ in walk(nodes))


def sum_mem_current(nodes: list[ProfileNode]) -> int:
    return sum(node.mem_current_bytes for node, _ in walk(nodes))


def name_matches(name: str, pattern: str) -> bool:
    """Case-insensitive regex match, falling back to a substring test for an invalid regex."""
    if not pattern:
        return True
    try:
        return re.search(pattern, name, re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in name.lower()


def find_node(snap: ProfileSnapshot, name: str) -> ProfileNode | None:
    """The first node called *name* in any thread, depth first."""
    return next((node for node, _ in walk(snap.all_roots()) if node.name == name), None)


def exclusive_by_name(snap: ProfileSnapshot) -> dict[str, float]:
    """Exclusive time per zone name (the last node of a name wins, as the history has always kept)."""
    return {node.name: node.exclusive_ms for node, _ in walk(snap.all_roots())}


def is_anomaly(node: ProfileNode) -> bool:
    """Whether the slowest or fastest call lies more than two standard deviations from the mean."""
    if node.stddev_ms <= 0 or node.call_count <= 0:
        return False
    mean = node.mean_ms
    if node.exclusive_max_ms > mean + 2 * node.stddev_ms:
        return True
    return 0 < node.exclusive_min_ms < mean - 2 * node.stddev_ms


def numeric_annotation(annotations: dict[str, Any], key: str) -> float | None:
    """A numeric annotation's value: the plain value, or ``avg`` of a min/max/avg record."""
    value = annotations.get(key)
    if isinstance(value, dict):
        value = value.get("avg")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class DerivedRates:
    gflops: float | None = None
    gb_per_s: float | None = None
    intensity: float | None = None  # FLOP/byte


def derived_rates(node: ProfileNode) -> DerivedRates:
    """GFLOP/s, GB/s and arithmetic intensity from the ``flops``/``bytes_*`` annotations."""
    rates = DerivedRates()
    if node.exclusive_ms <= 0:
        return rates
    seconds = node.exclusive_ms / 1000.0
    flops = numeric_annotation(node.annotations, "flops")
    read = numeric_annotation(node.annotations, "bytes_read")
    written = numeric_annotation(node.annotations, "bytes_written")
    if flops is not None:
        rates.gflops = flops / seconds / 1e9
    if read is not None or written is not None:
        total = (read or 0.0) + (written or 0.0)
        rates.gb_per_s = total / seconds / 1e9
        if flops is not None and total > 0:
            rates.intensity = flops / total
    return rates


# ---------------------------------------------------------------------------
# Tree flattening for the table
# ---------------------------------------------------------------------------


@dataclass
class FlatRow:
    depth: int
    node: ProfileNode
    path: str  # "/"-joined names from the root, the key for collapse state
    collapsed: bool
    parent_inclusive_ms: float  # 0 for a root


def child_path(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def flatten_tree(
    nodes: list[ProfileNode],
    name_filter: str = "",
    collapsed: set[str] | frozenset[str] = frozenset(),
    *,
    depth: int = 0,
    prefix: str = "",
    parent_inclusive_ms: float = 0.0,
) -> list[FlatRow]:
    """The rows of a tree view. A node that fails the filter stays when a descendant passes it."""
    rows: list[FlatRow] = []
    for node in nodes:
        path = child_path(prefix, node.name)
        is_collapsed = path in collapsed
        child_rows = (
            []
            if is_collapsed
            else flatten_tree(
                node.children,
                name_filter,
                collapsed,
                depth=depth + 1,
                prefix=path,
                parent_inclusive_ms=node.inclusive_ms,
            )
        )
        if child_rows or name_matches(node.name, name_filter):
            rows.append(FlatRow(depth, node, path, is_collapsed and bool(node.children), parent_inclusive_ms))
            rows.extend(child_rows)
    return rows


def all_paths_with_children(nodes: list[ProfileNode], prefix: str = "") -> set[str]:
    """The path of every node that has children, i.e. everything "collapse all" collapses."""
    paths: set[str] = set()
    for node in nodes:
        if node.children:
            path = child_path(prefix, node.name)
            paths.add(path)
            paths |= all_paths_with_children(node.children, path)
    return paths


def hot_path(nodes: list[ProfileNode], prefix: str = "") -> set[str]:
    """The paths along the most expensive (exclusive time) child at each level."""
    if not nodes:
        return set()
    hottest = max(nodes, key=lambda n: n.exclusive_ms)
    path = child_path(prefix, hottest.name)
    return {path} | hot_path(hottest.children, path)


SORT_KEYS = ("natural", "exclusive", "count", "mean", "name")


def sort_tree(nodes: list[ProfileNode], key: str) -> list[ProfileNode]:
    """A sorted copy of the tree; the snapshot itself is left alone. Names ascend, numbers descend."""
    from copy import copy

    result = []
    for node in nodes:
        clone = copy(node)
        clone.children = sort_tree(node.children, key)
        result.append(clone)
    if key == "natural":
        return result
    if key == "name":
        return sorted(result, key=lambda n: n.name.lower())
    metric = {"exclusive": lambda n: n.exclusive_ms, "count": lambda n: n.call_count, "mean": lambda n: n.mean_ms}
    return sorted(result, key=metric.get(key, metric["exclusive"]), reverse=True)


# ---------------------------------------------------------------------------
# Aggregated views
# ---------------------------------------------------------------------------


def _merge_into(total: ProfileNode, node: ProfileNode) -> None:
    total.exclusive_ms += node.exclusive_ms
    total.inclusive_ms = max(total.inclusive_ms, node.inclusive_ms)
    total.call_count += node.call_count
    if node.exclusive_min_ms > 0:
        total.exclusive_min_ms = min(total.exclusive_min_ms, node.exclusive_min_ms) if total.exclusive_min_ms > 0 else node.exclusive_min_ms
    total.exclusive_max_ms = max(total.exclusive_max_ms, node.exclusive_max_ms)
    total.stddev_ms = max(total.stddev_ms, node.stddev_ms)
    total.mem_alloc_count += node.mem_alloc_count
    total.mem_free_count += node.mem_free_count
    total.mem_alloc_bytes += node.mem_alloc_bytes
    total.mem_free_bytes += node.mem_free_bytes
    total.mem_current_bytes += node.mem_current_bytes
    total.mem_peak_bytes = max(total.mem_peak_bytes, node.mem_peak_bytes)
    for key, value in node.annotations.items():
        old = total.annotations.get(key)
        if isinstance(value, dict) and isinstance(old, dict):
            merged = dict(old)
            for stat, combine in (("min", min), ("max", max), ("avg", lambda a, b: a + b)):
                if stat in value and stat in merged:
                    try:
                        merged[stat] = combine(float(merged[stat]), float(value[stat]))
                    except (TypeError, ValueError):
                        pass
            total.annotations[key] = merged
        else:
            total.annotations[key] = value
    total.counters.update(node.counters)


def aggregate_flat(nodes: list[ProfileNode]) -> list[ProfileNode]:
    """One node per zone name, wherever it was called from (sums times, calls and bytes)."""
    totals: dict[str, ProfileNode] = {}
    for node, _ in walk(nodes):
        total = totals.get(node.name)
        if total is None:
            totals[node.name] = ProfileNode(
                name=node.name,
                call_count=node.call_count,
                exclusive_ms=node.exclusive_ms,
                inclusive_ms=node.inclusive_ms,
                exclusive_min_ms=node.exclusive_min_ms,
                exclusive_max_ms=node.exclusive_max_ms,
                stddev_ms=node.stddev_ms,
                file=node.file,
                line=node.line,
                function=node.function,
                annotations=dict(node.annotations),
                counters=dict(node.counters),
                mem_alloc_count=node.mem_alloc_count,
                mem_free_count=node.mem_free_count,
                mem_alloc_bytes=node.mem_alloc_bytes,
                mem_free_bytes=node.mem_free_bytes,
                mem_current_bytes=node.mem_current_bytes,
                mem_peak_bytes=node.mem_peak_bytes,
            )
        else:
            _merge_into(total, node)
    return list(totals.values())


def bottom_up(nodes: list[ProfileNode]) -> list[ProfileNode]:
    """Each zone as a root, with the zones that called it as its children (heaviest caller first)."""
    totals: dict[str, ProfileNode] = {}
    callers: dict[str, dict[str, ProfileNode]] = {}

    def visit(level: list[ProfileNode], caller: str) -> None:
        for node in level:
            total = totals.setdefault(node.name, ProfileNode(name=node.name))
            total.exclusive_ms += node.exclusive_ms
            total.inclusive_ms = max(total.inclusive_ms, node.inclusive_ms)
            total.call_count += node.call_count
            if caller:
                entry = callers.setdefault(node.name, {}).setdefault(caller, ProfileNode(name=caller))
                entry.exclusive_ms += node.exclusive_ms
                entry.inclusive_ms += node.exclusive_ms
                entry.call_count += node.call_count
            visit(node.children, node.name)

    visit(nodes, "")
    for name, total in totals.items():
        total.children = sorted(callers.get(name, {}).values(), key=lambda n: n.exclusive_ms, reverse=True)
    return list(totals.values())


@dataclass
class RooflinePoint:
    name: str
    intensity: float  # FLOP/byte
    gflops: float
    exclusive_ms: float


def roofline_points(nodes: list[ProfileNode]) -> list[RooflinePoint]:
    """Zones annotated with ``flops`` and ``bytes_read``/``bytes_written``."""
    points = []
    for node, _ in walk(nodes):
        rates = derived_rates(node)
        if rates.intensity is not None and rates.gflops is not None and rates.gflops > 0:
            points.append(RooflinePoint(node.name, rates.intensity, rates.gflops, node.exclusive_ms))
    return points


@dataclass
class CompareRow:
    name: str
    exclusive_a: float
    exclusive_b: float
    calls_a: int
    calls_b: int

    @property
    def delta_ms(self) -> float:
        return self.exclusive_b - self.exclusive_a

    @property
    def delta_pct(self) -> float:
        return self.delta_ms / self.exclusive_a * 100.0 if self.exclusive_a > 0 else 0.0


def compare_snapshots(a: ProfileSnapshot | None, b: ProfileSnapshot | None) -> list[CompareRow]:
    """Flat per-name comparison of two snapshots, largest absolute change first."""
    flat_a = {n.name: n for n in aggregate_flat(a.all_roots() if a else [])}
    flat_b = {n.name: n for n in aggregate_flat(b.all_roots() if b else [])}
    rows = []
    for name in flat_a.keys() | flat_b.keys():
        na, nb = flat_a.get(name), flat_b.get(name)
        rows.append(
            CompareRow(
                name,
                na.exclusive_ms if na else 0.0,
                nb.exclusive_ms if nb else 0.0,
                na.call_count if na else 0,
                nb.call_count if nb else 0,
            )
        )
    rows.sort(key=lambda r: abs(r.delta_ms), reverse=True)
    return rows


@dataclass
class CounterSummary:
    cycles: int = 0
    instructions: int = 0
    cache_misses: int = 0
    branch_misses: int = 0

    @property
    def ipc(self) -> float:
        return self.instructions / self.cycles if self.cycles > 0 else 0.0

    @property
    def cache_misses_per_k(self) -> float:
        return self.cache_misses / self.instructions * 1000.0 if self.instructions > 0 else 0.0

    @property
    def branch_misses_per_k(self) -> float:
        return self.branch_misses / self.instructions * 1000.0 if self.instructions > 0 else 0.0

    def classify(self) -> tuple[str, str]:
        """A rough top-down stall class and the color to show it in."""
        if self.ipc >= 2.0:
            return "Retiring", "green"
        if self.branch_misses_per_k > 5.0:
            return "Bad Speculation", "red"
        if self.cache_misses_per_k > 10.0:
            return "Back-end Bound (memory)", "yellow"
        if self.ipc < 0.5:
            return "Front-end Bound", "magenta"
        return "Compute Bound", "cyan"


def summarize_counters(counters: dict[str, Any]) -> CounterSummary:
    summary = CounterSummary()
    for name, data in counters.items():
        total = int(data.get("total", 0)) if isinstance(data, dict) else 0
        lower = name.lower()
        if "instruction" in lower:
            summary.instructions = total
        elif "cycle" in lower:
            summary.cycles = total
        elif "cache" in lower and "miss" in lower:
            summary.cache_misses = total
        elif "branch" in lower and "miss" in lower:
            summary.branch_misses = total
    return summary
