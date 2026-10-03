# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""The per-thread call-tree table."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable

from ..analysis import (
    SORT_KEYS,
    FlatRow,
    aggregate_flat,
    all_paths_with_children,
    bottom_up,
    child_path,
    flatten_tree,
    hot_path,
    is_anomaly,
    name_matches,
    sort_tree,
    sum_exclusive,
)
from ..format import format_bytes, heat, make_bar
from ..model import ProfileNode

COLUMNS = ("", "%", "% parent", "bar", "excl(ms)", "incl(ms)", "count", "mean(ms)", "alloc", "peak", "⚑", "anom", "name")
VIEW_MODES = ("tree", "flat", "bottom-up")


@dataclass
class ViewOptions:
    """How every thread table of a session is shown; shared, so a toggle applies to all of them."""

    mode: str = "tree"
    sort_key: str = "natural"
    hot_path: bool = False
    heat: bool = True
    name_filter: str = ""
    bookmarks: set[str] = field(default_factory=set)


class ProfileTable(DataTable):
    """A DataTable whose space, enter and click toggle the row's subtree."""

    class ToggleRequest(Message):
        def __init__(self, table: ProfileTable, row: int) -> None:
            super().__init__()
            self.table = table
            self.row = row

    BINDINGS = [Binding("space,enter", "toggle_row", "Expand/collapse", show=False)]

    def action_toggle_row(self) -> None:
        self.post_message(self.ToggleRequest(self, self.cursor_row))

    def on_click(self) -> None:
        # The table moves its cursor on the same click, so read the row afterwards.
        self.call_after_refresh(lambda: self.post_message(self.ToggleRequest(self, self.cursor_row)))


class ThreadTreeView(Widget):
    """One thread's zones as a table: tree, flat (per name), or bottom-up (callers under callees)."""

    DEFAULT_CSS = """
    ThreadTreeView { height: 1fr; }
    ThreadTreeView ProfileTable { height: 1fr; }
    """

    def __init__(self, thread_id: str, options: ViewOptions, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.thread_id = thread_id
        self.options = options
        self.roots: list[ProfileNode] = []
        self.total_ms = 1.0
        self._collapsed: set[str] = set()
        self._rows: list[FlatRow] = []
        self._row_keys: list[Any] = []
        self._col_keys: list[Any] = []

    def compose(self) -> ComposeResult:
        yield ProfileTable(cursor_type="row")

    def on_mount(self) -> None:
        self._col_keys = list(self.query_one(ProfileTable).add_columns(*COLUMNS))
        # A snapshot can arrive before the table is mounted (a program run with
        # a wait-for-viewer setting streams one the instant the viewer attaches);
        # it was kept on self, so draw it now.
        self.refresh_rows()

    @property
    def table(self) -> ProfileTable:
        return self.query_one(ProfileTable)

    def set_roots(self, roots: list[ProfileNode]) -> None:
        self.roots = roots
        self.refresh_rows()

    def node_at(self, row: int) -> ProfileNode | None:
        return self._rows[row].node if 0 <= row < len(self._rows) else None

    @property
    def selected(self) -> ProfileNode | None:
        try:
            return self.node_at(self.table.cursor_row)
        except NoMatches:
            return None

    def share_of_total(self, node: ProfileNode) -> float:
        return node.exclusive_ms / self.total_ms * 100.0 if self.total_ms > 0 else 0.0

    # -- building rows ------------------------------------------------------

    def _build_rows(self) -> list[FlatRow]:
        opts = self.options
        if opts.mode == "flat":
            return [
                FlatRow(0, n, n.name, False, 0.0)
                for n in sort_tree(aggregate_flat(self.roots), opts.sort_key)
                if name_matches(n.name, opts.name_filter)
            ]
        if opts.mode == "bottom-up":
            rows = []
            for callee in sort_tree(bottom_up(self.roots), opts.sort_key):
                if not name_matches(callee.name, opts.name_filter):
                    continue
                collapsed = callee.name in self._collapsed
                rows.append(FlatRow(0, callee, callee.name, collapsed and bool(callee.children), 0.0))
                if not collapsed:
                    rows += [
                        FlatRow(1, caller, child_path(callee.name, caller.name), False, callee.inclusive_ms)
                        for caller in callee.children
                    ]
            return rows
        return flatten_tree(sort_tree(self.roots, opts.sort_key), opts.name_filter, self._collapsed)

    def _cells(self, row: FlatRow, hot: bool) -> tuple[str, ...]:
        node = row.node
        pct = self.share_of_total(node)
        pct_parent = node.exclusive_ms / row.parent_inclusive_ms * 100.0 if row.parent_inclusive_ms > 0 else pct
        marker = (">" if row.collapsed else "v ") if node.children else "  "
        name = "  " * row.depth + marker + node.name
        excl, incl, mean = f"{node.exclusive_ms:10.3f}", f"{node.inclusive_ms:10.3f}", f"{node.mean_ms:8.3f}"
        if self.options.heat:
            excl, incl, mean, name = (heat(s, pct) for s in (excl, incl, mean, name))
        if hot:
            name = f"[bold magenta]* {name}[/]"
        return (
            ">" if row.collapsed else ("v" if node.children else " "),
            heat(f"{pct:5.1f}%", pct),
            heat(f"{pct_parent:5.1f}%", pct_parent),
            heat(make_bar(pct), pct),
            excl,
            incl,
            str(node.call_count),
            mean,
            format_bytes(node.mem_alloc_bytes),
            format_bytes(node.mem_peak_bytes),
            "⚑" if node.name in self.options.bookmarks else "",
            "!" if is_anomaly(node) else "",
            name,
        )

    def refresh_rows(self) -> None:
        """Redraw the table in place: rows that still exist are updated, so the cursor stays put."""
        try:
            table = self.table
        except NoMatches:
            return
        if not self._col_keys:
            return
        self.total_ms = sum_exclusive(self.roots) or 1.0
        rows = self._build_rows()
        hot = hot_path(sort_tree(self.roots, self.options.sort_key)) if self.options.hot_path and self.options.mode == "tree" else set()

        old_count = len(self._row_keys)
        follow_tail = old_count > 0 and table.cursor_row >= old_count - 1
        keys = self._row_keys[: len(rows)]
        for i, row in enumerate(rows):
            cells = self._cells(row, row.path in hot)
            if i < old_count:
                for col, value in zip(self._col_keys, cells):
                    table.update_cell(self._row_keys[i], col, value)
            else:
                keys.append(table.add_row(*cells))
        for key in self._row_keys[len(rows):]:
            table.remove_row(key)
        self._rows, self._row_keys = rows, keys
        if follow_tail and len(rows) > old_count:
            table.move_cursor(row=len(rows) - 1)

    # -- commands -----------------------------------------------------------

    def toggle(self, row: int) -> None:
        node = self.node_at(row)
        if node is None or not node.children:
            return
        self._collapsed ^= {self._rows[row].path}
        self.refresh_rows()

    def expand_all(self) -> None:
        self._collapsed.clear()
        self.refresh_rows()

    def collapse_all(self) -> None:
        self._collapsed = all_paths_with_children(self.roots)
        self.refresh_rows()

    def matching_rows(self, pattern: str) -> list[int]:
        return [i for i, row in enumerate(self._rows) if pattern and name_matches(row.node.name, pattern)]

    def bookmarked_rows(self) -> list[int]:
        return [i for i, row in enumerate(self._rows) if row.node.name in self.options.bookmarks]

    def jump(self, rows: list[int], forward: bool = True) -> bool:
        """Move to the next (or previous) of *rows* after the cursor, wrapping around."""
        if not rows:
            return False
        current = self.table.cursor_row
        if forward:
            target = next((r for r in rows if r > current), rows[0])
        else:
            target = next((r for r in reversed(rows) if r < current), rows[-1])
        self.table.move_cursor(row=target)
        return True


def next_in(cycle: tuple[str, ...], current: str) -> str:
    return cycle[(cycle.index(current) + 1) % len(cycle)] if current in cycle else cycle[0]


__all__ = ["ProfileTable", "SORT_KEYS", "ThreadTreeView", "VIEW_MODES", "ViewOptions", "next_in"]
