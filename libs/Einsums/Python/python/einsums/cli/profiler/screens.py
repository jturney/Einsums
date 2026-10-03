# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Modal screens: file and endpoint prompts, confirmation, session pickers, comparison, help."""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, SelectionList, Static

from .analysis import compare_snapshots
from .session import Session


class Dialog(ModalScreen):
    """A centered box; Escape dismisses it with ``None`` (``False`` for a confirmation)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CANCEL_RESULT: Any = None

    def action_cancel(self) -> None:
        self.dismiss(self.CANCEL_RESULT)


class PromptDialog(Dialog):
    """Ask for one line of text: a filename or an endpoint."""

    def __init__(self, title: str, value: str = "", placeholder: str = "", ok: str = "OK", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._title, self._value, self._placeholder, self._ok = title, value, placeholder, ok

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self._title)
            yield Input(value=self._value, placeholder=self._placeholder)
            with Horizontal(classes="buttons"):
                yield Button(self._ok, variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def _submit(self) -> None:
        if value := self.query_one(Input).value.strip():
            self.dismiss(value)

    def on_input_submitted(self) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self._submit() if event.button.id == "ok" else self.dismiss(None)


class ConfirmDialog(Dialog):
    CANCEL_RESULT = False

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self._message)
            with Horizontal(classes="buttons"):
                yield Button("Yes", variant="primary", id="yes")
                yield Button("No", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class SessionsDialog(Dialog):
    """Pick sessions; with *filename* set, also ask where to save them.

    Dismisses with the chosen session ids, or ``(ids, filename)`` when saving.
    """

    def __init__(
        self,
        title: str,
        sessions: list[Session],
        *,
        exactly: int | None = None,
        preselect: bool = False,
        filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._title, self._sessions, self._exactly = title, sessions, exactly
        self._preselect, self._filename = preselect, filename

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide"):
            yield Label(self._title)
            yield SelectionList[str](*((escape(s.label), s.session_id, self._preselect) for s in self._sessions))
            if self._filename is not None:
                yield Label("File")
                yield Input(value=self._filename)
            with Horizontal(classes="buttons"):
                yield Button("OK", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_input_submitted(self) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self._submit() if event.button.id == "ok" else self.dismiss(None)

    def _submit(self) -> None:
        chosen = list(self.query_one(SelectionList).selected)
        if self._exactly is not None and len(chosen) != self._exactly:
            self.notify(f"Select exactly {self._exactly} sessions", severity="warning")
            return
        if not chosen:
            self.notify("Select at least one session", severity="warning")
            return
        if self._filename is None:
            self.dismiss(chosen)
        elif path := self.query_one(Input).value.strip():
            self.dismiss((chosen, path))


class CompareScreen(Dialog):
    """Per-zone exclusive time and calls of two sessions, largest change first."""

    def __init__(self, a: Session, b: Session, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._a, self._b = a, b

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog large"):
            yield Label(f"[bold]A:[/bold] {escape(self._a.label)}    [bold]B:[/bold] {escape(self._b.label)}")
            yield DataTable(cursor_type="row")
            with Horizontal(classes="buttons"):
                yield Button("Close", id="cancel")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("name", "excl A(ms)", "excl B(ms)", "Δ(ms)", "Δ%", "calls A", "calls B", "Δcalls")
        for row in compare_snapshots(self._a.snapshot, self._b.snapshot):
            color = "green" if row.delta_ms < -0.01 else "red" if row.delta_ms > 0.01 else ""
            wrap = (lambda s: f"[{color}]{s}[/]") if color else (lambda s: s)
            table.add_row(
                escape(row.name),
                f"{row.exclusive_a:.3f}",
                f"{row.exclusive_b:.3f}",
                wrap(f"{row.delta_ms:+.3f}"),
                wrap(f"{row.delta_pct:+.1f}%"),
                str(row.calls_a),
                str(row.calls_b),
                f"{row.calls_b - row.calls_a:+d}",
            )
        table.focus()

    def on_button_pressed(self) -> None:
        self.dismiss(None)


COLUMN_HELP = """\
[bold underline]Table columns[/]
  [bold]%[/]          share of the thread's total exclusive time
  [bold]% parent[/]   exclusive time as a share of the parent's inclusive time
  [bold]excl(ms)[/]   time in this zone itself, not its children
  [bold]incl(ms)[/]   time in this zone including its children
  [bold]count[/]      times the zone was entered
  [bold]mean(ms)[/]   exclusive time per call
  [bold]alloc[/]      bytes allocated inside the zone
  [bold]peak[/]       peak net bytes (allocated minus freed) inside the zone
  [bold]⚑[/]          bookmarked
  [bold]anom[/]       the slowest or fastest call is more than 2 standard deviations from the mean
"""

PANEL_HELP = """\
[bold underline]Panels[/]
  [bold]Hardware counters[/] derive IPC and cache/branch misses per 1000 instructions from
  Linux perf_event counters, and classify the zone (retiring, compute bound, front-end bound,
  back-end bound, bad speculation).
  [bold]Disassembly[/] runs llvm-objdump (or objdump) on the executable and the shared libraries
  beside it and in ../lib, so it needs the binaries on this machine.
  [bold]Roofline[/] plots zones annotated with flops and bytes_read/bytes_written.
"""


class HelpScreen(Dialog):
    BINDINGS = [Binding("escape,question_mark,q", "cancel", "Close")]

    def __init__(self, keymap: list[tuple[str, list[tuple[str, str]]]], *, title: str = "Waggle profiler", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._keymap = keymap
        self._title = title

    def compose(self) -> ComposeResult:
        lines = ["[bold underline]Keys[/]  (also searchable in the command palette, ctrl+p)"]
        for section, keys in self._keymap:
            lines.append(f"\n[bold]{section}[/]")
            lines += [f"  [bold cyan]{escape(key):<10}[/] {escape(desc)}" for key, desc in keys]
        with Vertical(classes="dialog large"):
            yield Label(f"[bold]{escape(self._title)}[/bold]")
            with VerticalScroll():
                yield Static("\n".join(lines) + "\n\n" + COLUMN_HELP + "\n" + PANEL_HELP)


class GraphScreen(Dialog):
    """The session's compute graphs: a tree of graphs, nodes and tensors, and the selection's details."""

    def __init__(self, title: str, graphs: list, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._title, self._graphs = title, graphs

    def compose(self) -> ComposeResult:
        from textual.widgets import Tree

        with Vertical(classes="dialog large"):
            yield Label(f"[bold]Compute graphs[/bold]  {escape(self._title)}")
            with Horizontal(classes="graph-panes"):
                yield Tree("graphs", id="graph-tree")
                with VerticalScroll(id="graph-detail"):
                    yield Static("", id="graph-detail-text")

    def on_mount(self) -> None:
        from textual.widgets import Tree

        from .graphs import graph_summary

        tree = self.query_one("#graph-tree", Tree)
        tree.show_root = False
        for graph in self._graphs:
            timing = f"  {graph.total_ms:.2f} ms" if graph.total_ms else ""
            branch = tree.root.add(f"{escape(graph.name)}  ({len(graph.nodes)} nodes){timing}", data=(graph, None), expand=False)
            for node in graph.nodes:
                when = f"  {node.timing_ms:.3f} ms" if node.timing_ms is not None else ""
                leaf = branch.add(f"#{node.id} {escape(node.title)}{when}", data=(graph, node))
                for tid in node.inputs:
                    leaf.add_leaf(f"in  {escape(graph.tensor_name(tid))}", data=(graph, node))
                for tid in node.outputs:
                    leaf.add_leaf(f"out {escape(graph.tensor_name(tid))}", data=(graph, node))
        if self._graphs:
            self.query_one("#graph-detail-text", Static).update(graph_summary(self._graphs[0]))
        tree.focus()

    def on_tree_node_highlighted(self, event) -> None:
        from .graphs import graph_summary, node_summary

        if event.node.data is None:
            return
        graph, node = event.node.data
        text = graph_summary(graph) if node is None else node_summary(graph, node)
        self.query_one("#graph-detail-text", Static).update(text)
