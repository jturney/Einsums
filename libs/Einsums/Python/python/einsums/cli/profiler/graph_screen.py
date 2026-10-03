# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""The compute-graph screen the Einsums plugin opens in Waggle's viewer."""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Label, Static

from waggle.screens import Dialog


class GraphScreen(Dialog):
    """The session's compute graphs: a tree of graphs, nodes and tensors, and the selection's details."""

    DEFAULT_CSS = """
    GraphScreen .graph-panes {
        height: 1fr;
    }
    GraphScreen #graph-tree {
        width: 1fr;
        height: 1fr;
    }
    GraphScreen #graph-detail {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }
    """

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
