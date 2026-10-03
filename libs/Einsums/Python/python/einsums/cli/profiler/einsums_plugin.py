# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""What Einsums adds to Waggle's viewer: TaskPool workers and compute graphs.

The data comes from handlers Einsums registers with its program's profiler,
``get_taskpool_metrics`` (``TaskPool.cpp``) and ``get_compute_graphs`` (``Graph/Report.cpp``),
and from the ``einsums.compute_graphs`` extension of a saved session.
"""

from __future__ import annotations

from typing import Any

from waggle.plugin import PluginAction, PluginPanel, Requirement, ViewerPlugin

from .graphs import parse_graphs, parse_taskpool, taskpool_text

#: The session extension that holds compute graphs.
GRAPHS_EXTENSION = "einsums.compute_graphs"


def _taskpool_widget(widget_id: str) -> Any:
    from waggle.widgets.panels import TextPanel

    panel = TextPanel("", id=widget_id, classes="panel")
    # As tall as the workers need, up to a screenful's share.
    panel.styles.height = "auto"
    panel.styles.max_height = 20
    return panel


async def _refresh_taskpool(app: Any, widget: Any, session: Any, client: Any) -> None:
    if client is None:
        widget.show(taskpool_text(None, "metrics come from a live connection; this session has none"))
        return
    reply = await client.request("get_taskpool_metrics", timeout=2.0)
    widget.show(taskpool_text(parse_taskpool(reply), reply.get("error", "") if "unknown method" not in str(reply) else ""))


async def _show_compute_graphs(app: Any, session: Any, client: Any) -> None:
    from .graph_screen import GraphScreen

    payload: object = session.extensions.get(GRAPHS_EXTENSION, [])
    if client is not None:
        reply = await client.request("get_compute_graphs")
        if "error" not in reply:
            payload = reply
            session.extensions[GRAPHS_EXTENSION] = reply.get("graphs", [])  # kept, so Save includes them
    graphs = parse_graphs(payload)
    if not graphs:
        app.notify("No compute graphs: the program registered none, or the session file has none", severity="warning")
        return
    app.push_screen(GraphScreen(session.label, graphs))


def einsums_viewer_plugin() -> ViewerPlugin:
    """The plugin, as ``einsums profiler`` passes it and the ``waggle.viewer`` entry point names it."""
    return ViewerPlugin(
        name="einsums",
        title="Einsums",
        panels=[
            PluginPanel(
                name="taskpool",
                key="W",
                description="TaskPool workers (live sessions)",
                requirement=Requirement(handler="get_taskpool_metrics", what="TaskPool metrics"),
                make_widget=_taskpool_widget,
                refresh=_refresh_taskpool,
                interval=1.0,
            )
        ],
        actions=[
            PluginAction(
                name="compute_graphs",
                key="K",
                description="Compute graphs: nodes, tensors, timings",
                requirement=Requirement(handler="get_compute_graphs", extension=GRAPHS_EXTENSION, what="compute graphs"),
                run=_show_compute_graphs,
            )
        ],
    )
