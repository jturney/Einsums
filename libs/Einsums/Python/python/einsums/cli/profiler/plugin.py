# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Viewer plugins: what a library adds to the viewer.

The viewer knows the profiler's own data. A library that registers request handlers with its
program's profiler (Einsums: ``get_taskpool_metrics``, ``get_compute_graphs``) or adds sections to
its session files ships a plugin that shows them: panels, refreshed while visible, and actions
bound to keys. A plugin names, for each, the handler it calls and the session extension it reads,
so the viewer offers it only where it can be filled:

* in a live session, when the program's meta message lists the handler, or when the server
  predates advertising handlers and so says nothing either way;
* in a loaded session, when the file holds the extension.

Plugins come from the ``waggle.viewer`` entry-point group, each entry a callable returning a
:class:`ViewerPlugin`, and from whoever starts the viewer, as ``einsums profiler`` passes
Einsums' own.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .client import ProfileClient
    from .session import Session

ENTRY_POINT_GROUP = "waggle.viewer"


@dataclass
class Requirement:
    """What a plugin feature needs to have anything to show."""

    #: The request handler it calls on a live program, if any.
    handler: str | None = None
    #: The session extension it reads from a loaded session, if any.
    extension: str | None = None
    #: What the data is, in words, for the message that says it is missing.
    what: str = "this data"

    def met_by(self, session: Session, live: bool) -> bool:
        if live and self.handler is not None:
            handlers = session.meta.handlers if session.meta else None
            return handlers is None or self.handler in handlers
        if self.extension is not None:
            return self.extension in session.extensions
        return self.handler is None

    def explain(self, live: bool) -> str:
        """Why there is nothing to show, as a sentence."""
        if live and self.handler is not None:
            return f"This program does not report {self.what}."
        if self.extension is not None:
            return f"This session has no {self.what}."
        return f"{self.what[:1].upper()}{self.what[1:]} come only from a live program."


@dataclass
class PluginPanel:
    """A panel the plugin adds, toggled by a key and redrawn while visible."""

    name: str
    key: str
    description: str
    requirement: Requirement
    #: Builds the panel's widget, which the viewer gives the id ``<plugin>-<name>``.
    make_widget: Callable[[str], Any]
    #: Redraws the panel: (app, widget, session, live client or None).
    refresh: Callable[[Any, Any, Session, ProfileClient | None], Awaitable[None]]
    #: Seconds between redraws while visible; None redraws only when shown.
    interval: float | None = None


@dataclass
class PluginAction:
    """An action the plugin binds to a key."""

    name: str
    key: str
    description: str
    requirement: Requirement
    #: Runs the action: (app, session, live client or None).
    run: Callable[[Any, Session, ProfileClient | None], Awaitable[None]]


@dataclass
class ViewerPlugin:
    #: A short identifier, used in widget ids and the help screen.
    name: str
    #: The help screen's heading for this plugin's keys.
    title: str
    panels: list[PluginPanel] = field(default_factory=list)
    actions: list[PluginAction] = field(default_factory=list)

    def panel(self, name: str) -> PluginPanel:
        return next(p for p in self.panels if p.name == name)

    def action(self, name: str) -> PluginAction:
        return next(a for a in self.actions if a.name == name)


def discover_plugins() -> list[ViewerPlugin]:
    """Every plugin installed packages register under the ``waggle.viewer`` entry points."""
    found = []
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        try:
            plugin = entry.load()()
        except Exception as exc:  # noqa: BLE001 - one broken plugin must not stop the viewer
            print(f"viewer plugin {entry.name} failed to load: {exc}")
            continue
        if isinstance(plugin, ViewerPlugin):
            found.append(plugin)
    return found


def merge_plugins(given: list[ViewerPlugin], discovered: list[ViewerPlugin]) -> list[ViewerPlugin]:
    """*given* first, then the discovered ones not already given, by name."""
    names = {p.name for p in given}
    return list(given) + [p for p in discovered if p.name not in names]
