# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""The profiler TUI: live sessions from profiler servers, saved sessions, and replays."""

from __future__ import annotations

import asyncio
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.widgets import DataTable, Footer, Header, Input, TabbedContent, TabPane

from .client import DEFAULT_HOST, DEFAULT_PORT, ProfileClient, Recorder, StreamState, parse_endpoint, read_recording
from .discovery import HAVE_ZEROCONF, ServerBrowser
from .disasm import disassemble
from .format import sparkline
from .model import ProfileMeta, ProfileNode
from .graphs import parse_graphs, parse_taskpool, taskpool_text
from .screens import CompareScreen, ConfirmDialog, GraphScreen, HelpScreen, PromptDialog, SessionsDialog
from .session import Session, export_snapshot, read_session_file, session_from_dict, write_session_file
from .widgets.graphs import FlameGraph, GanttChart, RooflinePlot, TimelinePlot
from .widgets.panels import (
    LogPanel,
    ResizeHandle,
    StatusBar,
    TextPanel,
    counter_analysis,
    highlight_asm,
    hotspots,
    node_details,
    source_context,
)
from .widgets.tree import SORT_KEYS, VIEW_MODES, ProfileTable, ThreadTreeView, ViewOptions, next_in

#: (section, [(key, action, description, footer label or None)]). The bindings and the help
#: screen are both generated from this, so the help cannot drift from the keys.
KEYMAP: list[tuple[str, list[tuple[str, str, str, str | None]]]] = [
    ("General", [
        ("q", "quit", "Quit", "Quit"),
        ("question_mark", "help", "Help", "Help"),
        ("slash", "filter", "Filter", "Filter"),
        ("escape", "clear_filter", "Clear the filter", None),
        ("n", "search(True)", "Next filter match", None),
        ("N", "search(False)", "Previous filter match", None),
        ("p", "toggle_pause", "Pause", "Pause"),
        ("r", "refresh", "Redraw", None),
    ]),
    ("Tree", [
        ("space", "toggle_row", "Expand or collapse the row (also enter, click)", None),
        ("e", "expand_all", "Expand all", None),
        ("w", "collapse_all", "Collapse all", None),
        ("s", "cycle_sort", "Sort: " + " → ".join(SORT_KEYS), "Sort"),
        ("a", "toggle_mode('flat')", "Flat view: one row per zone name", None),
        ("u", "toggle_mode('bottom-up')", "Bottom-up view: callers under each zone", None),
        ("h", "toggle_hot_path", "Mark the hot path", None),
        ("T", "toggle_heat", "Heat colors on or off", None),
        ("b", "toggle_bookmark", "Bookmark the zone", None),
        ("B", "next_bookmark", "Next bookmark", None),
        ("y", "copy_node", "Copy the zone's details", None),
        ("d", "toggle_baseline", "Capture a baseline to diff against (again to clear)", None),
    ]),
    ("Panels", [
        ("H", "toggle_panel('hotspots')", "Hotspots", None),
        ("f", "toggle_panel('flame')", "Flame graph (click to zoom, Esc to zoom out)", None),
        ("g", "toggle_panel('timeline')", "Timeline of CPU and memory", None),
        ("o", "toggle_panel('roofline')", "Roofline", None),
        ("G", "toggle_panel('gantt')", "Thread Gantt chart", None),
        ("M", "toggle_panel('counters')", "Hardware counters", None),
        ("A", "toggle_panel('disasm')", "Disassembly", None),
        ("V", "toggle_panel('source')", "Source", None),
        ("L", "toggle_panel('log')", "Log", None),
        ("W", "toggle_panel('taskpool')", "TaskPool workers (live sessions)", None),
        ("K", "compute_graphs", "Compute graphs: nodes, tensors, timings", None),
        ("l", "cycle_log_level", "Log level: INFO → WARN → ERROR → TRACE → DEBUG", None),
    ]),
    ("Sessions", [
        ("t", "connect", "Connect to a server", "Connect"),
        ("c", "retry", "Retry connections now", None),
        ("D", "toggle_listen", "Discover servers over mDNS", None),
        ("R", "toggle_record", "Record the stream to a .jsonl file", None),
        ("S", "save_session", "Save this session", None),
        ("ctrl+s", "save_all", "Save several sessions to one file", None),
        ("C", "compare", "Compare two sessions", None),
        ("x", "export", "Export this snapshot as JSON and CSV", None),
        ("plus", "replay_speed(2.0)", "Replay faster", None),
        ("minus", "replay_speed(0.5)", "Replay slower", None),
    ]),
]  # fmt: skip

#: Panel name -> widget id. Every one starts hidden.
PANELS = ("hotspots", "flame", "timeline", "roofline", "gantt", "counters", "disasm", "source", "log", "taskpool")
TASKPOOL_INTERVAL = 1.0  # seconds between metric requests while the panel is open

REFRESH_INTERVAL = 0.25  # seconds between redraws of sessions with new data


def _widget_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", text)


@dataclass
class SessionUI:
    session: Session
    options: ViewOptions
    tabs_id: str
    views: dict[str, ThreadTreeView]  # pane id -> view


class ProfilerApp(App):
    TITLE = "Einsums profiler"
    CSS_PATH = "profiler.tcss"
    BINDINGS = [
        Binding(key, action, footer or desc, show=footer is not None, priority=(key == "q"))
        for _, keys in KEYMAP
        for key, action, desc, footer in keys
        if key != "space"  # the table binds space and enter itself
    ]

    def __init__(
        self,
        endpoints: list[tuple[str, int]] | None = None,
        *,
        load: list[str] | None = None,
        replay: str | None = None,
        replay_speed: float = 1.0,
        record: str | None = None,
        mdns: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._load = load or []
        self._replay = replay
        self._replay_speed = max(0.1, replay_speed)
        self._record_path = record
        self._recorder: Recorder | None = None
        explicit = list(endpoints or [])
        if not explicit and not self._load and not self._replay:
            explicit = [(DEFAULT_HOST, DEFAULT_PORT)]
        self._endpoints = explicit
        self._listen = mdns and HAVE_ZEROCONF and not self._load and not self._replay
        self._browser = ServerBrowser(self._on_server_found)
        self._clients: list[ProfileClient] = []
        self._ui: dict[str, SessionUI] = {}
        self._active: str | None = None
        self._counter = 0
        self._dirty: set[str] = set()
        self._paused = False
        self._name_filter = ""
        self._event = "Starting"
        self._disasm_target: tuple[str, str] | None = None
        #: The live connection each session came from, for requests to its server.
        self._session_clients: dict[str, ProfileClient] = {}

    # -- layout ---------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield TextPanel(id="hotspots", classes="panel")
        yield TabbedContent(id="sessions")
        yield FlameGraph(id="flame", classes="panel")
        yield TimelinePlot(id="timeline", classes="panel")
        yield RooflinePlot(id="roofline", classes="panel")
        yield GanttChart(id="gantt", classes="panel")
        yield TextPanel("Select a row", id="counters", classes="panel")
        yield TextPanel("Select a row", id="disasm", classes="panel")
        yield TextPanel("Select a row", id="source", classes="panel")
        yield ResizeHandle("detail", id="detail-resize")
        yield TextPanel("Select a row to see its details", id="detail")
        yield ResizeHandle("log", id="log-resize")
        yield LogPanel(id="log", classes="panel")
        yield TextPanel("", id="taskpool", classes="panel")
        yield Input(placeholder="Filter zones by name (regex)", id="filter")
        yield StatusBar(id="status")
        yield Footer()

    async def on_mount(self) -> None:
        if self._record_path:
            self._start_recording(self._record_path)
        for path in self._load:
            await self.load_file(path)
        if self._replay:
            self.run_worker(self._replay_loop(self._replay), name="replay")
        else:
            for host, port in self._endpoints:
                self.add_client(host, port)
            if self._listen:
                self._browser.start()
        self.set_interval(REFRESH_INTERVAL, self._flush)
        self.set_interval(TASKPOOL_INTERVAL, self._poll_taskpool)
        self._update_status()

    # -- sessions -------------------------------------------------------------

    async def new_session(self, meta: ProfileMeta | None = None, *, source: str = "", data: dict[str, Any] | None = None) -> Session:
        self._counter += 1
        sid = f"s{self._counter}"
        if data is not None:
            session = session_from_dict(data, sid, source)
        else:
            session = Session(session_id=sid, label=meta.label if meta else f"Session {self._counter}", meta=meta, source=source)
        ui = SessionUI(session, ViewOptions(bookmarks=session.bookmarks), f"threads-{sid}", {})
        self._ui[sid] = ui
        sessions = self.query_one("#sessions", TabbedContent)
        await sessions.add_pane(TabPane(session.label, TabbedContent(id=ui.tabs_id), id=f"session-{sid}"))
        sessions.active = f"session-{sid}"
        self._active = sid
        self._dirty.add(sid)
        return session

    async def load_file(self, path: str) -> list[Session]:
        try:
            records = read_session_file(path)
        except (OSError, ValueError) as exc:
            self.notify(f"Cannot load {path}: {exc}", severity="error")
            return []
        loaded = [await self.new_session(source=path, data=record) for record in records]
        empty = sum(1 for s in loaded if s.snapshot is None)
        self.notify(f"Loaded {len(loaded)} session(s) from {path}" + (f" ({empty} without data)" if empty else ""))
        await self._flush()
        return loaded

    @property
    def active_ui(self) -> SessionUI | None:
        return self._ui.get(self._active or "")

    @property
    def active_session(self) -> Session | None:
        ui = self.active_ui
        return ui.session if ui else None

    @property
    def active_view(self) -> ThreadTreeView | None:
        ui = self.active_ui
        if ui is None:
            return None
        try:
            pane = self.query_one(f"#{ui.tabs_id}", TabbedContent).active
        except NoMatches:
            return None
        return ui.views.get(pane)

    @property
    def selected_node(self) -> ProfileNode | None:
        view = self.active_view
        return view.selected if view else None

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tabbed_content.id == "sessions":
            pane = event.pane.id or ""
            self._active = pane.removeprefix("session-")
            self.query_one("#log", LogPanel).reset()
        self._refresh_panels()
        self._show_selected()

    def absorb(self, session: Session, state: StreamState, kind: str) -> None:
        """Copy what a message changed from the stream into its session."""
        if kind == "snapshot" and state.snapshot is not None:
            session.record_snapshot(state.snapshot)
        elif kind == "timeline":
            session.timeline = state.timeline
        elif kind in ("log", "output"):
            session.log_entries, session.log_total = state.log_entries, state.log_total
        else:
            return
        self._dirty.add(session.session_id)

    # -- redrawing ----------------------------------------------------------------

    async def _flush(self) -> None:
        """Redraw the sessions that received data since the last tick."""
        if self._paused or not self._dirty:
            return
        dirty, self._dirty = self._dirty, set()
        for sid in dirty:
            if sid in self._ui:
                await self._draw_session(self._ui[sid])
        if self._active in dirty:
            self._refresh_panels()
            self._show_selected()
        self._update_status()

    async def _draw_session(self, ui: SessionUI) -> None:
        snap = ui.session.snapshot
        if snap is None:
            return
        try:
            tabs = self.query_one(f"#{ui.tabs_id}", TabbedContent)
        except NoMatches:
            return
        ui.options.name_filter = self._name_filter
        for tid, thread in snap.threads.items():
            pane_id = f"thread-{ui.session.session_id}-{_widget_id(tid)}"
            view = ui.views.get(pane_id)
            if view is None:
                view = ThreadTreeView(tid, ui.options)
                ui.views[pane_id] = view
                view.roots = thread.children
                await tabs.add_pane(TabPane(thread.label, view, id=pane_id))
            else:
                tab = tabs.get_tab(pane_id)
                if str(tab.label) != thread.label:
                    tab.label = thread.label
                view.set_roots(thread.children)

    def _refresh_panels(self) -> None:
        session, view = self.active_session, self.active_view
        if session is None:
            return
        roots = view.roots if view else []
        if self._visible("hotspots") and session.snapshot:
            self.query_one("#hotspots", TextPanel).show(hotspots(session.snapshot.all_roots()))
        if self._visible("flame"):
            self.query_one(FlameGraph).set_roots(roots)
        if session.snapshot:
            self.query_one(TimelinePlot).record(session.snapshot)
        if self._visible("roofline"):
            self.query_one(RooflinePlot).set_roots(roots)
        if self._visible("gantt"):
            self.query_one(GanttChart).set_events(session.timeline)
        if self._visible("log"):
            self.query_one(LogPanel).show(session.log_entries, session.log_total)

    def _show_selected(self, node: ProfileNode | None = None) -> None:
        """Fill the detail panel, and the source, counter and disassembly panels when shown."""
        view, session = self.active_view, self.active_session
        node = node or self.selected_node
        detail = self.query_one("#detail", TextPanel)
        if node is None or session is None:
            detail.show("Select a row to see its details")
            return
        history = sparkline(list(session.history.get(node.name, ())))
        share = view.share_of_total(node) if view else 0.0
        detail.show(
            node_details(node, share, baseline=session.baseline, history=history, bookmarked=node.name in session.bookmarks)
        )
        if self._visible("source"):
            self.query_one("#source", TextPanel).show(source_context(node))
        if self._visible("counters"):
            self.query_one("#counters", TextPanel).show(counter_analysis(node))
        if self._visible("disasm"):
            target = (node.function or node.name, session.meta.executable_path if session.meta else "")
            # Redraws reach here four times a second; only a new function starts a lookup.
            if target != self._disasm_target:
                self._disasm_target = target
                self._disassemble(*target)

    @work(exclusive=True, group="disasm")
    async def _disassemble(self, function: str, executable: str) -> None:
        # Exclusive: moving to another row cancels this, which kills its nm/objdump.
        panel = self.query_one("#disasm", TextPanel)
        panel.show(f"Looking for {function}…")
        panel.show(highlight_asm(await disassemble(function, executable)))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if isinstance(event.data_table, ProfileTable):
            self._show_selected()

    def on_profile_table_toggle_request(self, event: ProfileTable.ToggleRequest) -> None:
        view = self.active_view
        if view is not None:
            view.toggle(event.row)

    def on_flame_graph_span_clicked(self, event: FlameGraph.SpanClicked) -> None:
        self._show_selected(event.node)

    def _visible(self, panel: str) -> bool:
        return self.query_one(f"#{panel}").has_class("visible")

    def _say(self, text: str) -> None:
        self._event = text
        self._update_status()

    def _update_status(self) -> None:
        session = self.active_session
        parts = ["PAUSED" if self._paused else self._event]
        if session and session.meta and session.meta.pid:
            parts.append(f"PID {session.meta.pid}")
        if session and session.snapshot and session.snapshot.dropped:
            parts.append(f"dropped={session.snapshot.dropped}")
        if session and session.baseline is not None:
            parts.append("[DIFF]")
        if self._recorder:
            parts.append("[REC]")
        if self._listen:
            parts.append("[mDNS]")
        if self._replay:
            parts.append(f"speed={self._replay_speed:g}x")
        if len(self._ui) > 1:
            parts.append(f"{len(self._ui)} sessions")
        if len(self._clients) > 1:
            parts.append(f"{len(self._clients)} servers")
        self.query_one(StatusBar).text = "  ".join(parts)

    # -- live connections ---------------------------------------------------------

    def add_client(self, host: str, port: int, *, discovered: bool = False) -> ProfileClient | None:
        if any(c.host == host and c.port == port for c in self._clients):
            return None
        client = ProfileClient(host, port, discovered=discovered)
        self._clients.append(client)
        self.run_worker(self._run_client(client), name=f"client-{client.endpoint}", group="clients")
        return client

    async def _run_client(self, client: ProfileClient) -> None:
        delay, failures = 1.0, 0
        while True:
            self._say(f"Connecting to {client.endpoint}")
            if await client.connect():
                delay, failures = 1.0, 0
                self._say(f"Connected to {client.endpoint}")
                session: Session | None = None
                async for msg in client.messages():
                    if self._recorder:
                        self._recorder.write(msg)
                    kind = client.state.apply(msg)
                    if kind == "meta":
                        session = await self.new_session(client.state.meta, source=client.endpoint)
                        self._session_clients[session.session_id] = client
                    elif session is not None:
                        self.absorb(session, client.state, kind)
                self._say(f"{client.endpoint}: disconnected")
            else:
                failures += 1
            # A server found over mDNS is gone for good once it stops answering.
            if client.discovered and (failures >= 3 or not self._listen):
                self._clients.remove(client)
                self._update_status()
                return
            await self._wait_to_retry(client, delay)
            delay = min(delay * 2, 30.0)

    async def _wait_to_retry(self, client: ProfileClient, delay: float) -> None:
        remaining = int(delay)
        while remaining > 0:
            self._say(f"{client.endpoint}: no server; retrying in {remaining}s (c retries now)")
            try:
                await asyncio.wait_for(client.retry_now.wait(), 1.0)
                break
            except asyncio.TimeoutError:
                remaining -= 1
        client.retry_now.clear()

    def _on_server_found(self, host: str, port: int, exe: str) -> None:
        def connect() -> None:
            if self.add_client(host, port, discovered=True):
                self.notify(f"Found {exe} at {host}:{port}")

        self.call_from_thread(connect)

    def _live_client(self) -> ProfileClient | None:
        session = self.active_session
        client = self._session_clients.get(session.session_id) if session else None
        return client if client is not None and client.connected else None

    async def _poll_taskpool(self) -> None:
        if not self._visible("taskpool"):
            return
        panel = self.query_one("#taskpool", TextPanel)
        client = self._live_client()
        if client is None:
            panel.show(taskpool_text(None, "metrics come from a live connection; this session has none"))
            return
        reply = await client.request("get_taskpool_metrics", timeout=2.0)
        panel.show(taskpool_text(parse_taskpool(reply), reply.get("error", "") if "unknown method" not in str(reply) else ""))

    async def action_compute_graphs(self) -> None:
        session = self.active_session
        if session is None:
            self.notify("No session", severity="warning")
            return
        payload: object = session.compute_graphs
        if client := self._live_client():
            reply = await client.request("get_compute_graphs")
            if "error" not in reply:
                payload = reply
                session.compute_graphs = reply.get("graphs", [])  # kept, so Save includes them
        graphs = parse_graphs(payload)
        if not graphs:
            self.notify("No compute graphs: the program registered none, or the session file has none", severity="warning")
            return
        self.push_screen(GraphScreen(session.label, graphs))

    # -- replay -------------------------------------------------------------------

    async def _replay_loop(self, path: str) -> None:
        try:
            entries = list(read_recording(path))
        except (OSError, ValueError) as exc:
            self._say(f"Cannot replay {path}: {exc}")
            return
        state, session = StreamState(), None
        for i, (ts, msg) in enumerate(entries):
            while self._paused:
                await asyncio.sleep(0.1)
            if i > 0 and (gap := (ts - entries[i - 1][0]) / self._replay_speed) > 0:
                await asyncio.sleep(gap)
            kind = state.apply(msg)
            if kind == "meta":
                session = await self.new_session(state.meta, source=path)
            elif session is not None:
                self.absorb(session, state, kind)
            self._say(f"Replay {i + 1}/{len(entries)}  t={ts - entries[0][0]:.1f}s")
        self._say(f"Replay finished ({len(entries)} messages)")

    # -- actions: general -------------------------------------------------------

    async def action_quit(self) -> None:
        self._browser.stop()
        if self._recorder:
            self._recorder.close()
        self.exit()

    def action_help(self) -> None:
        keymap = [(section, [(k, d) for k, _, d, _ in keys]) for section, keys in KEYMAP]
        self.push_screen(HelpScreen(keymap))

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.value = self._name_filter
        box.add_class("visible")
        box.focus()

    def action_clear_filter(self) -> None:
        self.query_one("#filter", Input).remove_class("visible")
        if self._name_filter:
            self._set_filter("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter":
            event.input.remove_class("visible")
            self._set_filter(event.value.strip())

    def _set_filter(self, pattern: str) -> None:
        self._name_filter = pattern
        self._dirty |= self._ui.keys()
        self.call_later(self._flush)
        if self.active_view:
            self.active_view.table.focus()

    def action_search(self, forward: bool) -> None:
        view = self.active_view
        if not self._name_filter:
            self.notify("No filter set; press / first", severity="warning")
        elif view and not view.jump(view.matching_rows(self._name_filter), forward):
            self.notify("No matches", severity="warning")

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        if not self._paused:
            self._dirty |= self._ui.keys()
        self._update_status()

    def action_refresh(self) -> None:
        self._dirty |= self._ui.keys()

    # -- actions: tree ----------------------------------------------------------

    def _redraw_active(self) -> None:
        ui = self.active_ui
        if ui:
            for view in ui.views.values():
                view.refresh_rows()
        self._show_selected()

    def action_expand_all(self) -> None:
        if self.active_view:
            self.active_view.expand_all()

    def action_collapse_all(self) -> None:
        if self.active_view:
            self.active_view.collapse_all()

    def action_cycle_sort(self) -> None:
        if ui := self.active_ui:
            ui.options.sort_key = next_in(SORT_KEYS, ui.options.sort_key)
            self._redraw_active()
            self.notify(f"Sort: {ui.options.sort_key}")

    def action_toggle_mode(self, mode: str) -> None:
        if ui := self.active_ui:
            ui.options.mode = "tree" if ui.options.mode == mode else mode
            assert ui.options.mode in VIEW_MODES
            self._redraw_active()
            self.notify(f"View: {ui.options.mode}")

    def action_toggle_hot_path(self) -> None:
        if ui := self.active_ui:
            ui.options.hot_path = not ui.options.hot_path
            self._redraw_active()
            self.notify(f"Hot path: {'on' if ui.options.hot_path else 'off'}")

    def action_toggle_heat(self) -> None:
        if ui := self.active_ui:
            ui.options.heat = not ui.options.heat
            self._redraw_active()

    def action_toggle_bookmark(self) -> None:
        session, node = self.active_session, self.selected_node
        if session and node:
            # In place: every view's options hold this same set.
            session.bookmarks.symmetric_difference_update({node.name})
            self.notify(f"{'Bookmarked' if node.name in session.bookmarks else 'Removed bookmark'}: {node.name}")
            self._redraw_active()

    def action_next_bookmark(self) -> None:
        view = self.active_view
        if view and not view.jump(view.bookmarked_rows()):
            self.notify("No bookmarked zones in this thread (b bookmarks one)", severity="warning")

    def action_copy_node(self) -> None:
        session, view, node = self.active_session, self.active_view, self.selected_node
        if node is None or session is None:
            self.notify("No zone selected", severity="warning")
            return
        text = node_details(node, view.share_of_total(node) if view else 0.0)
        self.copy_to_clipboard(Text.from_markup(text).plain)
        self.notify("Copied")

    def action_toggle_baseline(self) -> None:
        session = self.active_session
        if session is None:
            return
        if session.baseline is not None:
            session.baseline = None
            self.notify("Baseline cleared")
        elif session.snapshot is not None:
            session.baseline = copy.deepcopy(session.snapshot)
            self.notify("Baseline captured; the detail panel now shows changes since it")
        else:
            self.notify("No snapshot yet", severity="warning")
        self._update_status()
        self._show_selected()

    # -- actions: panels --------------------------------------------------------

    def action_toggle_panel(self, panel: str) -> None:
        assert panel in PANELS
        widget = self.query_one(f"#{panel}")
        widget.toggle_class("visible")
        if panel == "log":
            self.query_one("#log-resize").set_class(widget.has_class("visible"), "visible")
            self.query_one(LogPanel).reset()
        if panel == "disasm":
            self._disasm_target = None
        if panel == "taskpool" and widget.has_class("visible"):
            self.call_later(self._poll_taskpool)
        if panel == "flame" and not widget.has_class("visible"):
            self.query_one(FlameGraph).zoom_reset()
        if widget.has_class("visible"):
            self._refresh_panels()
            self._show_selected()
            if panel == "flame":
                widget.focus()
            if panel == "roofline" and not self.query_one(RooflinePlot).points:
                self.notify("No zone has flops and bytes_read/bytes_written annotations", severity="warning")

    def action_cycle_log_level(self) -> None:
        level = self.query_one(LogPanel).cycle_level()
        self._refresh_panels()
        self.notify(f"Log level: {level} and above")

    # -- actions: sessions -------------------------------------------------------

    def action_connect(self) -> None:
        def connect(value: str | None) -> None:
            if not value:
                return
            try:
                host, port = parse_endpoint(value)
            except ValueError:
                self.notify(f"Expected host:port or a port, not {value!r}", severity="error")
                return
            if self.add_client(host, port) is None:
                self.notify(f"Already connected to {host}:{port}", severity="warning")

        self.push_screen(PromptDialog("Connect to a profiler server", placeholder="host:port or port", ok="Connect"), connect)

    def action_retry(self) -> None:
        for client in self._clients:
            client.retry_now.set()

    def action_toggle_listen(self) -> None:
        if not HAVE_ZEROCONF:
            self.notify("mDNS discovery needs zeroconf (conda install -c conda-forge zeroconf)", severity="warning")
            return
        self._listen = not self._listen
        self._browser.start() if self._listen else self._browser.stop()
        self.notify(f"mDNS discovery {'on' if self._listen else 'off'}")
        self._update_status()

    def _start_recording(self, path: str) -> None:
        try:
            self._recorder = Recorder(path)
        except OSError as exc:
            self.notify(f"Cannot record to {path}: {exc}", severity="error")
            return
        self.notify(f"Recording to {path}")

    def action_toggle_record(self) -> None:
        if self._replay:
            self.notify("Cannot record a replay", severity="warning")
        elif self._recorder:
            self._recorder.close()
            self.notify(f"Recording saved to {self._recorder.path}")
            self._recorder = None
        else:
            self._start_recording(self._record_path or "einsums_profile_recording.jsonl")
        self._update_status()

    def action_replay_speed(self, factor: float) -> None:
        if self._replay:
            self._replay_speed = min(64.0, max(0.125, self._replay_speed * factor))
            self._update_status()

    def _save(self, sessions: list[Session], path: str) -> None:
        def write(ok: bool = True) -> None:
            if ok:
                write_session_file(path, sessions)
                self.notify(f"Saved {len(sessions)} session(s) to {path}")

        if Path(path).exists():
            self.push_screen(ConfirmDialog(f"{path} exists. Overwrite?"), write)
        else:
            write()

    def action_save_session(self) -> None:
        session = self.active_session
        if session is None or session.snapshot is None:
            self.notify("Nothing to save yet", severity="warning")
            return
        name = "einsums_session_" + _widget_id(session.label) + ".json"
        self.push_screen(
            PromptDialog("Save this session", value=name, ok="Save"),
            lambda path: path and self._save([session], path),
        )

    def _sessions_with_data(self) -> list[Session]:
        return [ui.session for ui in self._ui.values() if ui.session.snapshot is not None]

    def action_save_all(self) -> None:
        sessions = self._sessions_with_data()
        if not sessions:
            self.notify("Nothing to save yet", severity="warning")
            return

        def save(result: tuple[list[str], str] | None) -> None:
            if result:
                ids, path = result
                self._save([s for s in sessions if s.session_id in ids], path)

        self.push_screen(SessionsDialog("Sessions to save", sessions, preselect=True, filename="einsums_sessions.json"), save)

    def action_compare(self) -> None:
        sessions = self._sessions_with_data()
        if len(sessions) < 2:
            self.notify("Comparing needs two sessions with data (--load several files)", severity="warning")
        elif len(sessions) == 2:
            self.push_screen(CompareScreen(*sessions))
        else:
            by_id = {s.session_id: s for s in sessions}
            self.push_screen(
                SessionsDialog("Pick two sessions to compare", sessions, exactly=2),
                lambda ids: ids and self.push_screen(CompareScreen(by_id[ids[0]], by_id[ids[1]])),
            )

    def action_export(self) -> None:
        session = self.active_session
        if session is None or session.snapshot is None:
            self.notify("Nothing to export yet", severity="warning")
            return
        export_snapshot(session.snapshot, "einsums_profile_snapshot.json", "einsums_profile_snapshot.csv")
        self.notify("Exported einsums_profile_snapshot.json and .csv")
