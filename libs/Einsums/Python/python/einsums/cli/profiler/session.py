# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Sessions (one program run each) and the files they are saved to.

Two writers produce session files, and both shapes load:

* the viewer's own "Save session", which nests the tree under ``"snapshot"``;
* ``--einsums:profile:save`` (``Server::export_session``), which puts ``seq``,
  ``dropped`` and ``threads`` at the top level beside ``meta``.

Either may be wrapped as ``{"sessions": [...]}``; the server appends to an existing
file that way, so repeated runs accumulate for comparison.

Both now mark a record ``"format": "waggle-session"`` with a ``"version"``, and keep what a
library adds (Einsums' compute graphs) under ``"extensions"``, keyed by the library's namespace.
Records from before carry Einsums' graphs at the top level as ``compute_graphs``, which loads
as the ``einsums.compute_graphs`` extension.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analysis import exclusive_by_name, is_anomaly, walk
from .model import (
    LogEntry,
    ProfileMeta,
    ProfileSnapshot,
    TimelineEvent,
    meta_to_dict,
    parse_meta,
    parse_snapshot,
    snapshot_to_dict,
)

HISTORY_LENGTH = 20

#: What session records this viewer writes say they are.
SESSION_FORMAT = "waggle-session"
SESSION_FORMAT_VERSION = 1


@dataclass
class Session:
    session_id: str
    label: str
    meta: ProfileMeta | None = None
    snapshot: ProfileSnapshot | None = None
    baseline: ProfileSnapshot | None = None
    timeline: list[TimelineEvent] = field(default_factory=list)
    #: Exclusive time of each zone name over recent snapshots, for the detail panel's sparkline.
    history: dict[str, deque[float]] = field(default_factory=dict)
    bookmarks: set[str] = field(default_factory=set)
    log_entries: deque[LogEntry] = field(default_factory=deque)
    log_total: int = 0
    #: Where the session came from: an endpoint, a file, or a recording.
    source: str = ""
    #: What libraries added to the session, by namespaced key (``einsums.compute_graphs``).
    extensions: dict[str, Any] = field(default_factory=dict)

    def record_snapshot(self, snap: ProfileSnapshot) -> None:
        self.snapshot = snap
        for name, ms in exclusive_by_name(snap).items():
            self.history.setdefault(name, deque(maxlen=HISTORY_LENGTH)).append(ms)


def session_to_dict(session: Session) -> dict[str, Any]:
    data: dict[str, Any] = {
        "format": SESSION_FORMAT,
        "version": SESSION_FORMAT_VERSION,
        "session_id": session.session_id,
        "label": session.label,
    }
    if session.meta:
        data["meta"] = meta_to_dict(session.meta)
    if session.bookmarks:
        data["bookmarks"] = sorted(session.bookmarks)
    if session.snapshot:
        data["snapshot"] = snapshot_to_dict(session.snapshot)
    if session.history:
        data["node_history"] = {name: list(values) for name, values in session.history.items()}
    if session.extensions:
        data["extensions"] = session.extensions
    return data


def session_from_dict(data: dict[str, Any], session_id: str, source: str = "") -> Session:
    meta = parse_meta(data["meta"]) if isinstance(data.get("meta"), dict) else None
    label = data.get("label") or ""
    # The server labels a session with the bare executable name, which every run shares.
    if meta and label in ("", meta.executable):
        label = meta.label
    session = Session(session_id=session_id, label=label or session_id, meta=meta)
    session.source = source
    session.bookmarks = set(data.get("bookmarks", []))
    if isinstance(data.get("snapshot"), dict):
        session.snapshot = parse_snapshot(data["snapshot"])
    elif "threads" in data:
        session.snapshot = parse_snapshot(data)
    for name, values in (data.get("node_history") or {}).items():
        session.history[name] = deque(values, maxlen=HISTORY_LENGTH)
    if isinstance(data.get("extensions"), dict):
        session.extensions = dict(data["extensions"])
    if "compute_graphs" in data and "einsums.compute_graphs" not in session.extensions:
        session.extensions["einsums.compute_graphs"] = list(data["compute_graphs"] or [])
    return session


def read_session_file(path: str | Path) -> list[dict[str, Any]]:
    """The session records in *path*, whichever writer made it."""
    with Path(path).open() as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("sessions"), list):
        return data["sessions"]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"{path}: not a profiler session file")


def write_session_file(path: str | Path, sessions: list[Session]) -> None:
    payload: dict[str, Any] = (
        session_to_dict(sessions[0])
        if len(sessions) == 1
        else {"format": SESSION_FORMAT, "version": SESSION_FORMAT_VERSION, "sessions": [session_to_dict(s) for s in sessions]}
    )
    with Path(path).open("w") as f:
        json.dump(payload, f, indent=2)


def export_snapshot(snap: ProfileSnapshot, json_path: str | Path, csv_path: str | Path) -> None:
    """The snapshot as JSON, and every node as a CSV row (thread, depth, timings, memory)."""
    with Path(json_path).open("w") as f:
        json.dump(snapshot_to_dict(snap), f, indent=2)
    with Path(csv_path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["thread", "depth", "name", "call_count", "exclusive_ms", "inclusive_ms", "mean_ms", "stddev_ms",
             "anomaly", "mem_alloc_bytes", "mem_peak_bytes"]
        )  # fmt: skip
        for thread in snap.threads.values():
            for node, depth in walk(thread.children):
                writer.writerow(
                    [thread.label, depth, node.name, node.call_count, f"{node.exclusive_ms:.3f}",
                     f"{node.inclusive_ms:.3f}", f"{node.mean_ms:.3f}", f"{node.stddev_ms:.3f}",
                     "!" if is_anomaly(node) else "", node.mem_alloc_bytes, node.mem_peak_bytes]
                )  # fmt: skip
