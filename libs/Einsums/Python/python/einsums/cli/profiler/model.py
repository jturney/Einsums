# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""The profiler's data, as the server streams it and as session files store it.

The server (``libs/Einsums/Profile/src/Server.cpp``) sends JSON Lines over TCP. Each
line is one message with a ``type``: ``meta`` once per connection, then ``snapshot``
(the aggregated call tree of every thread), ``timeline`` (recent zone spans),
``log``, ``output``, ``benchmark_result``, and ``response`` to a ``request``.

Nothing in this module imports Textual, so the bench commands and the tests use it
without the UI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileNode:
    """One zone in the call tree, aggregated over every entry at this call path."""

    name: str = ""
    call_count: int = 0
    exclusive_ms: float = 0.0
    inclusive_ms: float = 0.0
    exclusive_min_ms: float = 0.0
    exclusive_max_ms: float = 0.0
    stddev_ms: float = 0.0
    file: str = ""
    line: int = 0
    function: str = ""
    #: A plain value, or ``{"avg", "min", "max", "last"}`` for a numeric annotation.
    annotations: dict[str, Any] = field(default_factory=dict)
    #: Hardware counter name -> ``{"total", ...}``.
    counters: dict[str, Any] = field(default_factory=dict)
    mem_alloc_count: int = 0
    mem_free_count: int = 0
    mem_alloc_bytes: int = 0
    mem_free_bytes: int = 0
    mem_current_bytes: int = 0
    mem_peak_bytes: int = 0
    #: Per-call duration histogram, bucket label -> count.
    histogram: dict[str, int] = field(default_factory=dict)
    children: list[ProfileNode] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return self.exclusive_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def location(self) -> str:
        if not self.file:
            return ""
        return f"{self.file}:{self.line}" if self.line > 0 else self.file


@dataclass
class ThreadData:
    thread_id: str = ""
    thread_name: str = ""
    children: list[ProfileNode] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.thread_name or f"Thread {self.thread_id}"


@dataclass
class ProfileSnapshot:
    seq: int = 0
    dropped: int = 0
    threads: dict[str, ThreadData] = field(default_factory=dict)

    def all_roots(self) -> list[ProfileNode]:
        """Every thread's top-level zones, in one list."""
        return [node for thread in self.threads.values() for node in thread.children]


@dataclass
class ProfileMeta:
    pid: int = 0
    hostname: str = ""
    executable: str = ""
    executable_path: str = ""
    start_time: str = ""
    git_commit: str = ""
    git_branch: str = ""
    build_type: str = ""
    counters: list[str] = field(default_factory=list)
    #: Request handlers the program registered: what a viewer can ask it. None when the server
    #: predates advertising them, which says nothing about what it answers.
    handlers: list[str] | None = None
    #: The libraries using the profiler, each a dict of name, version and build.
    clients: list[dict[str, Any]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.executable or 'unknown'} ({self.start_time or '?'})"


@dataclass
class TimelineEvent:
    """One zone span, for the Gantt chart."""

    thread_id: str = ""
    name: str = ""
    start_ms: float = 0.0
    end_ms: float = 0.0


@dataclass
class LogEntry:
    level: int = 2  # spdlog's: 0 trace .. 5 critical; OUTPUT_LEVEL for println output
    timestamp: str = ""
    file: str = ""
    line: int = 0
    function: str = ""
    message: str = ""


#: The level given to ``output`` messages (``einsums::println``), above every log level.
OUTPUT_LEVEL = 7
LOG_LEVEL_NAMES = {0: "TRACE", 1: "DEBUG", 2: "INFO", 3: "WARN", 4: "ERROR", 5: "CRITICAL", OUTPUT_LEVEL: "OUTPUT"}


# ---------------------------------------------------------------------------
# JSON -> model
# ---------------------------------------------------------------------------


def parse_node(data: dict[str, Any]) -> ProfileNode:
    mem = data.get("memory") or {}
    histogram: dict[str, int] = {}
    raw_hist = data.get("histogram")
    if isinstance(raw_hist, dict):
        for bucket, count in raw_hist.items():
            try:
                histogram[bucket] = int(count)
            except (TypeError, ValueError):
                pass
    return ProfileNode(
        name=data.get("name", ""),
        call_count=data.get("call_count", 0),
        exclusive_ms=data.get("exclusive_ms", 0.0),
        inclusive_ms=data.get("inclusive_ms", 0.0),
        exclusive_min_ms=data.get("exclusive_min_ms", 0.0),
        exclusive_max_ms=data.get("exclusive_max_ms", 0.0),
        stddev_ms=data.get("stddev_ms", 0.0),
        file=data.get("file", ""),
        line=data.get("line", 0),
        function=data.get("function", ""),
        annotations=dict(data.get("annotations") or {}),
        counters=dict(data.get("counters") or {}),
        mem_alloc_count=mem.get("alloc_count", 0),
        mem_free_count=mem.get("free_count", 0),
        mem_alloc_bytes=mem.get("alloc_bytes", 0),
        mem_free_bytes=mem.get("free_bytes", 0),
        mem_current_bytes=mem.get("current_bytes", 0),
        mem_peak_bytes=mem.get("peak_bytes", 0),
        histogram=histogram,
        children=[parse_node(child) for child in data.get("children", [])],
    )


def parse_snapshot(data: dict[str, Any]) -> ProfileSnapshot:
    snap = ProfileSnapshot(seq=data.get("seq", 0), dropped=data.get("dropped", 0))
    for tid, tdata in (data.get("threads") or {}).items():
        snap.threads[str(tid)] = ThreadData(
            thread_id=str(tid),
            thread_name=tdata.get("name", ""),
            children=[parse_node(child) for child in tdata.get("children", [])],
        )
    return snap


def parse_meta(data: dict[str, Any]) -> ProfileMeta:
    return ProfileMeta(
        pid=data.get("pid", 0),
        hostname=data.get("hostname", ""),
        executable=data.get("executable", ""),
        executable_path=data.get("executable_path", ""),
        start_time=data.get("start_time", ""),
        git_commit=data.get("git_commit", ""),
        git_branch=data.get("git_branch", ""),
        build_type=data.get("build_type", ""),
        counters=list(data.get("counters") or []),
        handlers=list(data["handlers"]) if isinstance(data.get("handlers"), list) else None,
        clients=[c for c in data.get("clients") or [] if isinstance(c, dict)],
    )


def parse_timeline(data: dict[str, Any]) -> list[TimelineEvent]:
    return [
        TimelineEvent(
            thread_id=str(event.get("tid", "")),
            name=event.get("name", ""),
            start_ms=event.get("start_ms", 0.0),
            end_ms=event.get("end_ms", 0.0),
        )
        for event in data.get("events", [])
    ]


def parse_log(data: dict[str, Any]) -> LogEntry:
    if data.get("type") == "output":
        return LogEntry(level=OUTPUT_LEVEL, timestamp=data.get("timestamp", ""), message=data.get("message", ""))
    return LogEntry(
        level=data.get("level", 2),
        timestamp=data.get("timestamp", ""),
        file=data.get("file", ""),
        line=data.get("line", 0),
        function=data.get("function", ""),
        message=data.get("message", ""),
    )


# ---------------------------------------------------------------------------
# model -> JSON (the same shapes, so a saved file loads back)
# ---------------------------------------------------------------------------


def node_to_dict(node: ProfileNode) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": node.name,
        "call_count": node.call_count,
        "exclusive_ms": node.exclusive_ms,
        "inclusive_ms": node.inclusive_ms,
        "exclusive_min_ms": node.exclusive_min_ms,
        "exclusive_max_ms": node.exclusive_max_ms,
        "stddev_ms": node.stddev_ms,
        "file": node.file,
        "line": node.line,
        "function": node.function,
        "annotations": node.annotations,
        "counters": node.counters,
        "children": [node_to_dict(child) for child in node.children],
    }
    if node.mem_alloc_bytes or node.mem_free_bytes:
        data["memory"] = {
            "alloc_count": node.mem_alloc_count,
            "free_count": node.mem_free_count,
            "alloc_bytes": node.mem_alloc_bytes,
            "free_bytes": node.mem_free_bytes,
            "current_bytes": node.mem_current_bytes,
            "peak_bytes": node.mem_peak_bytes,
        }
    if node.histogram:
        data["histogram"] = node.histogram
    return data


def snapshot_to_dict(snap: ProfileSnapshot) -> dict[str, Any]:
    return {
        "seq": snap.seq,
        "dropped": snap.dropped,
        "threads": {
            tid: {"name": thread.thread_name, "children": [node_to_dict(n) for n in thread.children]}
            for tid, thread in snap.threads.items()
        },
    }


def meta_to_dict(meta: ProfileMeta) -> dict[str, Any]:
    return {
        "pid": meta.pid,
        "hostname": meta.hostname,
        "executable": meta.executable,
        "executable_path": meta.executable_path,
        "start_time": meta.start_time,
        "git_commit": meta.git_commit,
        "git_branch": meta.git_branch,
        "build_type": meta.build_type,
        "counters": meta.counters,
        "clients": meta.clients,
    } | ({"handlers": meta.handlers} if meta.handlers is not None else {})
