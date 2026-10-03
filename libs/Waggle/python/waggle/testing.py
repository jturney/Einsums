# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Helpers for testing the viewer and the plugins libraries write for it.

A fake server that speaks the wire protocol, the messages a program's server sends, and the
shape of a saved session, so a plugin's tests need no profiled program.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

#: A program's first message: who it is.
META: dict[str, Any] = {
    "type": "meta",
    "pid": 42,
    "executable": "prog",
    "executable_path": "/nonexistent/prog",
    "start_time": "T0",
}


def wire_node(name: str, excl: float = 1.0, children: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """One zone as the server sends it, its inclusive time its own plus its children's."""
    kids = list(children)
    return {
        "name": name,
        "call_count": 1,
        "exclusive_ms": excl,
        "inclusive_ms": excl + sum(k["inclusive_ms"] for k in kids),
        "file": "f.cpp",
        "line": 3,
        "function": name,
        "children": kids,
    }


def snapshot_msg(seq: int = 1) -> dict[str, Any]:
    """A snapshot message: thread 7, "main", running outer over inner and other."""
    tree = wire_node("outer", 2.0, [wire_node("inner", 5.0), wire_node("other", 1.0)])
    return {"type": "snapshot", "seq": seq, "dropped": 0, "threads": {"7": {"name": "main", "children": [tree]}}}


def server_export(label: str = "prog") -> dict[str, Any]:
    """The shape Server::export_session writes: the snapshot fields beside meta, at top level."""
    snap = snapshot_msg()
    return {"label": label, "type": "snapshot", "meta": dict(META), "seq": 3, "dropped": 0, "threads": snap["threads"]}


async def fake_server(chunks: Sequence[bytes], on_request: Callable[[str], Any] | None = None) -> tuple[asyncio.Server, int]:
    """A server on a free local port that writes *chunks* raw, then answers each request with
    ``on_request(method)``. Returns the server and its port."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        for chunk in chunks:
            writer.write(chunk)
            await writer.drain()
            await asyncio.sleep(0.01)
        while line := await reader.readline():
            req = json.loads(line)
            data = on_request(req["method"]) if on_request else {}
            writer.write((json.dumps({"type": "response", "id": req["id"], "data": data}) + "\n").encode())
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def wait_for(pilot: Any, condition: Callable[[], bool], timeout: float = 10.0) -> None:
    """Let a Textual test app run until *condition* holds, failing after *timeout* seconds."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await pilot.pause(0.05)
