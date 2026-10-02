# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""The TCP client for the profiler server, and the recording format.

The server writes newline-terminated JSON; a snapshot of a deep tree is far larger than
asyncio's default line limit, so lines are split from a byte buffer here rather than with
``StreamReader.readline``. Splitting bytes before decoding also keeps a multi-byte UTF-8
character that straddles two reads intact.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, TextIO

from .model import LogEntry, ProfileMeta, ProfileSnapshot, TimelineEvent, parse_log, parse_meta, parse_snapshot, parse_timeline

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 19216
#: The service type ``Server::register_mdns`` advertises (macOS only, through Bonjour).
MDNS_SERVICE = "_einsums-profile._tcp.local."


def parse_endpoint(text: str, default_host: str = DEFAULT_HOST) -> tuple[str, int]:
    """``host:port`` or a bare ``port``. Raises ``ValueError`` for anything else."""
    text = text.strip()
    if ":" in text:
        host, port = text.rsplit(":", 1)
        return host or default_host, int(port)
    return default_host, int(text)


class StreamState:
    """What one connection (or one replayed recording) has told us so far."""

    def __init__(self, log_capacity: int = 2000) -> None:
        self.meta: ProfileMeta | None = None
        self.snapshot: ProfileSnapshot | None = None
        self.timeline: list[TimelineEvent] = []
        self.log_entries: deque[LogEntry] = deque(maxlen=log_capacity)
        #: Total log entries ever appended, so a viewer can tell new ones from old after the deque wraps.
        self.log_total = 0

    def apply(self, msg: dict[str, Any]) -> str:
        """Fold *msg* into the state and return its type."""
        kind = msg.get("type", "")
        if kind == "meta":
            self.meta = parse_meta(msg)
        elif kind == "snapshot":
            self.snapshot = parse_snapshot(msg)
        elif kind == "timeline":
            self.timeline = parse_timeline(msg)
        elif kind in ("log", "output"):
            self.log_entries.append(parse_log(msg))
            self.log_total += 1
        return kind


class ProfileClient:
    """One connection to a profiler server."""

    def __init__(self, host: str, port: int, *, discovered: bool = False) -> None:
        self.host = host
        self.port = port
        #: Found through mDNS rather than asked for; such a client gives up once its server goes away.
        self.discovered = discovered
        self.state = StreamState()
        self.retry_now = asyncio.Event()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._ids = itertools.count(1)

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self, timeout: float = 2.0) -> bool:
        try:
            self._reader, self._writer = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), timeout)
        except (OSError, asyncio.TimeoutError):
            self._reader = self._writer = None
            return False
        self.state = StreamState()
        return True

    async def close(self) -> None:
        writer, self._reader, self._writer = self._writer, None, None
        for future in self._pending.values():
            if not future.done():
                future.set_result({"error": "disconnected"})
        self._pending.clear()
        if writer is not None:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        """Messages until the server closes the connection. Responses are routed to :meth:`request`."""
        reader = self._reader
        if reader is None:
            return
        buffer = b""
        while True:
            try:
                chunk = await reader.read(1 << 16)
            except (OSError, asyncio.IncompleteReadError):
                break
            if not chunk:
                break
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if msg.get("type") == "response":
                    future = self._pending.pop(msg.get("id", ""), None)
                    if future is not None and not future.done():
                        future.set_result(msg.get("data", {}))
                    continue
                yield msg
        await self.close()

    async def request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
        """Call a handler registered with ``Server::register_handler``, e.g. ``get_compute_graphs``.

        The reply is read by :meth:`messages`, so call this from a task other than the one
        iterating it; awaited inside that loop it can only time out.
        """
        if self._writer is None:
            return {"error": "not connected"}
        req_id = f"req_{next(self._ids)}"
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        line = json.dumps({"type": "request", "id": req_id, "method": method, "params": params or {}}) + "\n"
        try:
            self._writer.write(line.encode())
            await self._writer.drain()
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return {"error": "timeout"}
        except OSError as exc:
            return {"error": str(exc)}
        finally:
            self._pending.pop(req_id, None)


# ---------------------------------------------------------------------------
# Recording: one {"_ts": monotonic seconds, "_msg": message} object per line
# ---------------------------------------------------------------------------


class Recorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: TextIO | None = self.path.open("w")

    def write(self, msg: dict[str, Any]) -> None:
        if self._file is not None:
            self._file.write(json.dumps({"_ts": time.monotonic(), "_msg": msg}) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def read_recording(path: str | Path) -> Iterator[tuple[float, dict[str, Any]]]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                yield record.get("_ts", 0.0), record.get("_msg", {})
