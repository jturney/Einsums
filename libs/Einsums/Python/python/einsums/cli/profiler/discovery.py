# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Finding profiler servers over mDNS, when the optional ``zeroconf`` package is installed.

Only macOS servers advertise themselves (``Server::register_mdns`` uses Bonjour), so on
Linux this finds remote Macs at most; the app always tries the default port as well.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any

from .client import DEFAULT_HOST, MDNS_SERVICES

try:
    from zeroconf import ServiceBrowser, ServiceStateChange, Zeroconf

    HAVE_ZEROCONF = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_ZEROCONF = False


def _local_addresses() -> set[str]:
    try:
        return {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None)}
    except OSError:
        return set()


class ServerBrowser:
    """Calls ``on_found(host, port, executable)`` from zeroconf's thread for each server seen."""

    def __init__(self, on_found: Callable[[str, int, str], None]) -> None:
        self._on_found = on_found
        self._zeroconf: Any = None
        self._browser: Any = None

    @property
    def running(self) -> bool:
        return self._zeroconf is not None

    def start(self) -> bool:
        if not HAVE_ZEROCONF or self.running:
            return self.running
        self._zeroconf = Zeroconf()
        self._browser = ServiceBrowser(self._zeroconf, list(MDNS_SERVICES), handlers=[self._on_change])
        return True

    def stop(self) -> None:
        if self._browser is not None:
            self._browser.cancel()
        if self._zeroconf is not None:
            self._zeroconf.close()
        self._browser = self._zeroconf = None

    def _on_change(self, zeroconf: Any, service_type: str, name: str, state_change: Any) -> None:
        if state_change not in (ServiceStateChange.Added, ServiceStateChange.Updated):
            return
        info = zeroconf.get_service_info(service_type, name)
        if info is None or not info.parsed_addresses():
            return
        host = info.parsed_addresses()[0]
        # The server binds to loopback, but Bonjour advertises it on every interface.
        if host in ("::1", "0:0:0:0:0:0:0:1") or host in _local_addresses():
            host = DEFAULT_HOST
        props = {k.decode(): (v.decode() if v else "") for k, v in info.properties.items()}
        self._on_found(host, info.port, props.get("exe", "unknown"))
