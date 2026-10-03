# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""The ``waggle`` command: the terminal viewer, and ``report`` and ``diff`` for saved sessions.

A library that ships its own command (``einsums profiler``) calls :func:`main` with its program
name, its plugins, a window title and how its programs start a server.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .client import DEFAULT_HOST, DEFAULT_PORT, parse_endpoint

#: Subcommands that work on saved sessions without the viewer.
ACTIONS = ("report", "diff")


def viewer_parser(prog: str, server_hint: str = "WAGGLE_SERVER=1") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Terminal viewer for Waggle profiles. With no arguments it connects to "
        f"{DEFAULT_HOST}:{DEFAULT_PORT}, where a program run with {server_hint} listens.",
        epilog=f"Without the viewer: '{prog} report FILE' prints a saved session's hotspots or tree, and "
        f"'{prog} diff A B' compares two sessions (see their --help).",
    )
    p.add_argument("endpoints", nargs="*", metavar="HOST:PORT", help="servers to connect to (a bare PORT means localhost)")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"host for --port (default: {DEFAULT_HOST})")
    p.add_argument("--port", type=int, nargs="+", default=[], help="server port(s) on --host")
    p.add_argument("--load", metavar="FILE", nargs="+", default=[], help="open saved session file(s)")
    p.add_argument("--replay", metavar="FILE", help="replay a stream recorded with --record")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier (default: 1)")
    p.add_argument("--record", metavar="FILE", help="record the incoming stream to a .jsonl file")
    p.add_argument("--no-mdns", action="store_true", help="do not discover servers over mDNS")
    return p


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "waggle",
    plugins: Sequence = (),
    title: str = "Waggle",
    server_hint: str = "WAGGLE_SERVER=1",
) -> int:
    """Run the command line @p argv: ``report`` or ``diff`` first, else the viewer's arguments.

    The viewer takes servers as positional arguments, so ``report`` and ``diff`` cannot be
    ordinary subcommands beside them; the first argument picks. ``view`` may name the viewer
    explicitly.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ACTIONS:
        from . import report

        parser = argparse.ArgumentParser(prog=prog)
        report.add_parsers(parser.add_subparsers(dest="action_name", required=True))
        parsed = parser.parse_args(args)
        return parsed.action(parsed)
    if args and args[0] == "view":
        args = args[1:]
    return run(viewer_parser(prog, server_hint).parse_args(args), prog=prog, plugins=plugins, title=title)


def run(args: argparse.Namespace, *, prog: str = "waggle", plugins: Sequence = (), title: str = "Waggle") -> int:
    """Open the viewer with @p plugins first, then any an installed package registers."""
    try:
        from .app import ProfilerApp
        from .plugin import discover_plugins, merge_plugins
    except ImportError as exc:
        print(f"{prog} needs Textual ({exc}); install it with: conda install -c conda-forge textual")
        return 1
    try:
        endpoints = [(args.host, port) for port in args.port] + [parse_endpoint(e) for e in args.endpoints]
    except ValueError as exc:
        print(f"{prog}: bad endpoint ({exc}); expected HOST:PORT or PORT")
        return 2
    ProfilerApp(
        endpoints,
        load=args.load,
        replay=args.replay,
        replay_speed=args.speed,
        record=args.record,
        mdns=not args.no_mdns,
        plugins=merge_plugins(list(plugins), discover_plugins()),
        title=title,
    ).run()
    return 0
