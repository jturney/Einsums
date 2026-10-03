# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``einsums profiler``: a terminal viewer for the Einsums profiler.

It connects to the TCP server a program starts with ``--einsums:profile:server``, opens
session files written by ``--einsums:profile:save`` or by its own "Save session", and
replays streams it recorded. Only the viewer needs Textual, imported on use; ``report`` and
``diff`` print saved sessions without it.
"""

from __future__ import annotations

import argparse

from .client import DEFAULT_HOST, DEFAULT_PORT, parse_endpoint


_ACTIONS = ("report", "diff")


def _viewer_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Terminal viewer for the Einsums profiler. With no arguments it connects to "
        f"{DEFAULT_HOST}:{DEFAULT_PORT}, where a program run with --einsums:profile:server listens.",
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


def register(subparsers) -> None:
    # The viewer takes servers as positional arguments, so report and diff cannot be
    # ordinary subcommands beside them; the first argument picks, and everything else
    # is the viewer's.
    p = subparsers.add_parser(
        "profiler",
        help="View live or saved profiles in the terminal (also: report, diff)",
        add_help=False,
        prefix_chars="\0",
    )
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(func=dispatch)


def dispatch(args: argparse.Namespace) -> int:
    prog = f"{args.top_prog} profiler"
    argv = args.args
    if argv and argv[0] in _ACTIONS:
        from . import report

        parser = argparse.ArgumentParser(prog=prog)
        report.add_parsers(parser.add_subparsers(dest="action_name", required=True))
        parsed = parser.parse_args(argv)
        return parsed.action(parsed)
    return run(_viewer_parser(prog).parse_args(argv))


def run(args: argparse.Namespace) -> int:
    try:
        from .app import ProfilerApp
        from .einsums_plugin import einsums_viewer_plugin
        from .plugin import discover_plugins, merge_plugins
    except ImportError as exc:
        print(f"einsums profiler needs Textual ({exc}); install it with: conda install -c conda-forge textual")
        return 1
    try:
        endpoints = [(args.host, port) for port in args.port] + [parse_endpoint(e) for e in args.endpoints]
    except ValueError as exc:
        print(f"einsums profiler: bad endpoint ({exc}); expected HOST:PORT or PORT")
        return 2
    ProfilerApp(
        endpoints,
        load=args.load,
        replay=args.replay,
        replay_speed=args.speed,
        record=args.record,
        mdns=not args.no_mdns,
        # Einsums' own panels always, then any other library's an installed package registers.
        plugins=merge_plugins([einsums_viewer_plugin()], discover_plugins()),
        title="Einsums profiler",
    ).run()
    return 0
