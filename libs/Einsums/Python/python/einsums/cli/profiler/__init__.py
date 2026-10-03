# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``einsums profiler``: Waggle's terminal viewer with Einsums' panels.

It connects to the server a program starts with ``--einsums:profile:server``, opens session
files written by ``--einsums:profile:save`` or by its own "Save session", and replays streams it
recorded. ``report`` and ``diff`` print saved sessions without the viewer. Everything but the
TaskPool and compute-graph panels (:mod:`.einsums_plugin`) is Waggle's (:mod:`waggle.cli`).
"""

from __future__ import annotations

import argparse


def register(subparsers) -> None:
    # The viewer takes servers as positional arguments, so the arguments go to Waggle whole; its
    # first one picks report, diff or the viewer.
    p = subparsers.add_parser(
        "profiler",
        help="View live or saved profiles in the terminal (also: report, diff)",
        add_help=False,
        prefix_chars="\0",
    )
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(func=dispatch)


def dispatch(args: argparse.Namespace) -> int:
    from waggle.cli import main

    from .einsums_plugin import einsums_viewer_plugin

    return main(
        args.args,
        prog=f"{args.top_prog} profiler",
        plugins=[einsums_viewer_plugin()],
        title="Einsums profiler",
        server_hint="--einsums:profile:server",
    )
