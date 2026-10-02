# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``einsums stages``: the promote and extract tools of :mod:`einsums.stages`.

The arguments are handed to that package's own parser, so its options are defined once,
and it is imported only when this command runs (it costs every other command 50 ms).
``python -m einsums.stages`` remains the same tool.
"""

from __future__ import annotations

import argparse


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "stages",
        help="Generate and scaffold C++ for Python stages (promote, extract)",
        add_help=False,  # --help belongs to the forwarded parser
        prefix_chars="\0",  # take every argument, options included, as data
    )
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from ..stages.__main__ import main

    return main(args.args, prog=f"{args.top_prog} stages")
