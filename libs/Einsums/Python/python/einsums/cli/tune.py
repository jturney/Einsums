# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``einsums tune``: measure this machine and write its hardware profile, or report on the one in effect.

Every argument goes to the library's tune, the same code the ``einsums-tune`` executable runs, so
the three spellings cannot drift apart; ``einsums tune --help`` lists them.
"""

from __future__ import annotations

import argparse


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tune",
        help="Measure this machine for its hardware profile, or report on the profile in effect",
        add_help=False,
        prefix_chars="\0",
    )
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from .. import hardware

    return hardware.tune(list(args.arguments))
