# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""The ``einsums`` command line: ``einsums <command>`` or ``python -m einsums <command>``.

* ``profiler``: the terminal profile viewer
* ``bench``: run the performance tests and track their results
* ``stages``: promote Python stages to C++ (:mod:`einsums.stages`)
* ``info``: the build, the Python environment and the machine, for a bug report
* ``options``: every runtime option, with its flag, environment variable and default
* ``completion``: a shell completion script
"""

from __future__ import annotations

import argparse
import sys

from . import bench, completion, info, options, profiler, stages


def build_parser(prog: str = "einsums") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Einsums developer tools.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    profiler.register(sub)
    bench.register(sub)
    stages.register(sub)
    info.register(sub)
    options.register(sub)
    completion.register(sub)
    parser.set_defaults(top_prog=prog)
    return parser


def main(argv: list[str] | None = None, prog: str = "einsums") -> int:
    args = build_parser(prog).parse_args(argv)
    # A command that starts the runtime (stages promote imports the stages module)
    # would otherwise write the profiler's text report into the working directory.
    from .. import rc

    if rc.profile_report is None:
        rc.profile_report = False
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
