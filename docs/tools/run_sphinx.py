#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""Run sphinx-build, and make a failed build fail again until it is fixed.

Sphinx builds incrementally: it re-reads only the documents that changed, and a warning from a
document it already read is not reported again. With ``-W`` a build that warned therefore failed
once, and the next build, finding nothing changed, reported success. On failure this removes the
cached environment, so the next build reads every document again and repeats every warning.

Usage::

    run_sphinx.py --doctree DIR -- SPHINX-BUILD-COMMAND...
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doctree", required=True, help="the -d directory the command caches its environment in")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the sphinx-build command line, after --")
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no sphinx-build command given")

    result = subprocess.run(command, check=False).returncode
    if result != 0:
        shutil.rmtree(args.doctree, ignore_errors=True)
    return result


if __name__ == "__main__":
    sys.exit(main())
