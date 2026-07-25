# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``--einsums:*`` command-line flags reaching the runtime from a Python script.

A C++ program hands its argv to ``einsums::initialize``, so ``--einsums:*`` flags
work directly. A Python process has no such handoff -- the runtime is started
from ``einsums.rc`` (see ``PyEinsumsMain.cpp::argv_from_rc``) -- so flags typed
on the python command line were silently ignored. That was worst for
``--einsums:debug:no-attach-debugger``, whose absence turns a crash into a
process that hangs forever inside ``util::attach_debugger()``.

``einsums/__init__.py`` now claims the ``--einsums:`` namespace out of
``sys.argv`` at import time and forwards it to ``_initialize_from_rc``.
"""
from __future__ import annotations

import subprocess
import sys

import einsums


def _run(args, code):
    """Run `code` in a fresh interpreter with `args` after the script name."""
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=False,
    )


# ── the extraction helper, in isolation ────────────────────────────────────


def test_extract_pulls_only_einsums_flags_and_preserves_order():
    argv = ["prog", "--einsums:log:level=3", "--other", "7", "--einsums:pass:verbose"]
    taken = einsums._extract_cli_options(argv)
    assert taken == ["--einsums:log:level=3", "--einsums:pass:verbose"]
    # argv is mutated in place, keeping argv[0] and the non-einsums arguments.
    assert argv == ["prog", "--other", "7"]


def test_extract_leaves_argv_untouched_when_no_einsums_flags():
    argv = ["prog", "--other", "7"]
    assert einsums._extract_cli_options(argv) == []
    assert argv == ["prog", "--other", "7"]


def test_extract_ignores_argv0_even_if_it_looks_like_a_flag():
    argv = ["--einsums:not-a-flag-here", "--einsums:log:level=1"]
    assert einsums._extract_cli_options(argv) == ["--einsums:log:level=1"]
    assert argv == ["--einsums:not-a-flag-here"]


# ── end to end, in a subprocess (this process' argv has no einsums flags) ──


def test_flags_are_claimed_from_the_command_line_and_stripped():
    code = (
        "import sys, einsums\n"
        "print('OPTS', einsums.cli_options)\n"
        "print('ARGV', sys.argv[1:])\n"
    )
    r = _run(["--einsums:pass:verbose", "--mine", "3"], code)
    assert r.returncode == 0, r.stderr
    assert "OPTS ['--einsums:pass:verbose']" in r.stdout, r.stdout
    # The einsums flag is removed so a script's own argparse does not choke.
    assert "ARGV ['--mine', '3']" in r.stdout, r.stdout


def test_forwarded_flag_does_not_prevent_runtime_startup():
    # The flag has to be accepted by the runtime's parser, not just forwarded:
    # an unknown or malformed option would fail initialization.
    code = (
        "import einsums\n"
        "from einsums import linalg\n"  # first real use starts the runtime
        "assert einsums._core._is_initialized()\n"
        "print('STARTED', einsums.cli_options)\n"
    )
    r = _run(["--einsums:debug:no-attach-debugger"], code)
    assert r.returncode == 0, r.stderr
    assert "STARTED ['--einsums:debug:no-attach-debugger']" in r.stdout, r.stdout


def test_no_flags_still_starts_cleanly():
    code = (
        "import einsums\n"
        "from einsums import linalg\n"
        "assert einsums.cli_options == []\n"
        "assert einsums._core._is_initialized()\n"
        "print('OK')\n"
    )
    r = _run([], code)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
