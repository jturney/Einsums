# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""A Python process that aborts after the runtime starts still says where it aborted.

The runtime used to install a SIGABRT handler on every start, whatever
install-signal-handlers said, and the handler left through ``_Exit(-1)``. In a
Python process that replaced the interpreter's faulthandler, so an abort ended
the process with exit status 255 and nothing else: no traceback, no buffered
output, and a ctest verdict of "Failed" in place of "Subprocess aborted". That is
how a crash on the ThreadSanitizer leg came to leave no trace in the log.

This has to be a subprocess, since the outcome under test is the death of the
process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGABRT reporting is a POSIX signal disposition")


def test_an_abort_after_initialization_reaches_faulthandler():
    code = textwrap.dedent(
        """
        import faulthandler, os, sys
        import einsums

        # Any real use starts the runtime; importing the package alone does not.
        einsums.create_zero_tensor("t", [2], dtype="float64")
        assert einsums._core._is_initialized()
        faulthandler.enable()

        def the_aborting_frame():
            os.abort()

        the_aborting_frame()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ), timeout=300
    )
    assert result.returncode == -signal.SIGABRT, (
        f"an abort ended the process with status {result.returncode}, not SIGABRT\nstderr:\n{result.stderr[-3000:]}"
    )
    assert "Fatal Python error: Aborted" in result.stderr, result.stderr[-3000:]
    assert "the_aborting_frame" in result.stderr, result.stderr[-3000:]
