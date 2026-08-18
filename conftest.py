# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Root pytest configuration, loaded for every test in the tree.

Holds only what has to happen before any test runs, in any suite.
"""

import os
import sys

# Re-arm the sanitizer preload for subprocesses.
#
# On macOS dyld honors DYLD_INSERT_LIBRARIES and then REMOVES it from the
# environment the process can read, so a pytest run that ctest correctly
# preloaded cannot pass the preload on: a subprocess it spawns inherits an
# environment with no DYLD_INSERT_LIBRARIES, dlopens the instrumented
# extension with no runtime beneath it, and dies with
#
#     ERROR: Interceptors are not working. This may be because
#     AddressSanitizer is loaded too late (e.g. via dlopen).
#
# on its first intercepted call - which for this library is any OpenMP region,
# so it reaches tests that merely ask a subprocess what it measured. The build
# hands us the same path a second time under a name dyld does not scrub; put
# the real one back so children inherit it.
#
# Inert everywhere else: without a sanitizer build the carrier is unset, and on
# Linux LD_PRELOAD survives exec on its own, so the setdefault does nothing.
_preload = os.environ.get("EINSUMS_SANITIZER_PRELOAD")
if _preload:
    os.environ.setdefault(
        "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD",
        _preload,
    )

# Turn a hung test into a stack trace instead of a stopwatch reading.
#
# The DLPNO triples suites hang in CI perhaps one run in three, on Linux/mkl
# and Windows, while every other test on the same runner goes FASTER than on
# runs where they pass - so it is a hang, not contention. ctest reports it as
# "Timeout 1500 sec" and discards the test's output when it kills it, which
# says nothing about where the process stopped. 68 local runs across four
# thread and load configurations did not reproduce it, so the information has
# to come from the machine that does.
#
# faulthandler dumps every thread's Python stack, which is what identifies a
# deadlock: the C++ frames are absent but the thread that is waiting, and the
# call it is waiting in, are not. repeat=True because ONE dump cannot tell a
# deadlock from slow progress and two, a few minutes apart, can - identical
# stacks mean stuck.
#
# To a file rather than stderr: the whole problem is that the output of a
# killed test does not survive. The workflows already collect crash dumps, so
# this lands somewhere collectable. Off by setting the timeout to 0.
_stackdump_seconds = float(os.environ.get("EINSUMS_TEST_STACKDUMP_SECONDS", "600"))
if _stackdump_seconds > 0:
    import faulthandler
    import tempfile

    _stackdump_dir = os.environ.get("EINSUMS_TEST_STACKDUMP_DIR") or tempfile.gettempdir()
    try:
        os.makedirs(_stackdump_dir, exist_ok=True)
        # Kept open for the life of the process: faulthandler writes through
        # this object from its timer thread, and a closed file would take the
        # dump with it. Line buffered so a dump survives the kill that follows.
        _stackdump_file = open(  # noqa: SIM115
            os.path.join(_stackdump_dir, f"einsums-stackdump-{os.getpid()}.txt"),
            "w",
            buffering=1,
        )
    except OSError:
        # A read-only or missing dump directory is not a reason to fail a test
        # run; the dump is a diagnostic, not a subject.
        _stackdump_file = None
    if _stackdump_file is not None:
        faulthandler.dump_traceback_later(
            _stackdump_seconds, repeat=True, file=_stackdump_file, exit=False
        )

        import atexit

        @atexit.register
        def _drop_empty_stackdump() -> None:
            """Leave a file behind only when there was something to say.

            Every python test process arms this, and nearly all of them finish
            long before the timer fires. Keeping their empty files would bury
            the one dump that matters under hundreds that say nothing.
            """
            faulthandler.cancel_dump_traceback_later()
            try:
                empty = _stackdump_file.tell() == 0
                _stackdump_file.close()
                if empty:
                    os.unlink(_stackdump_file.name)
            except OSError:
                pass
