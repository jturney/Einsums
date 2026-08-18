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
