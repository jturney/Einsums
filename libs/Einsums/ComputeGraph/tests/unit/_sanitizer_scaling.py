# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Sample-count scaling for the ComputeGraph fuzz / property tests.

Every fuzz case runs 30-50x slower under a sanitizer (ThreadSanitizer /
AddressSanitizer). Worse, a test that builds hundreds of graphs in one process
accumulates the ThreadSanitizer deadlock detector's lock-order graph until the
exit-time traversal overruns the stack and the process aborts (the failure mode
of the old single-file fuzzer). A sanitizer run only needs SOME executions to
trip a race / leak / UB, not the full sweep, so the fuzzers shrink their sample
counts when a sanitizer runtime is active. Structural and dtype coverage are
unchanged; only the number of random samples per test drops.

Two knobs, one for each way a fuzzer scales:

  * ``fuzz_seeds(n)``        - for ``@pytest.mark.parametrize("seed", range(n))``
  * ``sanitizer_examples(n)`` - for Hypothesis ``@settings(max_examples=n)``

Detection keys on the sanitizer *runtime* being loaded (the thing that makes it
slow and that grows the lock-order graph), not on a build flag: TSan/ASan export
``__tsan_init`` / ``__asan_init`` into the global symbol table, so they resolve
here whether the run came through ctest or a manual preload.
``EINSUMS_FUZZ_MAX_SEEDS`` overrides the caps explicitly (0 = no cap, i.e. always
run the full sweep; N > 0 = cap every fuzzer at N samples).
"""

from __future__ import annotations

import ctypes
import os
import warnings

_SANITIZER_SEED_CAP = 4     # seeds per range()-based fuzz test under a sanitizer
_SANITIZER_EXAMPLE_CAP = 8  # Hypothesis max_examples under a sanitizer


def _sanitizer_runtime_active() -> bool:
    try:
        main = ctypes.CDLL(None)
    except OSError:
        return False
    return any(hasattr(main, sym) for sym in ("__tsan_init", "__asan_init"))


_ENV = os.environ.get("EINSUMS_FUZZ_MAX_SEEDS")
if _ENV is not None:
    _OVERRIDE = int(_ENV)        # <= 0 means "no cap, full sweep"
    _CAPPING = _OVERRIDE > 0
else:
    _OVERRIDE = None
    _CAPPING = _sanitizer_runtime_active()

if _CAPPING:
    # Not silent: pytest surfaces this in its warnings summary, so a capped run
    # always announces that it is not the full sweep.
    _why = "EINSUMS_FUZZ_MAX_SEEDS" if _ENV is not None else "a sanitizer run"
    warnings.warn(
        f"fuzz sample counts capped under {_why} "
        f"(seeds <= {_OVERRIDE if _ENV is not None else _SANITIZER_SEED_CAP}, "
        f"Hypothesis examples <= {_OVERRIDE if _ENV is not None else _SANITIZER_EXAMPLE_CAP}); "
        f"set EINSUMS_FUZZ_MAX_SEEDS=0 to run the full sweep",
        stacklevel=2,
    )


def _cap(n: int, default_cap: int) -> int:
    if _ENV is not None:
        return n if _OVERRIDE <= 0 else min(n, _OVERRIDE)
    return min(n, default_cap) if _CAPPING else n


def fuzz_seeds(n: int):
    """``range`` of seeds for a ``parametrize("seed", ...)`` fuzz test: the full
    ``range(n)`` normally, a capped range under a sanitizer run (or an explicit
    ``EINSUMS_FUZZ_MAX_SEEDS``). Evaluated at collection time, so the reduced set
    shows up in the item count."""
    return range(_cap(n, _SANITIZER_SEED_CAP))


def sanitizer_examples(n: int) -> int:
    """Hypothesis ``max_examples`` for a property test: ``n`` normally, a capped
    value under a sanitizer run (or an explicit ``EINSUMS_FUZZ_MAX_SEEDS``)."""
    return _cap(n, _SANITIZER_EXAMPLE_CAP)
