#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""hybrid_mp2: a psi4 plugin with einsums stages, in two sealed binaries.

The Einsums hybrid-framework plugin template. ``import hybrid_mp2`` in a psi4
input registers ``energy('hybrid_mp2')`` and loads the psi4-world binary; the
Einsums-world stage module under ``cpp/`` is loaded lazily, only when
``STAGE_BACKEND cpp`` asks for it.
"""

__version__ = "0.1"

import os

# The driver half needs the host; the method half must not. Without psi4 the
# package still imports - contracts, stages and numerics are host-free - which
# is what lets `python -m einsums.stages promote` and the differential tests
# under cpp/tests run in an environment with no psi4 at all.
try:
    import psi4
except ImportError:
    psi4 = None

if psi4 is not None:
    # Load Python modules (registers the energy procedure and the stages).
    from .pymodule import *  # noqa: F401,F403

    # Load the psi4-world binary. add_psi4_plugin leaves it in the cmake
    # build directory; accept it there or copied next to this file.
    _plugdir = os.path.dirname(os.path.abspath(__file__))
    for _sofile in (
        os.path.join(_plugdir, "hybrid_mp2.so"),
        os.path.join(_plugdir, "build", "hybrid_mp2.so"),
    ):
        if os.path.isfile(_sofile):
            psi4.core.plugin_load(_sofile)
            break
    else:
        raise ImportError(
            f"hybrid_mp2.so not found in {_plugdir} or {_plugdir}/build. Build "
            f"the psi4-world binary first; see the README next to this package."
        )
