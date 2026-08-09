#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""The driver: psi4's entry point, and the glue between the two worlds.

``energy('hybrid_mp2')`` lands in :func:`run_hybrid_mp2`, which runs the
sequence the whole template exists to demonstrate:

1. SCF in psi4.
2. The psi4-world binary (``hybrid_mp2.so``) transforms the integrals and
   leaves them on the wavefunction. It links psi4 and only psi4.
3. The adapter copies buffers into the host-free ``Reference``; psi4 types
   stop here.
4. The contracted stage computes the correlation energy in einsums, on
   whichever backend ``STAGE_BACKEND`` selects. The ``cpp`` backend is the
   stage module built under ``cpp/``, which links Einsums and only Einsums.
5. The finish reads the graph-written scalars and reports through psi4.

Backend selection is explicit, never a fallback: asking for ``cpp`` without
having built it is an error, because a silent Python fallback would report
C++ timings that are not C++.
"""

import os
import sys

import psi4
import psi4.driver.p4util as p4util

from einsums import stages as estages

from . import adapter
from . import stages as _stage_defs  # noqa: F401  (registers the python backend)
from .mp2 import finish_mp2_energy, plan_mp2_energy

__all__ = ["run_hybrid_mp2"]

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


def _select_backend(backend: str) -> None:
    """Point the contracted stages at *backend*, loading the module for cpp.

    The stage module lives on ``PYTHONPATH`` like any module; the build
    directory under ``cpp/`` is appended here so the built-in-place template
    works without environment setup. ``load_stage_module`` refuses a module
    built against a different Einsums world before reading anything out of
    it, so a stale build fails loudly at selection time.
    """
    st = estages.get_stage("mp2_energy")
    if backend == "cpp" and "cpp" not in st.backends:
        # build/cpp is the merged single-configure layout; cpp/build the
        # standalone one. Either works with the module left where it was built.
        for build_dir in (
            os.path.join(_PLUGIN_DIR, "build", "cpp"),
            os.path.join(_PLUGIN_DIR, "cpp", "build"),
        ):
            if os.path.isdir(build_dir) and build_dir not in sys.path:
                sys.path.append(build_dir)
        estages.load_stage_module("hybrid_mp2_stages")
    st.select(backend)


def run_hybrid_mp2(name, **kwargs):
    r"""Called by :py:func:`psi4.energy` for ``energy('hybrid_mp2')``."""
    kwargs = p4util.kwargs_lower(kwargs)

    ref_wfn = kwargs.get("ref_wfn", None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # The psi4-world binary: integral transform, result left on the wfn.
    wfn = psi4.core.plugin("hybrid_mp2.so", ref_wfn)

    # Buffers cross; psi4 types stop here.
    ref = adapter.from_wavefunction(wfn)

    backend = psi4.core.get_local_option("HYBRID_MP2", "STAGE_BACKEND").lower()
    _select_backend(backend)

    args = plan_mp2_energy(ref)
    result = estages.get_stage("mp2_energy").call(**args)
    energies = finish_mp2_energy(ref, result)

    psi4.core.print_out(
        f"\n  hybrid_mp2 ({backend} backend)\n"
        f"    SCF total energy          {ref.e_scf:20.12f}\n"
        f"    MP2 correlation energy    {energies['e_corr']:20.12f}\n"
        f"      opposite-spin           {energies['e_os']:20.12f}\n"
        f"      same-spin               {energies['e_ss']:20.12f}\n"
        f"    MP2 total energy          {energies['e_total']:20.12f}\n\n"
    )

    psi4.core.set_variable("MP2 CORRELATION ENERGY", energies["e_corr"])
    psi4.core.set_variable("MP2 OPPOSITE-SPIN CORRELATION ENERGY", energies["e_os"])
    psi4.core.set_variable("MP2 SAME-SPIN CORRELATION ENERGY", energies["e_ss"])
    psi4.core.set_variable("MP2 TOTAL ENERGY", energies["e_total"])
    psi4.core.set_variable("CURRENT ENERGY", energies["e_total"])
    psi4.core.set_variable("CURRENT REFERENCE ENERGY", ref.e_scf)
    wfn.set_energy(energies["e_total"])
    return wfn


# Integration with driver routines
psi4.driver.procedures["energy"]["hybrid_mp2"] = run_hybrid_mp2
