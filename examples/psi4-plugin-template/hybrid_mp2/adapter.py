#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""psi4 -> :class:`.reference.Reference`: the one module that touches the host.

Everything crosses as buffers. The wavefunction's Matrices and Vectors are
copied out through ``to_array()``; no psi4 type survives past this function,
and no einsums type appears in it. That is the sealed-worlds rule at the
Python layer: the two libraries meet only in numpy. (It needs no ``import
psi4`` of its own - the wavefunction comes in as an argument - which is what
lets the rest of the method stay importable with no psi4 installed.)
"""

import numpy as np

from .reference import Reference

__all__ = ["from_wavefunction"]

#: Where the psi4-world plugin binary stashes the transformed integrals.
IAJB_VARIABLE = "HYBRID_MP2 IAJB"


def from_wavefunction(wfn) -> Reference:
    """Build the host-free :class:`Reference` from a converged wavefunction.

    Expects the plugin's C++ half to have run already: it computes the MO
    ``(ia|jb)`` integrals in the psi4 world and leaves them on the
    wavefunction as an array variable, which is the buffer-level handoff.
    """
    if not wfn.has_array_variable(IAJB_VARIABLE):
        raise RuntimeError(
            f"wavefunction carries no {IAJB_VARIABLE!r}; the plugin binary "
            f"(hybrid_mp2.so) must run before the adapter reads its result"
        )

    eps_occ = np.asarray(wfn.epsilon_a_subset("MO", "ACTIVE_OCC").to_array())
    eps_vir = np.asarray(wfn.epsilon_a_subset("MO", "ACTIVE_VIR").to_array())
    nocc, nvir = eps_occ.size, eps_vir.size

    iajb = np.ascontiguousarray(
        np.asarray(wfn.array_variable(IAJB_VARIABLE).to_array())
        .reshape(nocc, nvir, nocc, nvir)
    )
    return Reference(
        e_scf=float(wfn.energy()),
        eps_occ=eps_occ,
        eps_vir=eps_vir,
        iajb=iajb,
    )
