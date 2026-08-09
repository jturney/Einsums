#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""The MP2 numerics, plus the planning and finishing that stay Python.

The numerics is a free function taking only cross-boundary types, which is
what makes it promotable: its module is the thing the C++ port under ``cpp/``
replaces, so its boundary is the boundary the port has to reproduce.

The split follows the framework's plan / numerics / finish pattern:

* :func:`plan_mp2_energy` turns the host-free :class:`.reference.Reference`
  into the stage's arguments (einsums tensors, host scalars).
* :func:`mp2_energy` is the promotable numerics: one captured graph over the
  ``i <= j`` pair loop, replayed as an OpenMP team.
* :func:`finish_mp2_energy` reads the graph-written scalars back into floats
  and derives the spin components, which is host-side accounting.
"""

import numpy as np

import einsums
from einsums import linalg as la
import einsums.graph as cg

from .contracts import Mp2Energy
from .reference import Reference

__all__ = ["plan_mp2_energy", "mp2_energy", "finish_mp2_energy"]


def _zeros(name, shape):
    return einsums.create_zero_tensor(name, [int(s) for s in shape], dtype="float64")


def _from_numpy(name, arr):
    T = _zeros(name, arr.shape)
    if arr.size:
        np.asarray(T)[...] = arr
    return T


def plan_mp2_energy(ref: Reference) -> dict:
    """The stage's arguments, from the host-free reference.

    Nothing here is worth promoting: it is three buffer copies. It exists as
    its own function so the stage's contract stays narrow - the stage takes
    tensors, not the ``Reference``, so a new field on the dataclass never
    widens the C++ signature.
    """
    return dict(
        iajb=_from_numpy("(ia|jb)", ref.iajb),
        eps_occ=_from_numpy("eps occ", ref.eps_occ),
        eps_vir=_from_numpy("eps vir", ref.eps_vir),
    )


def mp2_energy(iajb, eps_occ, eps_vir) -> Mp2Energy:
    """RHF MP2 correlation energy from MO ``(ia|jb)`` integrals.

    Pair-driven over ``i <= j``, with every tensor operation captured into
    one graph and replayed under the OpenMP executor: the per-pair chains are
    independent, so the parallelism worth having is across graph nodes, with
    each node's kernel left serial underneath (never a Python thread pool
    around BLAS; see the DLPNO example for why that returns wrong numbers).

    Per pair::

        I_ab   = (ia|jb)                slice of the input
        K_ab   = 2 I_ab - I_ba          permute + axpby
        D_ab   = e_i + e_j - e_a - e_b  outer_sum (hoisted) + shift
        T_ab   = I_ab / D_ab            direct_division
        e_pair = sum_ab K_ab T_ab       dot, written into a scalar tensor

    The energies come back as graph-written scalar TENSORS because a captured
    stage returns before its graphs run; the caller reads them after.
    """
    nocc = int(np.asarray(eps_occ).shape[0])
    nvir = int(np.asarray(eps_vir).shape[0])
    eo = np.asarray(eps_occ)

    e_corr = _zeros("E corr", [1])
    e_os = _zeros("E os", [1])

    pairs = [(i, j) for i in range(nocc) for j in range(i, nocc)]
    if pairs:
        # -e_a - e_b is pair-independent; build it once, eagerly, before the
        # capture. Same code would run captured - stage code must stay
        # capture-transparent - but a hoisted constant needs no replay.
        base = _zeros("-ea-eb", [nvir, nvir])
        la.outer_sum(base, [eps_vir, eps_vir], [-1.0, -1.0])

        # Every operand a captured op touches must exist before the capture
        # and stay put: the graph records addresses, so scratch created (or
        # a list reallocated) mid-capture is the classic dangling-node bug.
        scratch = [
            dict(I=_zeros("I", [nvir, nvir]), K=_zeros("K", [nvir, nvir]),
                 D=_zeros("D", [nvir, nvir]), T=_zeros("T", [nvir, nvir]),
                 e=_zeros("e pair", [1]), eos=_zeros("e pair os", [1]))
            for _ in pairs
        ]
        views = [iajb[i, :, j, :] for (i, j) in pairs]

        g = cg.Graph("mp2 pairs")
        with cg.capture(g):
            for (i, j), s, I_view in zip(pairs, scratch, views):
                la.axpby(1.0, I_view, 0.0, s["I"])            # densify the block
                einsums.permute("ab <- ba", s["K"], s["I"])   # K = I^T
                la.axpby(2.0, s["I"], -1.0, s["K"])           # K = 2 I - I^T
                la.axpby(1.0, base, 0.0, s["D"])
                la.shift(float(eo[i] + eo[j]), s["D"])        # D = e_i+e_j-e_a-e_b
                la.direct_division(1.0, s["I"], s["D"], 0.0, s["T"])
                la.dot(s["e"], s["K"], s["T"])                # pointer-writer dot:
                la.dot(s["eos"], s["I"], s["T"])              # returning form throws here
                f = 1.0 if i == j else 2.0
                la.axpby(f, s["e"], 1.0, e_corr)
                la.axpby(f, s["eos"], 1.0, e_os)
        g.set_executor(cg.OpenMPExecutor())
        g.execute()

    return Mp2Energy(e_corr=e_corr, e_os=e_os)


def finish_mp2_energy(ref: Reference, result: Mp2Energy) -> dict:
    """Read the stage's scalar tensors back into the numbers psi4 reports."""
    e_corr = float(np.asarray(result.e_corr)[0])
    e_os = float(np.asarray(result.e_os)[0])
    return dict(
        e_corr=e_corr,
        e_os=e_os,
        e_ss=e_corr - e_os,
        e_total=ref.e_scf + e_corr,
    )
