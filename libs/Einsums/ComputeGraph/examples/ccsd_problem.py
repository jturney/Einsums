# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""The closed-shell DF-CCSD problem every stage of the CCSD tutorial solves.

Water in cc-pVTZ, read from the DLPNO example's fixture, so the tutorial runs offline with no
integral engine and no psi4: 5 occupied orbitals, 53 virtuals, 141 auxiliary functions. Every
two-electron integral is density fitted, (pq|rs) = sum_Q B[Q,p,q] B[Q,r,s], as in psi4's DF-CCSD.

This is setup, done once in NumPy: canonical orbitals from the converged Fock matrix, and the
fitted three-index tensor transformed to them. Run it on its own to see the problem's size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

FIXTURE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "examples", "dlpno", "fixtures", "water-ccpvtz.npz")
)

# psi4 1.11a1 on the fixture's molecule and basis, scf_type, mp2_type and cc_type all DF,
# freeze_core false, e_convergence and d_convergence 1e-10, r_convergence 1e-9. The MP2 energy is
# the fixture's own; the CCSD energy was computed once with the same settings.
PSI4_DF_MP2 = -0.275204895580
PSI4_DF_CCSD = -0.281138811529

# How far stage 1 may sit from PSI4_DF_CCSD. The fixture's SCF was fitted in the JK basis and its
# CCSD in cc-pVTZ-RI, and on that mix stage 1 and psi4 differ by 1.3e-7. With the SCF fitted in
# cc-pVTZ-RI too, the same equations on psi4's own orbitals and integrals agree with it to 1e-11.
PSI4_DF_CCSD_TOLERANCE = 2e-7

# Stage 1's own energy, which every later stage reproduces.
REFERENCE_CCSD = -0.2811389390742597


@dataclass
class Problem:
    nocc: int
    nvir: int
    naux: int
    eps_occ: np.ndarray  # orbital energies, occupied
    eps_vir: np.ndarray  # orbital energies, virtual
    B: np.ndarray  # B[Q,p,q], fitted three-index tensor over all molecular orbitals, occupied first


def load() -> Problem:
    z = np.load(FIXTURE, allow_pickle=False)

    # Canonical orbitals: the converged Fock matrix diagonalized in the orthogonalized basis.
    values, vectors = np.linalg.eigh(z["S"])
    orthogonalizer = vectors @ np.diag(values**-0.5) @ vectors.T
    energies, rotation = np.linalg.eigh(orthogonalizer.T @ z["F"] @ orthogonalizer)
    orbitals = orthogonalizer @ rotation
    nocc = int(z["C_occ"].shape[1])

    # B = J^(-1/2) (Q|mn), then to the molecular orbitals.
    mvals, mvecs = np.linalg.eigh(z["metric"])
    inv_sqrt = mvecs @ np.diag(np.where(mvals > 1e-10, mvals**-0.5, 0.0)) @ mvecs.T
    three = np.einsum("PQ,Qmn,mp,nq->Ppq", inv_sqrt, z["eri_3index"], orbitals, orbitals, optimize=True)

    return Problem(
        nocc=nocc,
        nvir=orbitals.shape[1] - nocc,
        naux=three.shape[0],
        eps_occ=np.ascontiguousarray(energies[:nocc]),
        eps_vir=np.ascontiguousarray(energies[nocc:]),
        B=np.ascontiguousarray(three),
    )


if __name__ == "__main__":
    p = load()
    print(f"water/cc-pVTZ: {p.nocc} occupied, {p.nvir} virtual, {p.naux} auxiliary")
