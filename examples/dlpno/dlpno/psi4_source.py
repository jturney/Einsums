#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Build a :class:`~dlpno.reference.Reference` from a psi4 wavefunction.

The only module in this package that imports psi4. Everything crosses as plain
numpy buffers, so psi4 and einsums stay unlinked, matching the buffer-level
bridge in ``einsums.interop.psi4``.

The three-index integrals come out *raw*: ``(Q|mn)`` with no metric contracted
in, because DLPNO applies the metric per domain (it solves ``J_dom x = (Q|ia)``
for each LMO pair rather than fitting once globally).
"""

import numpy as np
import psi4

from .reference import Reference

__all__ = ["from_psi4", "raw_three_index", "aux_metric", "localize"]


def localize(basis, C_act_occ, kind="BOYS"):
    """Localize the active occupied orbitals, psi4's ``DLPNO_LOCAL_ORBITALS``.

    Convergence and iteration limits come from psi4's ``LOCAL_CONVERGENCE`` and
    ``LOCAL_MAXITER`` options, which the localizer reads when it is built; the
    Python class exposes no setters for them.
    """
    kind = kind.upper()
    if kind not in ("BOYS", "PIPEK_MEZEY"):
        raise ValueError(f"unknown localization {kind!r}; use BOYS or PIPEK_MEZEY")
    localizer = psi4.core.Localizer.build(kind, basis, C_act_occ)
    localizer.localize()
    if not localizer.converged:
        raise RuntimeError(f"{kind} localization did not converge")
    return np.asarray(localizer.L)


def aux_metric(aux):
    """The auxiliary Coulomb metric ``(P|Q)``."""
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(aux)
    J = np.squeeze(np.asarray(mints.ao_eri(aux, zero, aux, zero)))
    return np.ascontiguousarray(J)


def raw_three_index(primary, aux):
    """Raw three-index integrals ``(Q|mn)`` of shape ``(naux, nbf, nbf)``.

    Dense: every auxiliary function against every AO pair. That is the
    "build it all, slice per domain" strategy, exact for the domains DLPNO
    actually uses and fine at example scale. The linear-scaling replacement is
    psi4's ``DLPNO::compute_qia``, which drives a screened shell-triplet loop
    and transforms straight into each atom's extended LMO/PAO domain; it plugs
    in behind :attr:`Reference.eri_3index` without changing anything downstream.
    """
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(primary)
    Qmn = np.squeeze(np.asarray(mints.ao_eri(aux, zero, primary, primary)))
    return np.ascontiguousarray(Qmn.reshape(aux.nbf(), primary.nbf(), primary.nbf()))


def _atom_maps(basis):
    """Which basis functions sit on each atom."""
    natom = basis.molecule().natom()
    out = [[] for _ in range(natom)]
    for mu in range(basis.nbf()):
        out[basis.function_to_center(mu)].append(mu)
    return out


def from_psi4(wfn, aux=None, localization="BOYS", freeze_core=False):
    """Assemble a :class:`Reference` from a converged psi4 wavefunction.

    ``aux`` defaults to the RIFIT auxiliary basis for the primary basis, which
    is what psi4's DLPNO uses for the correlation integrals.
    """
    primary = wfn.basisset()
    mol = wfn.molecule()
    if aux is None:
        aux = psi4.core.BasisSet.build(
            mol, "DF_BASIS_MP2", "", "RIFIT", primary.name()
        )

    nocc = wfn.nalpha()
    nfrzc = primary.n_frozen_core("TRUE", mol) if freeze_core else 0

    Ca = np.asarray(wfn.Ca())
    C_occ = Ca[:, :nocc]
    C_act_occ = psi4.core.Matrix.from_array(Ca[:, nfrzc:nocc])

    ref = Reference(
        S=np.asarray(wfn.S()),
        F=np.asarray(wfn.Fa()),
        C_occ=C_occ,
        C_lmo=localize(primary, C_act_occ, localization),
        eri_3index=raw_three_index(primary, aux),
        metric=aux_metric(aux),
        atom_to_bf=_atom_maps(primary),
        atom_to_ribf=_atom_maps(aux),
        # psi4 keeps core orbitals correlated but scales their PNO threshold;
        # once they are frozen outright there is nothing left to scale.
        n_core=0 if freeze_core else primary.n_frozen_core("TRUE", mol),
        e_scf=wfn.energy(),
    )
    return ref.validate()
