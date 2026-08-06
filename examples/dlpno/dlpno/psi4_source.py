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

__all__ = ["from_psi4", "raw_three_index", "aux_metric", "localize", "ao_dipole",
           "grid_block_provider"]


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


def ao_dipole(basis):
    """AO dipole integrals, ``(3, nbf, nbf)``, for the dipole pair prescreening."""
    mints = psi4.core.MintsHelper(basis)
    return np.ascontiguousarray(np.array([np.asarray(d) for d in mints.ao_dipole()]))


def grid_block_provider(basis, spherical_points=50, radial_points=25,
                        pruning_scheme="ROBUST", basis_tolerance=1.0e-10):
    """Return a callable streaming the DFT grid as ``(phi, w, bf_map)`` blocks.

    The differential overlap integrals are numerical quadrature, so they need
    basis-function values on a grid. Those depend on the PAO coefficients, which
    :meth:`DLPNOBase.setup_orbitals` builds rather than the bridge, so the grid
    has to cross the boundary rather than the finished integrals.

    It crosses as a *stream* because it is the one quantity here that does not
    fit comfortably: collocating every point against every basis function at
    once is roughly 550 MB at ethanol/cc-pVTZ, where a single block is a few
    hundred kilobytes. psi4's own DOI routine blocks it for the same reason.

    This is the only entry in :class:`Reference` that is a callable rather than
    a buffer. The alternative was to duplicate the PAO projector in this module
    so the integrals could be finished here, which would put the same piece of
    physics in two places.

    Grid defaults match psi4's ``DOI_*`` options so the domains come out the
    same. ``DFT_BASIS_TOLERANCE`` and friends are set globally rather than
    passed, because the Python ``DFTGrid.build`` takes only the integer and
    string option maps and reads the rest from the environment.
    """
    mol = basis.molecule()
    psi4.set_options({"DFT_BASIS_TOLERANCE": basis_tolerance,
                      "DFT_BS_RADIUS_ALPHA": 1.0,
                      "DFT_PRUNING_ALPHA": 1.0,
                      "DFT_BLOCK_MAX_RADIUS": 3.0,
                      "DFT_WEIGHTS_TOLERANCE": 1e-15})
    grid = psi4.core.DFTGrid.build(
        mol, basis,
        {"DFT_SPHERICAL_POINTS": spherical_points,
         "DFT_RADIAL_POINTS": radial_points,
         "DFT_BLOCK_MIN_POINTS": 100,
         "DFT_BLOCK_MAX_POINTS": 256},
        {"DFT_PRUNING_SCHEME": pruning_scheme,
         "DFT_RADIAL_SCHEME": "TREUTLER",
         "DFT_NUCLEAR_SCHEME": "TREUTLER",
         "DFT_GRID_NAME": "",
         "DFT_BLOCK_SCHEME": "OCTREE"})

    def blocks():
        # BasisFunctions is stateful and sized to the largest block, so it is
        # built once here and its buffer re-trimmed per block.
        point_funcs = psi4.core.BasisFunctions(basis, grid.max_points(), basis.nbf())
        for block in grid.blocks():
            point_funcs.compute_functions(block)
            bf_map = block.functions_local_to_global()
            npoints = block.npoints()
            phi = np.asarray(point_funcs.basis_values()["PHI"])[:npoints, :len(bf_map)]
            # Copy, explicitly. Both of these are views into buffers psi4 reuses
            # across blocks - PHI is one allocation sized to the largest block,
            # and the trimmed slice of it is sometimes already contiguous, so
            # ascontiguousarray alone returns a view and hands out memory the
            # next iteration overwrites. A consumer that reads a block straight
            # away (prep_sparsity) never notices; one that accumulates blocks
            # (reference_io.save_reference) silently gets the last block's
            # values for every block that shared the buffer.
            yield (np.array(phi, dtype=np.float64, copy=True),
                   np.array(np.asarray(block.w())[:npoints], dtype=np.float64, copy=True),
                   list(bf_map))

    return blocks


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
        dipole_ao=ao_dipole(primary),
        grid_blocks=grid_block_provider(primary),
        # psi4 keeps core orbitals correlated but scales their PNO threshold;
        # once they are frozen outright there is nothing left to scale.
        n_core=0 if freeze_core else primary.n_frozen_core("TRUE", mol),
        e_scf=wfn.energy(),
    )
    return ref.validate()
