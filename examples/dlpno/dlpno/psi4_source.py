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

from . import tensors as ten
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


class DFHelperSource:
    """A :class:`~dlpno.integrals.ThreeIndexSource` driving psi4's ``DFHelper``.

    Opt-in, never the default, and the measurements are why. This exists because
    ``DFHelper`` is the psi4 interface that has the two things ``ao_eri`` lacks -
    it threads and it screens - and because writing it down is the only way to
    say precisely what psi4 would have to add for the approach to be worth
    taking. On ethanol/cc-pVTZ, build plus half-transform:

        threads     1        10
        DFHelper    0.302 s  0.148 s
        ao_eri      0.266 s  0.269 s

    So it is 1.8x at ten threads and a small regression at one. That is the good
    news, and it is the smaller half of the story.

    ``DFHelper`` returns ``J^-1/2``-fitted B tensors. DLPNO fits per pair domain,
    not globally, so what this port needs is the RAW ``(Q|i u)`` and there is no
    unfitted mode on the Python surface - ``hold_met`` governs core storage, not
    whether the metric is applied. The only route back is multiplying by
    ``J^1/2``, which costs 15 ms and, more to the point, round-trips every number
    through a metric whose condition number is 6.4e6 at this basis. Measured, the
    round trip lands at 2.2e-13 relative, against a validation that requires
    reproducing canonical DF-MP2 to 1e-13, and the conditioning worsens with
    basis size. That is why :attr:`screening_threshold` never reports zero here
    however tight the Schwarz cutoff is set: this source is not exact, and the
    untruncated fixtures are right to refuse it.

    The demand is also ignored, and not by choice. ``get_tensor``'s three index
    sequences are ``[start, stop)`` slabs; an arbitrary index list is rejected
    outright, so a pair's scattered auxiliary and PAO domains cannot be
    expressed and the full tensor is built regardless.

    What psi4 would have to add for this to become the default is therefore
    specific: raw, unfitted three-index integrals, and index-list slicing. With
    those two, this class stops being a demonstration.
    """

    #: Floor on :attr:`screening_threshold`, from the ``J^1/2 . J^-1/2`` round
    #: trip. Measured at 2.2e-13 relative on ethanol/cc-pVTZ; reported an order
    #: of magnitude above that so a caller comparing against its own tolerance
    #: is not misled by a number quoted at its best case.
    METRIC_ROUNDTRIP_FLOOR = 1e-12

    def __init__(self, primary, aux, metric=None, schwarz_cutoff=0.0, nthreads=None,
                 memory_doubles=int(4e9 / 8)):
        self._primary = primary
        self._aux = aux
        self._metric = metric if metric is not None else aux_metric(aux)
        self._schwarz = float(schwarz_cutoff)
        self._nthreads = nthreads
        self._memory = memory_doubles
        self._spaces = None
        self._q_ia = None

    @property
    def screening_threshold(self) -> float:
        return max(self._schwarz, self.METRIC_ROUNDTRIP_FLOOR)

    def declare(self, spaces, demand) -> None:
        # demand is unused: see the class docstring on slab-only slicing.
        self._spaces = spaces

    def build(self) -> None:
        if self._spaces is None:
            raise RuntimeError("declare() before build()")
        C_lmo = np.ascontiguousarray(ten.view(self._spaces.C_lmo), dtype=np.float64)
        C_pao = np.ascontiguousarray(ten.view(self._spaces.C_pao), dtype=np.float64)

        helper = psi4.core.DFHelper(self._primary, self._aux)
        helper.set_memory(self._memory)
        if self._nthreads is not None:
            helper.set_nthreads(int(self._nthreads))
        if self._schwarz > 0.0:
            helper.set_schwarz_cutoff(self._schwarz)
        helper.set_method("STORE")
        helper.initialize()
        helper.add_space("lmo", psi4.core.Matrix.from_array(C_lmo))
        helper.add_space("pao", psi4.core.Matrix.from_array(C_pao))
        helper.add_transformation("Qiu", "lmo", "pao", "Qpq")
        helper.transform()

        naux = self._aux.nbf()
        fitted = np.asarray(helper.get_tensor("Qiu")).reshape(naux, C_lmo.shape[1], C_pao.shape[1])

        # Undo the fitting. Symmetric square root rather than a solve against
        # J^1/2: the metric is symmetric positive definite and already
        # eigendecomposed here, and the negative eigenvalues a near-singular
        # metric can produce numerically are clamped rather than allowed to
        # become NaNs under the square root.
        w, v = np.linalg.eigh(self._metric)
        j_half = (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T
        raw = np.einsum("PQ,Qiu->Piu", j_half, fitted, optimize=True)

        self._q_ia = ten.from_numpy("(Q|i u)", np.ascontiguousarray(raw))

    def q_ia(self):
        if self._q_ia is None:
            raise RuntimeError("build() before q_ia()")
        return self._q_ia

    def describe(self):
        shape = ten.shape(self._q_ia)
        return (f"(Q|iu) is {shape[0]} x {shape[1]} x {shape[2]} "
                f"(DFHelper, unfitted; threshold {self.screening_threshold:.0e})")
