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

from . import integrals
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

    Exact, and faster than the dense path at every thread count. It builds the
    raw ``(Q|mn)`` through ``DFHelper::get_AO_tensor`` - threaded and Schwarz
    screened, where ``MintsHelper::ao_eri`` is neither - and then runs the same
    two contractions :class:`~dlpno.integrals.DenseSource` does, by handing the
    integrals to it. The transform is shared rather than reimplemented, so the
    two sources agree by construction rather than by testing.

    Measured on ethanol/cc-pVTZ, building the AO integrals:

        threads          1        10
        ao_eri           0.246 s  0.236 s
        get_AO_tensor    0.188 s  0.067 s
                         1.31x    3.51x

    ``ao_eri`` does not thread at all, which is what the middle row is really
    saying, and it is why this phase used to be flat from one core to ten.

    Exactness survives because no fitting metric is involved. ``DFHelper``'s
    transformations return ``J^-1/2`` fitted B tensors, which are the wrong
    quantity here - DLPNO fits per pair domain rather than globally - and
    recovering the raw integrals from them means multiplying by ``J^1/2``,
    round-tripping every element through a metric conditioned at 6.4e6. Going
    to the AO integrals directly sidesteps that entirely: with screening off
    this reproduces ``ao_eri`` bit for bit.

    ``get_AO_tensor`` is not in released psi4. It was added for this, and its
    absence is reported as a clear error rather than a missing attribute.
    """

    def __init__(self, primary, aux, schwarz_cutoff=0.0, nthreads=None,
                 memory_doubles=int(6e9 / 8)):
        self._primary = primary
        self._aux = aux
        self._schwarz = float(schwarz_cutoff)
        self._nthreads = nthreads
        self._memory = memory_doubles
        self._inner = None

    @property
    def screening_threshold(self) -> float:
        """The Schwarz cutoff, and nothing else.

        Zero means exact, and means it literally: the integrals match
        ``ao_eri`` to the last bit, so an untruncated calculation is entitled
        to use this source.
        """
        return self._schwarz

    def declare(self, spaces, demand) -> None:
        # The demand is not consulted. DFHelper's AO integrals are screened but
        # not domain-restricted, and get_tensor's slicing takes [start, stop)
        # slabs rather than index lists, so a pair's scattered domains cannot be
        # expressed. Building per domain is the next thing psi4 would have to
        # offer; see the module docstring on raw_three_index.
        helper = psi4.core.DFHelper(self._primary, self._aux)
        helper.set_memory(self._memory)
        if self._nthreads is not None:
            helper.set_nthreads(int(self._nthreads))
        # Always set it, including to zero. Skipping the call leaves psi4's own
        # default cutoff in force, which would make screening_threshold report
        # exactness this source was not actually delivering.
        helper.set_schwarz_cutoff(self._schwarz)
        # DIRECT rather than STORE: STORE contracts the fitting metric into the
        # AO integrals as it builds them, and raw is the whole point.
        helper.set_method("DIRECT")
        helper.set_AO_core(True)
        helper.initialize()

        if not hasattr(helper, "get_AO_tensor"):
            raise RuntimeError(
                "this psi4 has no DFHelper.get_AO_tensor, which DFHelperSource needs for raw "
                "three-index integrals; use dlpno.integrals.DenseSource instead")

        naux, nbf = self._aux.nbf(), self._primary.nbf()
        qmn = np.asarray(helper.get_AO_tensor()).reshape(naux, nbf, nbf)
        # Same transform as the dense source, because it IS the dense source:
        # only where (Q|mn) came from differs.
        self._inner = integrals.DenseSource(qmn)
        self._inner.declare(spaces, demand)

    def build(self) -> None:
        if self._inner is None:
            raise RuntimeError("declare() before build()")
        self._inner.build()

    def q_ia(self):
        if self._inner is None:
            raise RuntimeError("build() before q_ia()")
        return self._inner.q_ia()

    def describe(self):
        shape = ten.shape(self._inner.q_ia())
        exact = "exact" if self._schwarz == 0.0 else f"schwarz {self._schwarz:.0e}"
        return (f"(Q|iu) is {shape[0]} x {shape[1]} x {shape[2]} "
                f"(DFHelper raw AO, {exact})")
