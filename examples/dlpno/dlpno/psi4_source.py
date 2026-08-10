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
from .thresholds import Thresholds

__all__ = ["from_psi4", "raw_three_index", "aux_metric", "localize", "ao_dipole",
           "grid_block_provider", "DFHelperSource", "ScreenedQiaSource"]


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
    """The auxiliary Coulomb metric ``(P|Q)``.

    Through ``FittingMetric`` rather than ``MintsHelper::ao_eri``, which builds
    the same integrals: at a six-monomer water chain 1.6 ms against 6.6 ms,
    because ``ao_eri`` neither threads nor exploits the metric's symmetry. The
    two agree to 5e-14, which is round-off on a matrix with O(1) elements, and
    it is the routine psi4's own DLPNO uses for this block - so this moves the
    port's number towards psi4's rather than away from it.

    Not inverted and not powered: DLPNO solves the metric per pair domain, so
    what is wanted is ``(P|Q)`` itself.
    """
    metric = psi4.core.FittingMetric(aux, True)
    metric.form_fitting_metric()
    return np.ascontiguousarray(np.asarray(metric.get_metric()))


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

    The grid itself is built on the first call rather than here, so its cost
    lands in the phase that reads it. It is a differential-overlap input, and
    building it while assembling the reference charged 10 ms of it to the
    three-index integrals at a six-monomer water chain - which mattered not
    because 10 ms is large but because that row is the one the DF work is
    judged by.
    """
    mol = basis.molecule()
    grid = None

    def build_grid():
        psi4.set_options({"DFT_BASIS_TOLERANCE": basis_tolerance,
                          "DFT_BS_RADIUS_ALPHA": 1.0,
                          "DFT_PRUNING_ALPHA": 1.0,
                          "DFT_BLOCK_MAX_RADIUS": 3.0,
                          "DFT_WEIGHTS_TOLERANCE": 1e-15})
        return psi4.core.DFTGrid.build(
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
        nonlocal grid
        if grid is None:
            grid = build_grid()
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


def from_psi4(wfn, aux=None, localization="BOYS", freeze_core=False,
              integrals="dfhelper", schwarz_cutoff=0.0, nthreads=None,
              thresholds=None):
    """Assemble a :class:`Reference` from a converged psi4 wavefunction.

    ``aux`` defaults to the RIFIT auxiliary basis for the primary basis, which
    is what psi4's DLPNO uses for the correlation integrals.

    ``integrals`` chooses where ``(Q|i u)`` will come from:

    ``"dfhelper"``
        Attach a :class:`DFHelperSource` and do not build ``(Q|mn)`` at all.
        The default, because it is exact and threads: nothing downstream reads
        the dense tensor, so materializing 102 MB of it at ethanol/cc-pVTZ was
        only ever a way of getting at its half-transform.
    ``"screened"``
        Attach a :class:`ScreenedQiaSource`, which builds only the blocks the
        solver declares it will read. The fastest, and the only one that is
        domain-restricted rather than exact; see that class on what it gives up.
    ``"dense"``
        Build the dense ``(Q|mn)`` with ``MintsHelper::ao_eri``. Slower and
        unthreaded, and required for :func:`dlpno.reference_io.save_reference`,
        which freezes buffers and cannot freeze a live source.

    ``thresholds`` supplies the screened source's three tolerances and is
    ignored by the other two. It is the solver's own
    :class:`~dlpno.thresholds.Thresholds`, passed here rather than read later
    because a producer has to be configured before it is declared to - which is
    also why a caller that means to compare against psi4 should hand over the
    same object it hands the solver, not a fresh default.
    """
    if integrals not in ("dfhelper", "dense", "screened"):
        raise ValueError(
            f"from_psi4: integrals must be 'dfhelper', 'screened' or 'dense', got {integrals!r}")
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
    S = np.asarray(wfn.S())

    if integrals == "dense":
        source = None
    elif integrals == "dfhelper":
        source = DFHelperSource(primary, aux, schwarz_cutoff=schwarz_cutoff, nthreads=nthreads)
    else:
        cut = thresholds if thresholds is not None else Thresholds()
        source = ScreenedQiaSource(primary, aux, S,
                                   t_cut_clmo=cut.t_cut_clmo, t_cut_cpao=cut.t_cut_cpao,
                                   ao_ints_tol=cut.ao_ints_tol, nthreads=nthreads)

    ref = Reference(
        S=S,
        F=np.asarray(wfn.Fa()),
        C_occ=C_occ,
        C_lmo=localize(primary, C_act_occ, localization),
        eri_3index=raw_three_index(primary, aux) if integrals == "dense" else None,
        integral_source=source,
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
        # DFHelper defaults to ONE thread whatever psi4 has been told, so not
        # setting this silently gives up the threading that is the whole reason
        # to be here: the AO build measured 0.272 s at ten threads that way,
        # against 0.067 s with the team it should have had.
        threads = self._nthreads if self._nthreads is not None else psi4.core.get_num_threads()
        helper.set_nthreads(int(threads))
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


class ScreenedQiaSource:
    """A :class:`~dlpno.integrals.ThreeIndexSource` that builds only what is read.

    The one source that consults the declared demand, and the only one that can:
    it drives psi4's ``LocalQiaBuilder``, which takes the per-auxiliary-atom
    LMO and PAO lists and runs a screened shell-triplet loop restricted to the
    basis functions those orbitals live on, transforming each auxiliary shell's
    small block in place. Nothing outside the demand is built, and neither are
    the AO integrals it does not imply.

    That last clause is the whole point, and it is why this exists rather than a
    cleverer transform. ``DFHelperSource`` builds every Schwarz-surviving AO
    integral and then transforms them: at a six-monomer water chain the two
    transform contractions are 18 ms of a 164 ms phase, so transforming less
    could not have paid. Building less could, and only psi4 can build less,
    because the AO integral machinery is there.

    The price is that this source is *not* exact at its default settings, and
    the inexactness is not the Schwarz kind. Two screenings are in force:

    ``ao_ints_tol``
        Drops AO integrals that are numerically negligible. Controlled - the
        error goes to zero with the tolerance - and this is the same class of
        approximation ``DFHelperSource`` makes with its Schwarz cutoff.
    ``t_cut_clmo`` / ``t_cut_cpao``
        Restrict each atom's transform to the basis functions its requested
        orbitals actually live on. This drops *transform contributions*, not
        negligible integrals, and it stays wrong at exact arithmetic. The
        Boughton-Pulay refit inside the builder mitigates it by re-solving the
        LMO coefficients on the restricted domain, which is what makes the
        default 1e-4 usable; it does not remove it.

    psi4's own DLPNO makes exactly this approximation with exactly these
    defaults, so a run against psi4 at NORMAL thresholds compares like with
    like. A run that has to be exact should set all three to their untruncated
    values, at which point the domains become the full basis and this reproduces
    :class:`~dlpno.integrals.DenseSource` to round-off.

    ``LocalQiaBuilder`` is not in released psi4. It was added for this, and its
    absence is reported as a clear error rather than a missing attribute.
    """

    def __init__(self, primary, aux, S, t_cut_clmo=1e-4, t_cut_cpao=1e-4,
                 ao_ints_tol=1e-10, bp_refit=True, nthreads=None):
        self._primary = primary
        self._aux = aux
        self._S = np.ascontiguousarray(np.asarray(S, dtype=np.float64))
        self._t_cut_clmo = float(t_cut_clmo)
        self._t_cut_cpao = float(t_cut_cpao)
        self._ao_ints_tol = float(ao_ints_tol)
        self._bp_refit = bool(bp_refit)
        self._nthreads = nthreads

        self._builder = None
        self._spaces = None
        self._demand = None
        self._q_ia = None
        self._written = 0

    @property
    def screening_threshold(self) -> float:
        """The largest tolerance in force, and it means DOMAIN-RESTRICTED.

        Deliberately not the Schwarz-shaped promise ``DFHelperSource`` makes.
        A nonzero value here says the result differs from the exact transform
        by more than integral screening: the coefficient tolerances drop
        transform contributions outright, so an untruncated calculation is
        entitled to use this source only when all three are switched off.
        """
        return max(self._ao_ints_tol, self._t_cut_clmo, self._t_cut_cpao, 0.0)

    def declare(self, spaces, demand) -> None:
        if not hasattr(psi4.core, "LocalQiaBuilder"):
            raise RuntimeError(
                "this psi4 has no LocalQiaBuilder, which ScreenedQiaSource needs to build "
                "three-index integrals per auxiliary atom; use integrals='dfhelper' for the exact "
                "psi4-backed source or dlpno.integrals.DenseSource for the reference one")
        if not demand.aux_atom_to_lmos or not demand.aux_atom_to_paos:
            raise RuntimeError(
                "ScreenedQiaSource needs the per-auxiliary-atom demand, which the solver fills in "
                "from its pair list; a Demand carrying only the flat domain lists cannot say which "
                "orbitals are read against which atom")

        self._spaces = spaces
        self._demand = demand

        builder = psi4.core.LocalQiaBuilder(self._primary, self._aux)
        # Same trap DFHelper has: a builder that is not told the thread count
        # would default to the process one, and psi4's process thread count is
        # whatever the last set_num_threads left it at. Say it explicitly.
        threads = self._nthreads if self._nthreads is not None else psi4.core.get_num_threads()
        builder.set_nthreads(int(threads))
        builder.set_ints_tolerance(self._ao_ints_tol)
        builder.set_coef_tolerance_lmo(self._t_cut_clmo)
        builder.set_coef_tolerance_pao(self._t_cut_cpao)
        builder.set_bp_refit(self._bp_refit)
        builder.set_domains([list(map(int, lmos)) for lmos in demand.aux_atom_to_lmos],
                            [list(map(int, paos)) for paos in demand.aux_atom_to_paos])
        self._builder = builder

    def build(self) -> None:
        if self._builder is None:
            raise RuntimeError("declare() before build()")

        C_lmo = np.ascontiguousarray(ten.view(self._spaces.C_lmo))
        C_pao = np.ascontiguousarray(ten.view(self._spaces.C_pao))
        blocks = self._builder.compute(psi4.core.Matrix.from_array(C_lmo),
                                       psi4.core.Matrix.from_array(C_pao),
                                       psi4.core.Matrix.from_array(self._S))
        ranges = self._builder.atom_aux_ranges()

        naux = self._aux.nbf()
        naocc, npao = C_lmo.shape[1], C_pao.shape[1]
        # Zeroed, and the zeros are load-bearing: everything outside the
        # declared blocks is never read (see DLPNOBase._aux_atom_demand), so it
        # is left as it was allocated rather than filled with anything
        # meaningful. A consumer that started reading outside the demand would
        # get silent zeros, which is the failure mode the differential gate in
        # check_integral_sources.py exists to catch.
        self._q_ia = ten.zeros("(Q|i u)", [naux, naocc, npao])
        view = ten.view(self._q_ia)

        written = 0
        for atom, block in enumerate(blocks):
            if block is None:
                continue
            lo, hi = ranges[atom]
            lmos = np.asarray(self._demand.aux_atom_to_lmos[atom], dtype=np.intp)
            paos = np.asarray(self._demand.aux_atom_to_paos[atom], dtype=np.intp)
            blk = np.asarray(block).reshape(hi - lo, lmos.size, paos.size)
            # An axis whose list covers the whole space is a slice, not a
            # gather: the lists are sorted and distinct, so full length is the
            # whole test. Worth separating because a scattered assignment into
            # a column-major tensor walks the slowest axis innermost, and at a
            # six-monomer water chain most atoms read every LMO and every PAO -
            # the ones that do not are the far ends of the chain.
            target = view[lo:hi]
            if lmos.size == naocc and paos.size == npao:
                target[...] = blk
            elif paos.size == npao:
                target[:, lmos, :] = blk
            elif lmos.size == naocc:
                target[:, :, paos] = blk
            else:
                target[np.ix_(np.arange(hi - lo), lmos, paos)] = blk
            written += blk.size
        self._written = written

    def q_ia(self):
        if self._q_ia is None:
            raise RuntimeError("build() before q_ia()")
        return self._q_ia

    def declared_mask(self):
        """Boolean array marking the entries of ``q_ia`` this source wrote.

        For validation rather than for the solver. Comparing this source
        against an exact one elementwise would compare the untouched zeros too,
        which are outside the demand and are never read; the comparison that
        means something is the one restricted to here.
        """
        if self._q_ia is None:
            raise RuntimeError("build() before declared_mask()")
        mask = np.zeros(ten.shape(self._q_ia), dtype=bool)
        for atom, (lo, hi) in enumerate(self._builder.atom_aux_ranges()):
            lmos = self._demand.aux_atom_to_lmos[atom]
            paos = self._demand.aux_atom_to_paos[atom]
            if lo == hi or not lmos or not paos:
                continue
            mask[lo:hi][np.ix_(np.arange(hi - lo),
                               np.asarray(lmos, dtype=np.intp),
                               np.asarray(paos, dtype=np.intp))] = True
        return mask

    def describe(self):
        shape = ten.shape(self._q_ia)
        total = shape[0] * shape[1] * shape[2]
        computed, offered = self._builder.last_shell_pair_counts()
        return (f"(Q|iu) is {shape[0]} x {shape[1]} x {shape[2]}, "
                f"{100.0 * self._written / max(total, 1):.1f}% written "
                f"(domains T_CUT_CLMO {self._t_cut_clmo:.0e} / T_CUT_CPAO {self._t_cut_cpao:.0e}, "
                f"AO ints {self._ao_ints_tol:.0e} kept {100.0 * computed / max(offered, 1):.0f}% "
                f"of {offered} shell pairs)")
