#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""DLPNO truncation thresholds, mirroring psi4's DLPNO options.

Names and defaults follow psi4's ``read_options.cc`` and the ``PNO_CONVERGENCE``
presets applied in ``DLPNO::common_init``, so a run here can be lined up against
a psi4 run option for option.

One dataclass covers MP2 and the coupled-cluster methods, because psi4 has one
option block for both; what differs is the preset table, which
:meth:`Thresholds.preset` selects with its ``method`` argument. A CC field left
at its default in an MP2 calculation is simply never read.
"""

from dataclasses import dataclass, replace

__all__ = ["Thresholds"]


@dataclass(frozen=True)
class Thresholds:
    """Truncation parameters for a DLPNO calculation.

    The defaults are psi4's ``read_options.cc`` option defaults, which are the
    MP2 branch's ``PNO_CONVERGENCE NORMAL`` values for the fields the MP2 preset
    sets. :meth:`preset` is what applies a preset table on top.
    """

    # -- PNO truncation ----------------------------------------------------

    t_cut_pno: float = 1e-8
    t_cut_pno_diag_scale: float = 3e-2
    t_cut_pno_core_scale: float = 1e-2
    min_pnos: int = 5

    #: Minimum fraction of ``trace(D_ij)`` the kept PNOs must recover, psi4's
    #: ``T_CUT_TRACE``. Read only by the coupled-cluster PNO constructions:
    #: MP2 keeps a PNO on its occupation number alone.
    t_cut_trace: float = 0.999
    #: Minimum fraction of the pair energy the kept PNOs must recover, psi4's
    #: ``T_CUT_ENERGY``. CC only, like :attr:`t_cut_trace`.
    t_cut_energy: float = 0.997

    #: The same three criteria for the MP2-level PNOs a CC calculation builds
    #: during prescreening, before ``recompute_pnos`` tightens them. Looser
    #: nowhere: these are TIGHTER than the CC values, because the MP2 pair
    #: energies they produce are what the strong/weak split is decided on and
    #: an amplitude truncated here is never recovered.
    t_cut_pno_mp2: float = 1e-10
    t_cut_trace_mp2: float = 0.9999
    t_cut_energy_mp2: float = 0.999

    # -- LMO / PAO / auxiliary domain truncation ---------------------------

    t_cut_do: float = 1e-2
    t_cut_do_ij: float = 1e-5
    t_cut_do_pre: float = 3e-2
    t_cut_mkn: float = 1e-3
    t_cut_pre: float = 1e-6
    #: Strong/weak pair separation, psi4's ``T_CUT_PAIRS``. A pair whose
    #: converged LMP2 energy falls below this keeps that energy as its whole
    #: contribution and never enters the CC residual.
    t_cut_pairs: float = 1e-5
    #: Semicanonical-MP2 pair elimination, psi4's ``T_CUT_PAIRS_MP2``. Applied
    #: at the crude prescreening pass, and unlike the strong/weak split this one
    #: DELETES the pair: it leaves every domain and its energy is booked once,
    #: into ``de_lmp2_eliminated``.
    t_cut_pairs_mp2: float = 1e-6
    t_cut_clmo: float = 1e-4
    t_cut_cpao: float = 1e-4

    #: Shell-pair screening on the AO integrals a three-index producer builds,
    #: psi4's ``DLPNO_AO_INTS_TOL``. Unlike everything else here this one is a
    #: screening rather than an approximation: the error goes to zero with it,
    #: and only a producer that builds its own integrals reads it at all.
    ao_ints_tol: float = 1e-10

    # -- triples -----------------------------------------------------------

    #: TNO occupation cutoff, psi4's ``T_CUT_TNO``.
    t_cut_tno: float = 1e-9
    #: TNO cutoff for the triples prescreening pass, psi4's ``T_CUT_TNO_PRE``.
    t_cut_tno_pre: float = 1e-7
    #: Triplets whose prescreening energy falls below this are dropped, psi4's
    #: ``T_CUT_TRIPLES_WEAK``; their energy becomes ``de_lccsd_t_screened``.
    t_cut_triples_weak: float = 1e-8
    #: Scale on ``t_cut_tno`` for strong and weak triplets in iterative (T),
    #: psi4's ``T_CUT_TNO_{STRONG,WEAK}_SCALE``. Multipliers on the cutoff, so
    #: larger means a looser TNO space for that class.
    t_cut_tno_strong_scale: float = 10.0
    t_cut_tno_weak_scale: float = 100.0
    #: Domain thresholds for the triples passes, psi4's
    #: ``T_CUT_{DO,MKN}_TRIPLES`` and their ``_PRE`` variants.
    t_cut_do_triples: float = 1e-2
    t_cut_mkn_triples: float = 1e-2
    t_cut_do_triples_pre: float = 2e-2
    t_cut_mkn_triples_pre: float = 0.1
    #: Fock cutoff inside the iterative (T) coupling, psi4's ``F_CUT_T``.
    f_cut_t: float = 1e-3
    #: Convergence on the iterative (T) energy, psi4's ``T_CUT_ITER``.
    t_cut_iter: float = 1e-5
    #: Triplets keep at least this many TNOs, psi4's ``MIN_TNOS``.
    min_tnos: int = 9
    #: How many weak pairs a triplet may contain and still be formed, psi4's
    #: ``TRIPLES_MAX_WEAK_PAIRS``.
    triples_max_weak_pairs: int = 1
    #: Stop at the semicanonical (T0) energy, psi4's ``T0_APPROXIMATION``.
    t0_approximation: bool = False

    #: How much memory a phase that builds per-pair or per-triplet blocks may
    #: hold at once, in bytes.
    #:
    #: Covers the transient working set of a chunk, and the stores the
    #: iterative (T) keeps for every triplet at once, because the two are alive
    #: together and bounding only the first is how the triples came to page
    #: instead of refusing. ``compute_pno_integrals`` chunks against it for the
    #: same reason: the blocks a pair is BUILT from are an order larger than
    #: the blocks it produces, so the phase peaked at 10.9 GiB while reporting
    #: 0.6 GiB of stores.
    #:
    #: No psi4 analogue as a dial, and the difference is the point: psi4 spills
    #: W, V and the amplitudes to disk when they do not fit, which design
    #: decision 10 puts out of scope here - so the port reports its measured
    #: requirement and refuses instead. That also means this must NOT be set
    #: from psi4's ``--memory`` grant, which psi4 can honour only because it
    #: spills; a grant larger than the machine has authorises an allocation
    #: that pages rather than one that refuses.
    #:
    #: Named for what it bounds rather than for who asked first: it was
    #: ``t0_chunk_memory``, then ``triples_memory``, and each name went stale
    #: the moment another phase needed the same bound.
    in_core_memory: int = 2 * 2 ** 30

    # -- linear algebra cutoffs --------------------------------------------

    s_cut: float = 1e-8
    f_cut: float = 1e-5

    #: How many PNO-count buckets the pair blocks are padded into, or None to
    #: choose per machine. Blocks must share a shape to batch, but padding
    #: everything to the global maximum wastes elements; a handful of buckets
    #: keeps the batches large while cutting most of the waste. 1 reproduces a
    #: single padded store.
    #:
    #: None is the default because the right count is not a property of the
    #: molecule. More buckets is cheaper per element and dearer per batched
    #: call, and what a call costs is what entering an OpenMP region costs on
    #: this machine at this thread count - zero serial, tens of microseconds on
    #: ten threads. Measured on ethanol/cc-pVTZ the best count runs 12, 8, 8, 4
    #: at 1, 2, 4 and 10 threads. See :meth:`~dlpno.layout.PairLayout.choose`.
    #:
    #: Set an integer to pin it, which is what the calibration sweeps do.
    n_buckets: int | None = None

    #: Largest bucket count the automatic chooser will consider. Past this the
    #: shape classes outnumber the pairs and every batch is a handful of members.
    max_buckets: int = 16

    #: How many groups each PNO bucket's pairs are split into by the total width
    #: of their concatenated couplings. The residual folds all of a pair's
    #: couplings in one GEMM, and batching those across pairs needs them to agree
    #: on that width, so it is padded to the group's widest. One group per bucket
    #: is the fewest calls and the most padding (1.42x the flops at a six-monomer
    #: water chain); four costs 1.10x for sixteen calls against 32948 in the
    #: per-coupling form this replaced.
    n_width_groups: int = 4

    # -- iterative solver --------------------------------------------------

    maxiter: int = 50
    e_convergence: float = 1e-10
    r_convergence: float = 1e-8
    diis_max_vecs: int = 6

    #: What the LMP2 solve inside a CC calculation tightens its convergence by,
    #: psi4's hard-coded ``0.01 *`` on both criteria in
    #: ``DLPNOCCSD::pno_lmp2_iterations``. Those amplitudes are not an answer,
    #: they are the input to the PNO rebuild and the strong/weak split, so they
    #: are converged past what the energy alone would need.
    mp2_in_cc_convergence_scale: float = 1e-2

    # -- presets -----------------------------------------------------------

    #: psi4's ``PNO_CONVERGENCE`` tables. The MP2 branch sets three fields; the
    #: coupled-cluster branch sets nine, and its PNO cutoff is two orders of
    #: magnitude looser because CC then rebuilds the PNOs from converged LMP2
    #: amplitudes rather than semicanonical ones.
    _PRESETS = {
        "mp2": {
            "LOOSE": dict(t_cut_pno=1e-7, t_cut_do=2e-2, t_cut_mkn=1e-3),
            "NORMAL": dict(t_cut_pno=1e-8, t_cut_do=1e-2, t_cut_mkn=1e-3),
            "TIGHT": dict(t_cut_pno=1e-9, t_cut_do=5e-3, t_cut_mkn=1e-3),
            "VERY_TIGHT": dict(t_cut_pno=1e-10, t_cut_do=5e-3, t_cut_mkn=1e-4),
        },
        "cc": {
            "LOOSE": dict(t_cut_pno=1e-6, t_cut_trace=0.9, t_cut_energy=0.9,
                          t_cut_pairs=1e-3, t_cut_pno_mp2=1e-8,
                          t_cut_trace_mp2=0.99, t_cut_energy_mp2=0.99,
                          t_cut_do=2e-2, t_cut_mkn=1e-3),
            "NORMAL": dict(t_cut_pno=3.33e-7, t_cut_trace=0.99, t_cut_energy=0.99,
                           t_cut_pairs=1e-4, t_cut_pno_mp2=3.33e-9,
                           t_cut_trace_mp2=0.999, t_cut_energy_mp2=0.997,
                           t_cut_do=1e-2, t_cut_mkn=1e-3),
            "TIGHT": dict(t_cut_pno=1e-7, t_cut_trace=0.999, t_cut_energy=0.997,
                          t_cut_pairs=1e-5, t_cut_pno_mp2=1e-9,
                          t_cut_trace_mp2=0.9999, t_cut_energy_mp2=0.999,
                          t_cut_do=5e-3, t_cut_mkn=1e-3),
            "VERY_TIGHT": dict(t_cut_pno=1e-8, t_cut_trace=0.999, t_cut_energy=0.997,
                               t_cut_pairs=1e-6, t_cut_pno_mp2=1e-10,
                               t_cut_trace_mp2=0.9999, t_cut_energy_mp2=0.999,
                               t_cut_do=5e-3, t_cut_mkn=1e-4),
        },
    }

    @classmethod
    def preset(cls, name="NORMAL", method="mp2", **overrides):
        """psi4's ``PNO_CONVERGENCE`` presets, for one method branch.

        Args:
            name: ``LOOSE``, ``NORMAL``, ``TIGHT`` or ``VERY_TIGHT``.
            method: ``"mp2"`` or ``"cc"``. psi4 branches on ``algorithm_`` in
                ``common_init``, and the two tables disagree on nearly every
                field they share.
            **overrides: Applied last, and they also suppress the two derived
                rules below - psi4 gates those on ``has_changed()``, and an
                explicit value here is the same statement.
        """
        table = cls._PRESETS.get(method)
        if table is None:
            raise ValueError(f"unknown method {method!r}; expected 'mp2' or 'cc'")
        key = name.upper()
        if key not in table:
            raise ValueError(f"unknown PNO_CONVERGENCE preset {name!r}")

        values = {**table[key], **overrides}
        default = cls()
        t_cut_pairs = values.get("t_cut_pairs", default.t_cut_pairs)

        # psi4's two derived rules, each applied only where the user has not
        # spoken. The first is in the CC branch of common_init; the second sits
        # outside both branches and so applies to MP2 as well, where it is a
        # no-op at the default t_cut_pairs.
        if method == "cc" and "t_cut_pre" not in overrides:
            values["t_cut_pre"] = min(default.t_cut_pre, 0.01 * t_cut_pairs)
        if "t_cut_pairs_mp2" not in overrides:
            values["t_cut_pairs_mp2"] = min(1e-6, 0.1 * t_cut_pairs)

        return replace(default, **values)

    @classmethod
    def untruncated(cls, **overrides):
        """Every truncation switched off.

        With full PAO and auxiliary domains this makes local MP2 exactly
        equivalent to canonical DF-MP2, and T1-dressed local CCSD equivalent to
        canonical DF-CCSD, which is how the port is validated.

        The sign conventions differ by criterion and are not interchangeable,
        because what makes a threshold vacuous depends on the comparison it
        appears in.

        * A ``>`` test against an absolute value needs a NEGATIVE threshold.
          Zero would still drop an orbital whose overlap or population happened
          to vanish exactly. Every domain cutoff is one of these.
        * A ``>=`` test - the PNO and TNO occupation criteria - is already
          vacuous at zero, since an absolute value is never below it.
        * A *recovery fraction* keeps orbitals while the running total is BELOW
          the threshold, so zero switches the criterion OFF rather than on. That
          is what is wanted: with the occupation test already keeping everything,
          the trace and energy criteria must not be the reason anything survives,
          or a rounding difference in a partial sum would move a PNO count.
        * A threshold gating a product against its square goes to zero, which
          already admits everything.
        """
        return replace(
            cls(),
            t_cut_pno=0.0,
            t_cut_pno_diag_scale=1.0,
            t_cut_pno_core_scale=1.0,
            min_pnos=0,
            f_cut=0.0,
            # Recovery fractions: see the docstring. Zero switches these off.
            t_cut_trace=0.0,
            t_cut_energy=0.0,
            t_cut_trace_mp2=0.0,
            t_cut_energy_mp2=0.0,
            t_cut_pno_mp2=0.0,
            # Negative rather than zero: the domain tests are on absolute
            # values, so a zero threshold would still drop an orbital whose
            # overlap or population happened to vanish exactly.
            t_cut_do=-1.0,
            t_cut_do_ij=-1.0,
            t_cut_do_pre=-1.0,
            t_cut_mkn=-1.0,
            t_cut_pre=-1.0,
            t_cut_clmo=-1.0,
            t_cut_cpao=-1.0,
            # Every pair strong, nothing eliminated.
            t_cut_pairs=-1.0,
            t_cut_pairs_mp2=-1.0,
            # Triples: every TNO kept, every triplet formed.
            t_cut_tno=0.0,
            t_cut_tno_pre=0.0,
            t_cut_triples_weak=-1.0,
            t_cut_do_triples=-1.0,
            t_cut_mkn_triples=-1.0,
            t_cut_do_triples_pre=-1.0,
            t_cut_mkn_triples_pre=-1.0,
            f_cut_t=0.0,
            min_tnos=0,
            # A triplet's weak-pair count cannot exceed three, and with
            # t_cut_pairs negative there are none anyway.
            triples_max_weak_pairs=3,
            # Zero rather than negative, unlike the domain thresholds: this one
            # gates a product against tol^2, so zero already admits everything.
            ao_ints_tol=0.0,
            **overrides,
        )
