#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""The DLPNO-MP2 phases, registered as ``einsums.stages`` stages.

:meth:`DLPNOMP2.compute_energy` was already a nine-line sequence of phases, so
the stage boundaries were not invented here; they were named. What the
framework adds is a measurement: :meth:`Session.report` prints what each phase
cost, which is the number the decision to write C++ has to be made against.
The DLPNO port's own history is the argument for measuring rather than
modelling it, since a capture-overhead model projected 12% for a change that
measured about 2.2x.

Two things about this file are deliberately unfinished, and both are visible in
the report rather than buried here.

**Every phase is an eager stage.** Each one builds and executes its own graphs
already (``cg.Graph`` plus ``cg.capture`` inside ``base.py`` and ``mp2.py``),
so from the session's point of view it is a phase that must run to completion
before the next one starts. Declaring that honestly means nine ``SPLIT`` rows,
each one a graph boundary no pass can cross. That is not a loss against the
status quo, which has the same nine boundaries and no way to see them; it turns
an invisible property into a line of output. Converting a phase to capture into
the session's graph is then a visible improvement: one fewer SPLIT.

**Most phases do not state a contract.** They are methods, so a contract means
splitting off the ``self`` fields each one touches, and the field counts say
that is not uniform work: five phases touch fewer than ten fields while
``lmp2_iterations`` touches 45. A phase states its contract when it is
promoted, and ``contract=False`` reports the debt until then. Three have been:
``compute_pno_overlaps`` (27 fields narrowed to seven parameters),
``transform_pnos`` (32 narrowed to fifteen) and the coupled-cluster
``compute_pno_integrals`` (19 narrowed to seventeen), each split into a plan
half that stays Python and a contracted numerics stage declared below, with C++
backends under ``cpp/``. ``python -m einsums.stages extract`` mechanizes the
analysis for the next one.
"""

import functools

from einsums.stages import TensorD, stage

from .cc_integrals import pno_integral_blocks as _pno_integral_blocks
from .contracts import CouplingPlan, PnoIntegralBlocks, PnoOverlaps, PnoTransform
from .mp2 import DLPNOMP2
from .pno_overlaps import compute_pno_overlaps as _compute_pno_overlaps
from .pno_xform import transform_pnos as _transform_pnos

__all__ = ["PHASES", "run_phases", "compute_pno_overlaps", "transform_pnos",
           "compute_pno_integrals"]

# In dependency order, which is the order compute_energy called them in.
#
# compute_pno_overlaps and pno_transform are absent as single phases because
# each is a planning half that stays Python followed by a contracted numerics
# stage declared below: plan_pno_couplings + compute_pno_overlaps (M3's
# promotion), and plan_pno_transform + transform_pnos (M6's).
PHASES = [
    "setup_orbitals",
    "compute_doi",
    "prep_sparsity",
    "compute_metric",
    "compute_qia",
    "precompute_fits",
    "plan_pno_transform",
    "transform_pnos",
    "plan_pno_couplings",
    "compute_pno_overlaps",
    "lmp2_iterations",
]

# Phases still carrying contract debt: methods reading fields off self.
# The plan_* phases are among them and are not promotion targets either - they
# are index arithmetic, and nothing measured says they are hot.
CONTRACTED = ("compute_pno_overlaps", "transform_pnos", "compute_pno_integrals")
UNCONTRACTED = [n for n in PHASES if n not in CONTRACTED]


def _register(name: str):
    """Register ``DLPNOMP2.<name>`` as an eager, not-yet-contracted stage."""
    method = getattr(DLPNOMP2, name)

    @stage(eager=True, contract=False)
    @functools.wraps(method)
    def run(mp2, **kwargs):
        return method(mp2, **kwargs)

    return run


_stages = {name: _register(name) for name in UNCONTRACTED}


@stage(eager=True)
def compute_pno_overlaps(
    X_pno: list[TensorD],
    S_pao: TensorD,
    lmopair_to_paos: list[list[int]],
    n_pno: list[int],
    bucket_of: list[int],
    bucket_dims: list[int],
    plan: CouplingPlan,
) -> PnoOverlaps:
    """Overlaps between every coupled pair of PNO bases, scaled by the Fock element.

    The first DLPNO phase to state a contract. Seven parameters, against the 27
    fields of ``self`` the phase touched before it was split - which is what
    "cut where the interface is narrow" means in practice, and why the split
    had to come first.

    Runs to completion and returns values; it does not emit into an ambient
    capture. That is a structural property rather than an unfinished one: the
    phase runs two graphs with a host-side scaling between them, and the second
    graph's inputs are the first graph's outputs after that scaling. There is
    no single graph to emit into until the scaling itself becomes a captured
    op, at which point the stage stops being ``eager`` and the report loses a
    SPLIT. So this stage is not what proves the shared-graph contract; a
    separate, deliberately minimal stage does that.

    Args:
        X_pno: Per pair, the PNO transform on its own domain.
        S_pao: PAO overlap matrix, ``(npao, npao)``.
        lmopair_to_paos: Per pair, the PAO indices its domain covers.
        n_pno: Per pair, its PNO count; zero means the pair is dead.
        bucket_of: Per pair, which padding bucket it is stored at.
        bucket_dims: Per bucket, the padded block dimension.
        plan: The layout, from ``plan_pno_couplings``.
    """
    return _compute_pno_overlaps(
        X_pno, S_pao, lmopair_to_paos, n_pno, bucket_of, bucket_dims, plan
    )


@stage(eager=True)
def transform_pnos(
    q_ia: TensorD,
    fits: list[TensorD],
    fit_of: list[int],
    fit_pos: list[int],
    dom_X: list[TensorD],
    dom_e: list[TensorD],
    dom_F: list[TensorD],
    dom_of: list[int],
    ribfs: list[list[int]],
    paos: list[list[int]],
    lmo_j: list[int],
    shift: list[float],
    pno_scale: list[float],
    min_pnos: int,
    t_cut_pno: float,
    t_cut_trace: float,
    t_cut_energy: float,
) -> PnoTransform:
    """Build each upper pair's truncated, canonical PNO basis.

    The second DLPNO phase to state a contract, and the widest so far:
    fifteen parameters against the 32 fields of ``self`` the phase touched
    before ``plan_pno_transform`` was split away. Everything domain-shaped
    arrives deduplicated (one tensor per distinct domain plus per-pair
    ordinals), which is the planner handing over the memo caches' sharing
    rather than one copy per pair.

    Promoted on the 2026-08-08 re-profile: capture EMISSION is 45-56% of the
    phase - ~1,630 nodes across three graphs at ~8 us of Python per node -
    which is the per-node cost a language change removes and nothing else
    does. See DESIGN-hybrid-framework.md, M6 step 3.

    Args:
        q_ia: Three-index integrals ``(Q | i a)``, the full tensor.
        fits: Per distinct domain block, ``J^-1 (Q | i u)``, rank 3.
        fit_of: Per upper pair, its index into ``fits``.
        fit_pos: Per upper pair, LMO ``i``'s slot on the fit's middle axis.
        dom_X: Per distinct PAO domain, the orthocanonicalizer.
        dom_e: Per distinct PAO domain, the canonical orbital energies.
        dom_F: Per distinct PAO domain, the canonical-basis Fock matrix.
        dom_of: Per upper pair, its index into the ``dom_*`` lists.
        ribfs: Per upper pair, the auxiliary indices of its fit domain.
        paos: Per upper pair, the PAO indices its domain covers.
        lmo_j: Per upper pair, LMO ``j`` for the ``(Q | j a)`` gather.
        shift: Per upper pair, ``F_ii + F_jj``.
        pno_scale: Per upper pair, the core-pair scaling on ``T_CUT_PNO``.
        min_pnos: Pairs keep at least this many PNOs, domain size permitting.
        t_cut_pno: The occupation-number cutoff.
        t_cut_trace: Minimum fraction of ``trace(D_ij)`` the kept PNOs recover.
            Zero switches the criterion off, which is the MP2 branch.
        t_cut_energy: Minimum fraction of the pair energy the kept PNOs
            recover. Zero switches it off, likewise.
    """
    return _transform_pnos(
        q_ia, fits, fit_of, fit_pos, dom_X, dom_e, dom_F, dom_of,
        ribfs, paos, lmo_j, shift, pno_scale, min_pnos, t_cut_pno,
        t_cut_trace, t_cut_energy,
    )


@stage(eager=True)
def compute_pno_integrals(
    q_ij: TensorD,
    q_ia: TensorD,
    q_ab: TensorD,
    metric: TensorD,
    X_pno: list[TensorD],
    i_lmo: list[int],
    j_lmo: list[int],
    n_pno: list[int],
    strong: list[bool],
    ribfs: list[list[int]],
    paos: list[list[int]],
    lmos: list[list[int]],
    extended: list[list[int]],
    rot_X: list[TensorD],
    rot_paos: list[list[int]],
    nb_ij: list[list[int]],
    nb_ji: list[list[int]],
) -> PnoIntegralBlocks:
    """Every PNO-basis integral block one chunk of pairs needs.

    The first COUPLED-CLUSTER phase to state a contract, and the third stage
    overall: seventeen parameters against the nineteen ``self`` attributes the
    phase measured before ``_plan_chunk`` was split away, two of which were not
    state at all (a print helper and a method). Design decision 6 asked for the
    phase to be kept promotable and ``stage_state_report.py`` is how that was
    checked; this is the promotion it was kept narrow for.

    Promoted because its measured shape is the promotion signature exactly:
    parity with psi4 at one thread, several times adrift at ten, on a long
    per-pair chain of fits, GEMMs and gathers that psi4 threads over pairs.
    Python cannot drive that loop concurrently - against the OpenMP-built
    OpenBLAS a caller-created thread pool returns silently wrong numbers (trap
    7) - so the C++ backend is the only place the per-pair chains can overlap.

    One chunk of pairs at a time, because the blocks a pair is BUILT from are an
    order larger than the blocks it produces and the whole set does not fit in
    core. The chunking is the caller's (:func:`dlpno.cc_integrals._chunks`),
    which is what keeps it a memory decision rather than a numerical one: no
    pair reads another pair's raw blocks, so any split gives bit-identical
    integrals.

    Args:
        q_ij: Three-index integrals ``(Q | i j)``, the full tensor.
        q_ia: Three-index integrals ``(Q | i u)``, the full tensor.
        q_ab: Three-index integrals ``(Q | u v)``, the full tensor.
        metric: The auxiliary-basis metric ``(P | Q)``, the full matrix.
        X_pno: Per pair in the chunk, its own PAO domain -> PNO transform.
        i_lmo: Per pair, LMO ``i``.
        j_lmo: Per pair, LMO ``j``; never less than ``i``.
        n_pno: Per pair, its PNO count.
        strong: Per pair, whether it is a strong pair. The density-fitted
            factors and both non-projected families are built for strong pairs
            only, exactly as psi4 does, because only their residual reads them.
        ribfs: Per pair, the auxiliary indices of its fit domain.
        paos: Per pair, the PAO indices its own domain covers.
        lmos: Per pair, the LMOs it interacts with.
        extended: Per pair, its PAO domain widened by every interacting LMO's,
            which is the basis the non-projected families go through.
        rot_X: The PNO transform of every neighbour pair this chunk reads,
            deduplicated: a neighbour is shared by many pairs.
        rot_paos: Parallel to ``rot_X``, that neighbour pair's PAO domain, from
            which the numerics recovers its position inside ``extended``.
        nb_ij: Per pair, per interacting LMO ``k``, the index into ``rot_X`` of
            pair ``(k, j)``, or ``-1`` when it does not exist or has no PNOs.
            Empty for a weak pair, which reads no neighbour block at all.
        nb_ji: The same for pair ``(k, i)``, which the ``ji`` blocks read.
            Empty on the diagonal, where there are no ``ji`` blocks.
    """
    return _pno_integral_blocks(
        q_ij, q_ia, q_ab, metric, X_pno, i_lmo, j_lmo, n_pno, strong, ribfs,
        paos, lmos, extended, rot_X, rot_paos, nb_ij, nb_ji,
    )


_stages["compute_pno_overlaps"] = compute_pno_overlaps
_stages["transform_pnos"] = transform_pnos
_stages["compute_pno_integrals"] = compute_pno_integrals
globals().update(_stages)
__all__ += PHASES


def _autoload_compiled_backends():
    """Load ``dlpno_stages`` and select its ``cpp`` backends, if it is importable.

    The main build compiles the stage module into ``build/lib`` next to the
    ``einsums`` package, so on any PYTHONPATH that can import einsums the
    compiled backends are there too - and measured at 10 threads they are the
    difference between parity with psi4 and a 4x deficit on the integral
    phase.
    Making their selection depend on a per-driver flag meant every entry point
    that forgot the flag silently benchmarked the Python numerics, which is
    exactly what happened.
    Presence of the module is the intent; importability is the test.

    Precedence, weakest first: this autoload, then ``EINSUMS_STAGE_BACKEND``,
    then an explicit ``--backend`` spec.
    The env var wins by suppressing the autoload entirely rather than by being
    re-applied over it, because the registry applies an env spec only once; so
    ``EINSUMS_STAGE_BACKEND=python`` is also the off switch.

    A module that fails the sealed-world handshake is a hard error here, same
    as on the explicit path: a stale ``dlpno_stages`` must be rebuilt or
    deleted, never silently swapped for the Python backend - a build glitch
    must not change which code computed a published number.
    """
    import importlib.util
    import os

    if os.environ.get("EINSUMS_STAGE_BACKEND") is not None:
        return
    if importlib.util.find_spec("dlpno_stages") is None:
        return

    from einsums import stages as _est

    module = _est.load_stage_module("dlpno_stages")
    for attr in dir(module):
        if attr.startswith("stage_") and callable(getattr(module, attr)):
            _est.select(**{attr[len("stage_"):]: "cpp"})


_autoload_compiled_backends()


def run_phases(mp2, session, **per_phase):
    """Run every phase of *mp2* as a stage inside *session*.

    Args:
        mp2: The :class:`DLPNOMP2` instance.
        session: An :class:`einsums.stages.Session`, already capturing.
        **per_phase: Keyword arguments forwarded to a named phase, as
            ``lmp2_iterations={"optimize": False}``.
    """
    pno_args = None
    for name in PHASES:
        # The contracted stages are called with their explicit arguments
        # rather than with the solver object, which is the whole point of
        # having split them. Their outputs go back onto the instance because
        # later phases still read them off self.
        if name == "plan_pno_transform":
            pno_args = _stages[name](mp2)
            continue
        if name == "transform_pnos":
            mp2._finish_pno_transform(_stages[name](**pno_args))
            pno_args = None
            continue
        if name == "compute_pno_overlaps":
            solver = mp2.lmp2_solver()
            solver.take_overlaps(_stages[name](*solver.overlap_args()))
            continue
        _stages[name](mp2, **per_phase.get(name, {}))
