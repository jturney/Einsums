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
promoted, and ``contract=False`` reports the debt until then. Two have been:
``compute_pno_overlaps`` (27 fields narrowed to seven parameters) and
``transform_pnos`` (32 narrowed to fifteen), each split into a plan half that
stays Python and a contracted numerics stage declared below, with C++ backends
under ``cpp/``. ``python -m einsums.stages extract`` mechanizes the analysis
for the next one.
"""

import functools

from einsums.stages import TensorD, stage

from .contracts import CouplingPlan, PnoOverlaps, PnoTransform
from .mp2 import DLPNOMP2
from .pno_overlaps import compute_pno_overlaps as _compute_pno_overlaps
from .pno_xform import transform_pnos as _transform_pnos

__all__ = ["PHASES", "run_phases", "compute_pno_overlaps", "transform_pnos"]

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
CONTRACTED = ("compute_pno_overlaps", "transform_pnos")
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


_stages["compute_pno_overlaps"] = compute_pno_overlaps
_stages["transform_pnos"] = transform_pnos
globals().update(_stages)
__all__ += PHASES


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
