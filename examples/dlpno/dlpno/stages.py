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

**No phase states a contract yet.** They are methods, so a contract means
extracting the ``self`` fields each one touches into a state dataclass. The
sizes say that is not uniform work: five phases touch fewer than ten fields
between reads and writes, while ``pno_transform`` touches 32,
``compute_pno_overlaps`` 27 and ``lmp2_iterations`` 45. Writing nine
extractions before the timing table exists would be the exact mistake this
framework is built to prevent, so each phase states its contract when it is
promoted, and ``contract=False`` reports the debt until then.
"""

import functools

from einsums.stages import TensorD, stage

from .contracts import CouplingPlan, PnoOverlaps
from .mp2 import DLPNOMP2
from .pno_overlaps import compute_pno_overlaps as _compute_pno_overlaps

__all__ = ["PHASES", "run_phases", "compute_pno_overlaps"]

# In dependency order, which is the order compute_energy called them in.
#
# compute_pno_overlaps is absent because it is no longer one phase: it is
# plan_pno_couplings, which stays Python, followed by the contracted numerics
# stage declared below. It is the first phase to state a contract, and the one
# M3 promotes.
PHASES = [
    "setup_orbitals",
    "compute_doi",
    "prep_sparsity",
    "compute_metric",
    "compute_qia",
    "precompute_fits",
    "pno_transform",
    "plan_pno_couplings",
    "compute_pno_overlaps",
    "lmp2_iterations",
]

# Phases still carrying contract debt: methods reading fields off self.
# plan_pno_couplings is among them and is not a promotion target either - it is
# index arithmetic, and nothing measured says it is hot.
UNCONTRACTED = [n for n in PHASES if n != "compute_pno_overlaps"]


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


_stages["compute_pno_overlaps"] = compute_pno_overlaps
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
    for name in PHASES:
        if name == "compute_pno_overlaps":
            # The one contracted stage: called with its seven arguments rather
            # than with the solver object, which is the whole point of having
            # split it. Its outputs go back onto the instance because
            # lmp2_iterations still reads them off self.
            overlaps = _stages[name](
                mp2.X_pno, mp2.S_pao, mp2.lmopair_to_paos, mp2.n_pno,
                mp2.bucket_of, mp2.bucket_dims, mp2.plan,
            )
            mp2._S_cls, mp2._S_T = overlaps.S_cls, overlaps.S_T
            continue
        _stages[name](mp2, **per_phase.get(name, {}))
