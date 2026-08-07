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

from einsums.stages import stage

from .mp2 import DLPNOMP2

__all__ = ["PHASES", "run_phases"]

# In dependency order, which is the order compute_energy called them in.
PHASES = [
    "setup_orbitals",
    "compute_doi",
    "prep_sparsity",
    "compute_metric",
    "compute_qia",
    "precompute_fits",
    "pno_transform",
    "compute_pno_overlaps",
    "lmp2_iterations",
]


def _register(name: str):
    """Register ``DLPNOMP2.<name>`` as an eager, not-yet-contracted stage."""
    method = getattr(DLPNOMP2, name)

    @stage(eager=True, contract=False)
    @functools.wraps(method)
    def run(mp2, **kwargs):
        return method(mp2, **kwargs)

    return run


_stages = {name: _register(name) for name in PHASES}
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
        _stages[name](mp2, **per_phase.get(name, {}))
