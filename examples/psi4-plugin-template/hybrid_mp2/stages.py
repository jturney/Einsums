#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""The plugin's phases, registered as ``einsums.stages`` stages.

One contracted stage, which is the template being honest about its size. A
real method grows a list of phases here (see ``examples/dlpno/dlpno/stages.py``
for eleven of them); each starts uncontracted (``contract=False`` reports the
debt), gets measured under a Session, and states its contract when it is
promoted. The promotion workflow, in order:

1. ``python -m einsums.stages extract`` when the phase is a method: it reports
   every ``self`` field the phase touches and refuses to scaffold the split
   until a cut spec says what crosses, what stays in the plan, and what stays
   in the finish.
2. Annotate the numerics' signature with cross-boundary types and declare the
   ``@contract`` return; the validator refuses anything that cannot cross.
3. ``python -m einsums.stages promote hybrid_mp2/stages.py --out hybrid_mp2/cpp``
   generates the C++ side; only the port skeleton's body is yours to fill.
"""

from einsums.stages import TensorD, stage

from .contracts import Mp2Energy
from .mp2 import mp2_energy as _mp2_energy

__all__ = ["mp2_energy"]


@stage(eager=True)
def mp2_energy(iajb: TensorD, eps_occ: TensorD, eps_vir: TensorD) -> Mp2Energy:
    """RHF MP2 correlation energy from MO ``(ia|jb)`` integrals.

    Pair-driven over ``i <= j``: one captured graph over the pair loop,
    replayed as an OpenMP team. ``eager`` because the stage runs its own
    graph to completion rather than emitting into an ambient capture, which
    the session reports as a graph split.

    Args:
        iajb: Chemists' ``(ia|jb)``, shape ``(nocc, nvir, nocc, nvir)``.
        eps_occ: Active occupied orbital energies, shape ``(nocc,)``.
        eps_vir: Active virtual orbital energies, shape ``(nvir,)``.
    """
    return _mp2_energy(iajb, eps_occ, eps_vir)
