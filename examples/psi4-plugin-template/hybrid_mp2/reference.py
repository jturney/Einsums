#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""The method's true input: a host-free dataclass of plain arrays.

This module imports numpy and nothing else. That is the host-integration
rule made structural: one adapter module per host fills this in
(:mod:`.adapter` for psi4), and nothing downstream of it may import the host.
The method itself - :mod:`.mp2`, :mod:`.stages`, the C++ port under ``cpp/`` -
sees only this, so it runs unchanged from a different host or from arrays
loaded off disk.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["Reference"]


@dataclass(frozen=True)
class Reference:
    """Everything the correlated method needs, as plain arrays.

    Keep this to buffers and scalars. A psi4 ``Matrix`` held here would work
    until the day the method runs without psi4, which is the day this class
    exists for.
    """

    #: SCF total energy of the reference determinant.
    e_scf: float
    #: Active occupied orbital energies, shape ``(nocc,)``.
    eps_occ: np.ndarray
    #: Active virtual orbital energies, shape ``(nvir,)``.
    eps_vir: np.ndarray
    #: MO-basis two-electron integrals ``(ia|jb)`` in chemists' notation,
    #: shape ``(nocc, nvir, nocc, nvir)``, C-contiguous.
    iajb: np.ndarray
