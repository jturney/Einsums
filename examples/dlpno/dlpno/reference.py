#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The reference-wavefunction data a DLPNO calculation starts from.

This is the whole contract with the SCF program. Everything is a plain numpy
buffer or an index list, so nothing under ``dlpno/`` imports psi4 and neither
library needs to be built against the other; ``psi4_source.py`` is the only
module that knows psi4 exists.

All quantities are in the AO basis, C1 (no point-group symmetry), chemists'
notation for the integrals.
"""

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Reference"]


@dataclass
class Reference:
    """Everything DLPNO needs from the converged reference determinant."""

    #: AO overlap, ``(nbf, nbf)``.
    S: np.ndarray
    #: AO Fock matrix, ``(nbf, nbf)``.
    F: np.ndarray
    #: All doubly occupied orbitals, ``(nbf, nocc)``. Defines the PAO projector.
    C_occ: np.ndarray
    #: Localized *active* occupied orbitals, ``(nbf, naocc)``.
    C_lmo: np.ndarray
    #: Raw three-index integrals ``(Q|mn)``, ``(naux, nbf, nbf)``. No metric applied.
    eri_3index: np.ndarray
    #: Auxiliary metric ``(P|Q)``, ``(naux, naux)``.
    metric: np.ndarray

    #: Which orbital basis functions sit on each atom.
    atom_to_bf: list = field(default_factory=list)
    #: Which auxiliary basis functions sit on each atom.
    atom_to_ribf: list = field(default_factory=list)
    #: AO dipole integrals ``(3, nbf, nbf)``, for the dipole pair prescreening.
    dipole_ao: np.ndarray = None
    #: Callable yielding DFT-grid blocks ``(phi, w, bf_map)`` for the
    #: differential overlap integrals. The one non-buffer entry in the contract:
    #: the grid has to stream (collocating it whole is hundreds of MB), and the
    #: integrals cannot be finished in the bridge because they need the PAO
    #: coefficients that :meth:`DLPNOBase.setup_orbitals` builds. ``None``
    #: disables screening, which is the untruncated reference calculation.
    grid_blocks: object = None
    #: Number of core orbitals to scale PNO thresholds for (0 when core is frozen).
    n_core: int = 0
    #: SCF total energy, carried through so the driver can report a total.
    e_scf: float = 0.0

    @property
    def nbf(self):
        return self.S.shape[0]

    @property
    def naocc(self):
        return self.C_lmo.shape[1]

    @property
    def naux(self):
        return self.metric.shape[0]

    def validate(self):
        nbf, naux = self.nbf, self.naux
        checks = {
            "S": (nbf, nbf),
            "F": (nbf, nbf),
            "C_occ": (nbf, self.C_occ.shape[1]),
            "C_lmo": (nbf, self.naocc),
            "eri_3index": (naux, nbf, nbf),
            "metric": (naux, naux),
        }
        for name, want in checks.items():
            got = np.asarray(getattr(self, name)).shape
            if got != want:
                raise ValueError(f"Reference.{name}: expected shape {want}, got {got}")
        return self
