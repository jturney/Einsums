#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""DLPNO correlation methods on the einsums ComputeGraph.

A port of psi4's ``psi4/src/psi4/dlpno`` with the tensor algebra expressed in
einsums and the iterative solvers captured as ComputeGraphs. The module split
follows psi4's so the two can be read side by side:

===========================  ==========================================
this package                 psi4
===========================  ==========================================
``sparse.py``                ``dlpno/sparse.cc``
``base.py``                  ``dlpno/dlpno.cc``   (class ``DLPNO``)
``mp2.py``                   ``dlpno/mp2.cc``     (class ``DLPNOMP2``)
===========================  ==========================================

``reference.py`` and ``psi4_source.py`` have no psi4 counterpart: they define
the psi4-free data contract the port starts from and the one adapter that fills
it in, so nothing else here knows psi4 exists.
"""

from .base import DLPNOBase
from .mp2 import DLPNOMP2
from .reference import Reference
from .thresholds import Thresholds

__all__ = ["DLPNOBase", "DLPNOMP2", "Reference", "Thresholds"]
