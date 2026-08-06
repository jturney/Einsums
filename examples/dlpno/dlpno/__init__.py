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

``reference.py``, ``reference_io.py`` and ``psi4_source.py`` have no psi4
counterpart: they define the psi4-free data contract the port starts from, its
on-disk form, and the one adapter that fills it in, so nothing else here knows
psi4 exists.

The solver names below are resolved lazily. ``reference`` and ``reference_io``
need nothing but numpy, and importing either should not drag in einsums:
otherwise a fixture could fail to load for reasons belonging to the library it
exists to test.
"""

import importlib

__all__ = ["DLPNOBase", "DLPNOMP2", "Reference", "Thresholds"]

_LAZY = {
    "DLPNOBase": "base",
    "DLPNOMP2": "mp2",
    "Reference": "reference",
    "Thresholds": "thresholds",
}


def __getattr__(name):
    if name in _LAZY:
        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
