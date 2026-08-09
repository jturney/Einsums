#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""DLPNO correlation methods on the einsums ComputeGraph.

A port of psi4's ``psi4/src/psi4/dlpno`` with the tensor algebra expressed in
einsums and the solvers captured as ComputeGraphs: every contraction is
recorded once and replayed, so the per-iteration Python cost is a few
``execute()`` calls however many GEMMs they stand for. DLPNO-MP2 is complete
and validated against psi4 two ways (see ``run_dlpno_mp2.py``); CCSD and (T)
are not started.

The core follows psi4's module split, so the two can be read side by side:

===========================  ==========================================
this package                 psi4
===========================  ==========================================
``sparse.py``                ``dlpno/sparse.cc``  (SparseMap helpers)
``base.py``                  ``dlpno/dlpno.cc``   (class ``DLPNO``)
``mp2.py``                   ``dlpno/mp2.cc``     (class ``DLPNOMP2``)
``thresholds.py``            the ``PNO_CONVERGENCE`` presets
===========================  ==========================================

A calculation is :meth:`DLPNOMP2.compute_energy`, a fixed sequence of phases:
localize orbitals and build PAOs (``setup_orbitals``), integrate differential
overlaps and dipoles for the screening decisions (``compute_doi``,
``prep_sparsity``), build the local density-fitting blocks (``compute_metric``,
``compute_qia``, ``precompute_fits``), transform each surviving pair into its
truncated pair-natural-orbital basis (``pno_transform``), form the PNO overlap
matrices that couple pairs (``compute_pno_overlaps``), and iterate the local
MP2 residual to convergence as one loop graph (``lmp2_iterations``). Each
phase's method documents its own physics and its psi4 counterpart.

Around that core:

* ``reference.py`` / ``reference_io.py`` / ``psi4_source.py`` have no psi4
  counterpart. They define the psi4-free data contract the port starts from
  (plain numpy buffers), its on-disk ``.npz`` form, and the one adapter that
  fills it from a psi4 wavefunction - so nothing else here knows psi4 exists
  and neither library links the other.
* ``tensors.py`` wraps ``einsums.linalg`` in the small helpers psi4's DLPNO
  source leans on (``doublet``/``triplet``, descending ``eigh``, ...), so the
  ported code reads like the original.
* ``stages.py`` registers the phases with ``einsums.stages`` for per-phase
  timing, and ``contracts.py`` states the cross-boundary contracts of the two
  promoted stages, whose numerics live in ``pno_overlaps.py`` and
  ``pno_xform.py`` with C++ backends under ``../cpp/``.
* ``molecules.py`` holds the shared test geometries.

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
