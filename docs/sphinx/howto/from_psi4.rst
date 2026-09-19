..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto-from-psi4:

******************
Working with psi4
******************

How to move integrals and orbital data between psi4 and Einsums.
The connection is at the buffer level, through ``einsums.interop.psi4``.

.. contents:: On this page
    :local:
    :depth: 1

What the bridge is, and is not
==============================

Nothing in ``einsums.interop.psi4`` imports psi4, and psi4 is not compiled against Einsums.
Every function is duck-typed against the data it needs, which for a symmetry-blocked matrix means ``nirrep()``, ``symmetry()``, and ``.nph``, and for single-block data means any two-dimensional array-like.

That is a deliberate choice rather than an accident of implementation.
It means the two projects have no shared ABI, that either can be rebuilt without the other, and that a version mismatch cannot produce the kind of failure that only shows up at load time.
What you give up is automatic conversion: you ask for a specific reshape, and you get a tensor.

Get a matrix across
===================

``from_matrix`` takes a symmetry-blocked psi4 ``Matrix`` and produces a rank-2 ``TiledRuntimeTensorD``.
Block ``h`` lands in tile ``(h, h ^ symmetry)``, and empty blocks stay unstored as structural zeros.

.. code-block:: python

    import psi4
    from einsums.interop import psi4 as bridge

    mints = psi4.core.MintsHelper(wfn.basisset())
    S = bridge.from_matrix(mints.ao_overlap(), name="S")
    # -> TiledRuntimeTensorD

The result is tiled even when the molecule has no symmetry, because the block structure is what the psi4 object carries.
If you want a dense tensor from a single-block matrix, use ``dense`` instead.

Get a dense block across
========================

``dense`` reshapes any single-block matrix into a dense ``RuntimeTensorD`` of the shape you ask for.
The element count has to match exactly, which is the check that catches a wrong ``nbf`` before it becomes a wrong energy.

.. code-block:: python

    nbf = wfn.nso()
    G = bridge.dense(mints.ao_eri(), (nbf, nbf, nbf, nbf), name="AO ERI")

This is the usual way to get conventional two-electron integrals in.
The psi4 object is a chemists'-pair matrix; the shape argument is what turns it back into a rank-4 tensor.

Get DF three-index integrals across
===================================

``df_tensor`` takes the ``(naux, d2*d3)`` matrix psi4's ``DFTensor`` returns and gives a rank-3 ``(naux, d2, d3)`` tensor:

.. code-block:: python

    aux   = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", basis)
    dfobj = psi4.core.DFTensor(wfn.basisset(), aux, wfn.Ca(), nocc, nvirt)

    Qso = bridge.df_tensor(dfobj.Qso(), nbf, nbf, name="Qso")

``naux`` is inferred from the element count rather than passed, so only the two trailing dimensions are given.

The half-transformed case
=========================

``mo_bra_half_transform`` reshapes ``mints.mo_bra_half_transform(C1, C2)`` into a rank-4 ``(n1, n2, nbf, nbf)`` tensor: chemists' notation with the bra transformed and the ket still in AOs.
It exists because the half transform is the point where handing work to Einsums starts to pay, and doing the reshape by hand is easy to get subtly wrong.

Symmetry-blocked two-electron integrals
=======================================

``so_eri`` assembles ``mints.so_eri_blocked()`` into a rank-4 ``TiledRuntimeTensorD``.
Pass the SO-per-irrep dimensions alongside it:

.. code-block:: python

    T = bridge.so_eri(mints.so_eri_blocked(), wfn.nsopi().to_tuple())

Only the symmetry-allowed blocks you supply are stored, which is the entire reason to use the tiled type here.

Things to watch
===============

**Everything copies.**
These functions build a tensor and fill it.
Changing the psi4 object afterwards does not change the tensor, and vice versa.
This is the same asymmetry described in :ref:`howto-from-numpy`, and it is usually what you want for integrals, which are computed once.

**Importing psi4 sets the thread count.**
Importing psi4 sets the process-wide OpenMP thread count, to ``OMP_NUM_THREADS`` if that is exported and to one otherwise.
Leaving both unset makes Einsums run silently serial, which is a confusing way to measure performance.
Call ``psi4.set_num_threads(n)`` before any Einsums work, not after.

**Give the tensors names.**
Every bridge function takes a ``name``.
It is what appears in profiler output and graph reports.

**Take the slices contiguously.**
When slicing a NumPy array before handing it over, wrap it in ``np.ascontiguousarray``.
The conversion copies either way; a contiguous source makes the copy cheaper.

Where this is going
===================

The long-term aim is for Einsums tensors to replace psi4's ``Matrix`` and ``Vector`` rather than convert between them, with a zero-copy bridge instead of the copying one above.
Until then, treat the boundary as a place where data is handed over once, and keep the repeated work on the Einsums side of it.

Next
====

- :ref:`howto-ccsd` for a full method built on integrals brought across this way.
- :ref:`howto-from-numpy` for the NumPy half of the same boundary.
