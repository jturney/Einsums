..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto-from-numpy:

*******************
Porting from NumPy
*******************

You have working NumPy code and want the Einsums equivalent.
Most of it translates directly.
This page gives the mapping and then the four places where the semantics genuinely differ, which are the ones worth reading even if the rest looks obvious.

.. contents:: On this page
    :local:
    :depth: 1

The mapping
===========

Everything in this table was checked against NumPy on the version of the library these docs were built from.

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - NumPy
      - Einsums
    * - ``np.einsum("ik,kj->ij", A, B)``
      - ``einsums.einsum("ik;kj->ij", C, A, B)``
    * - ``A @ B``
      - ``A @ B``
    * - ``A + B``, ``A * B``
      - ``A + B``, ``A * B``
    * - ``A.T``
      - ``einsums.permute("ij->ji", T, A)``
    * - ``np.sum(A * B)``
      - ``einsums.linalg.dot(A, B)``
    * - ``np.zeros((m, n))``
      - ``einsums.zeros([m, n])``
    * - ``np.ones((m, n))``
      - ``einsums.ones([m, n])``
    * - ``np.eye(n)``
      - ``einsums.eye(n)``
    * - ``A[:n, n:]``
      - ``A[:n, n:]``

Shapes are passed as a list rather than a tuple, and the dtype argument is a string such as ``"float64"``.

The four real differences
=========================

The output is an argument, not a return value
---------------------------------------------

``np.einsum`` returns a new array.
``einsums.einsum`` writes into an output you supply, and returns nothing:

.. code-block:: python

    C = np.einsum("ik,kj->ij", A, B)        # NumPy

    C = einsums.zeros([m, n])               # Einsums
    einsums.einsum("ik;kj->ij", C, A, B)

This is not an inconvenience to work around.
It is what lets a contraction accumulate into an existing result with ``c_pf=1.0``, and what lets a captured graph plan its memory, since the destination exists before the operation is recorded.

The separator is a semicolon
----------------------------

``"ik;kj->ij"``, not ``"ik,kj->ij"``.
In Python the comma raises a ``ValueError`` when the call runs.
In C++ it fails to compile, because the spec is parsed by a ``consteval`` constructor.

The arrow-left spelling ``"ij <- ik ; kj"`` is equivalent if you prefer it.

Conversion into Einsums copies; conversion out does not
--------------------------------------------------------

This asymmetry is the one that causes real bugs, so it is worth stating precisely:

.. code-block:: python

    t = einsums.asarray(numpy_array)    # COPIES; later writes to numpy_array are not seen
    a = np.asarray(einsums_tensor)      # ALIASES; writes through a change the tensor

``einsums.asarray`` copies regardless of the input's memory order, because Einsums tensors are column-major and NumPy arrays are usually not.
Viewing a tensor back through ``np.asarray`` gives a genuine zero-copy view, which reports itself as Fortran-contiguous.

``einsums.array`` copies as well, matching ``np.array``.

So a NumPy array handed to Einsums is a snapshot, and a tensor handed to NumPy is a window.
Code that writes into ``numpy_array`` after converting it, and expects the tensor to follow, is the bug this produces.

Einsums calls do not take NumPy arrays
--------------------------------------

Passing raw NumPy arrays to ``einsums.einsum`` raises ``TypeError``.
Convert first:

.. code-block:: python

    A = einsums.asarray(numpy_A)
    B = einsums.asarray(numpy_B)
    C = einsums.zeros([m, n])
    einsums.einsum("ik;kj->ij", C, A, B)

Smaller things worth knowing
============================

**Tensors are column-major.**
Constructed tensors are column-major, which is why conversion in copies and why ``np.asarray`` of a tensor reports ``F_CONTIGUOUS``.
It rarely matters through the tensor API and matters a great deal if you walk memory yourself.

**Import submodules with ``from``.**
``import einsums.linalg`` raises ``ModuleNotFoundError``; ``from einsums import linalg`` works, as does reaching it as an attribute after ``import einsums``.

**Tensors have names.**
``einsums.create_random_tensor("A", [m, n])`` takes a name, which appears in profiler output and in graph reports.
It is worth giving a meaningful one; it is how you find the operation again in ``explain()``.

**No implicit transposition.**
NumPy will happily rearrange whatever you ask for.
Einsums picks a contraction route at compile time and will not insert a physical permutation to reach a BLAS call, so a spec that does not map onto one runs on the generic algorithm.
If a permutation would pay, write it with ``permute`` and contract the result.
:ref:`howto-contractions` shows how to confirm which route a contraction actually took.

Where the win is
================

A direct translation of NumPy code gives you Einsums' dispatch, which is worth having but is not the reason to move.
The reason is that a sequence of contractions can be captured, optimized as a unit, and replayed.
That is where fusion, memory planning, and batching come from, and none of it is available to eager calls in either library.

See :ref:`howto-graphs` for the capture recipe, and :ref:`howto-ccsd` for a whole method built that way.
