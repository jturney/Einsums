..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _howto-views:

******************
Views and Slicing
******************

Recipes for working on part of a tensor without copying it.
:doc:`/user/tutorial_views` introduces views; this page covers the cases that come up in real code and the places where a view behaves differently from the tensor it came from.

.. contents:: On this page
    :local:
    :depth: 1

Take a sub-block
================

Index a tensor with :cpp:struct:`einsums::Range` to get a view rather than a copy.
The interval is half-open, so ``Range{0, 3}`` is the first three entries.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto A     = einsums::create_random_tensor("A", 6, 6);
            auto block = A(einsums::Range{0, 3}, einsums::Range{0, 3});

            block(0, 0) = 999.0;     // A(0, 0) is now 999.0

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [6, 6])
            block = A[0:3, 0:3]

            block[0, 0] = 999.0      # A[0, 0] is now 999.0

A view writes through to its parent.
That is the point of it, and it is also the thing to keep in mind when you pass one to a routine that writes its output.

Use ``All`` to keep a whole dimension.
Indexing a dimension with a single integer drops it, so ``A(2, All)`` is a rank-1 view of row two, not a one-row matrix.

Take occupied and virtual blocks
================================

This is the slicing pattern most quantum chemistry code is made of.
Given a matrix over all molecular orbitals, the occupied-virtual block is one view and costs nothing to form.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            using einsums::Range;

            RuntimeTensor<double> F("F", std::vector<size_t>{nmo, nmo});
            // ... fill F ...

            auto Foo = F(Range{0, nocc},    Range{0, nocc});
            auto Fov = F(Range{0, nocc},    Range{nocc, nmo});
            auto Fvv = F(Range{nocc, nmo},  Range{nocc, nmo});

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            F = einsums.zeros([nmo, nmo])
            # ... fill F ...

            Foo = F[:nocc, :nocc]
            Fov = F[:nocc, nocc:]
            Fvv = F[nocc:, nocc:]

A view is an ordinary operand.
Contracting one needs no copy and no special call:

.. code-block:: cpp

    cg::einsum("ia;a->i", &out, Fov, v);

Know what ``data()`` points at
==============================

:cpp:func:`~einsums::TensorView::data` on a view returns the pointer to the view's own first element, already adjusted for the offset into the parent.
It does not return the parent's base pointer.
This is what you want when handing a view to a BLAS call, and it is a trap if you assumed otherwise and then applied the offset a second time.

Strides are in elements rather than bytes, and tensors are column-major at construction.
For a 5 by 5 matrix, element ``F(0, 2)`` is the eleventh element of the buffer, because column two starts after two full columns of five.
Code that reaches past the tensor API and walks memory directly has to respect that.

Contract two slices of the same tensor
======================================

Two disjoint slices of one parent may be used as the two inputs of a contraction, and a slice may even be the output while another is an input.
The overlap check rejects only *provable* overlap, meaning both regions are contiguous or the base pointers are identical.
Disjoint column-major slices of one parent interleave in memory without sharing an element, so rejecting them would be wrong and the library does not.

What is rejected is an output that provably overlaps an input with a different index list.
:ref:`howto-contractions` covers that rule in full.

Use views inside a graph
========================

Views are first-class in a capture: you may form one before the capture and contract it inside, and the graph records the aliasing relationship rather than a bare pointer.
That matters because the optimizer resolves aliases to the underlying buffer when it decides what is live, so a write through a view of a tensor you own keeps its producer alive.

The one thing to avoid is forming a view of a tensor whose storage the graph may move.
Bind or materialize first, then slice.

Pitfalls
========

**A view does not own anything.**
It is valid only while its parent is.
Returning a view of a local tensor from a function gives you a dangling view, exactly as a pointer would.

**A rank-reducing index changes the spec you need.**
``A(2, All)`` is rank one, so it contracts as ``"a"`` and not as ``"ia"``.
Mismatches here are caught, but the message talks about ranks rather than about the slice that produced them.

**Python slicing follows Python conventions.**
``F[:nocc, nocc:]`` uses half-open intervals like ``Range``, so the two languages agree, but negative indices and steps behave as Python users expect rather than as ``Range`` does.

Next
====

- :ref:`howto-contractions` for the aliasing rule in full.
- :ref:`howto-ccsd` for slicing used throughout a real method.
