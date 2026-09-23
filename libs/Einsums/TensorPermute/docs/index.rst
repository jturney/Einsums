..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_TensorPermute:

=============
TensorPermute
=============

This module reorders the axes of a tensor, with each axis named by a character.
``permute(0.0, "kij", &C, 1.0, "ijk", A)`` sets ``C(k, i, j) = A(i, j, k)``.
It works on the rank-erased ``TensorImpl`` that every tensor type wraps, and nothing in it is templated on the index letters.
A permutation can therefore be built at run time and rewritten as data, which is what the compute graph's string specs need.

The kernel is HPTT.
Building an HPTT plan is the expensive part of a small transpose, so plans are cached per thread, keyed on the shapes, and reused with new data and scalars.

See the :ref:`API reference <modules_Einsums_TensorPermute_api>` of this module for more
details.

Public API
----------

- :cpp:func:`~einsums::tensor_permute::permute` - computes ``C = beta * C + alpha * permute(A)``, with the axes of each operand named by a string.
- :cpp:func:`~einsums::tensor_permute::transpose` - computes ``C = A^T`` for matrices, and checks the ranks and the output's size first.
- :cpp:func:`~einsums::tensor_permute::compile_permute` - builds the plan without running it, for a permutation repeated on new data of the same shapes.

The typed ``einsums::tensor_algebra::permute``, which names the axes with compile-time index types, is built on the same kernel and the same plan cache.
