..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-views:

*************************************
Tutorial: TensorViews and Slicing
*************************************

A ``TensorView`` is a non-owning window into an existing ``Tensor``. Views let
you work with submatrices, rows, columns, and blocks without copying data.

Setup
=====

.. code-block:: cpp

    #include <Einsums/Tensor/Tensor.hpp>
    #include <Einsums/TensorAlgebra.hpp>
    #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

Creating Views with Range
=========================

Use ``Range{start, end}`` (half-open interval) to slice dimensions:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 10, 10);

    // Top-left 3x3 submatrix
    auto block = A(Range{0, 3}, Range{0, 3});
    // block is a TensorView<double, 2> -- no data copy!

    // Modify through the view -- changes A
    block(0, 0) = 999.0;
    // A(0, 0) is now 999.0

    // Row slice: row 5, all columns
    auto row = A(5, All);
    // row is a TensorView<double, 1> with 10 elements

    // Column slice: all rows, column 3
    auto col = A(All, 3);
    // col is a TensorView<double, 1> with 10 elements

Quantum Chemistry Example: Occupied/Virtual Blocks
===================================================

A common pattern in quantum chemistry is splitting a matrix into occupied and
virtual blocks:

.. code-block:: cpp

    int n_occ  = 5;
    int n_virt = 15;
    int n_orbs = n_occ + n_virt;

    auto F = create_random_tensor<double>("Fock", n_orbs, n_orbs);

    // F_oo: occupied-occupied block
    auto Foo = F(Range{0, n_occ}, Range{0, n_occ});

    // F_ov: occupied-virtual block
    auto Fov = F(Range{0, n_occ}, Range{n_occ, n_orbs});

    // F_vv: virtual-virtual block
    auto Fvv = F(Range{n_occ, n_orbs}, Range{n_occ, n_orbs});

    // These are views -- Fov(i, a) accesses F(i, n_occ + a)

Views in Einsum
===============

Views work seamlessly with ``einsum``:

.. code-block:: cpp

    auto C = create_random_tensor<double>("C", 20, 20);
    auto Co = C(Range{0, 20}, Range{0, 5});    // 20 x 5 occupied MOs
    auto Cv = C(Range{0, 20}, Range{5, 20});   // 20 x 15 virtual MOs

    auto AO_ints = create_random_tensor<double>("ints", 20, 20);
    auto MO_ints = create_zero_tensor<double>("MO_ints", 5, 15);

    // Transform: MO_ints_ia = Co^T * AO_ints * Cv
    auto tmp = create_zero_tensor<double>("tmp", 5, 20);
    einsum(Indices{i, k}, &tmp, Indices{k, i}, Co, Indices{k, j}, AO_ints);
    einsum(Indices{i, j}, &MO_ints, Indices{i, k}, tmp, Indices{k, j}, Cv);

Strides
=======

Views track strides automatically. A contiguous matrix has ``stride(1) == 1``
(column-major innermost), but a view may have larger strides:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 10, 10);
    auto block = A(Range{2, 5}, Range{3, 7});

    println("block dim:    {}x{}", block.dim(0), block.dim(1));    // 3x4
    println("block stride: {}, {}", block.stride(0), block.stride(1));
    // stride(0) = 10 (A's row stride), stride(1) = 1

Einsums and BLAS handle non-unit strides correctly.

View Lifetime
=============

.. warning::

   A ``TensorView`` holds a pointer into the original tensor. The original
   tensor **must outlive** all views into it. Using a view after the tensor
   is destroyed is undefined behavior.

.. code-block:: cpp

    TensorView<double, 2> bad_view;
    {
        auto A = create_random_tensor<double>("A", 5, 5);
        bad_view = A(Range{0, 3}, Range{0, 3});
    }
    // A is destroyed here!
    // bad_view(0, 0);  // UNDEFINED BEHAVIOR -- dangling pointer

``All`` Sentinel
================

Use ``All`` to select the full extent of a dimension:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 5, 5);

    auto row2 = A(2, All);     // Row 2 as a vector view
    auto col3 = A(All, 3);     // Column 3 as a vector view

What's Next
===========

- :ref:`tutorial-compute-graph` -- Recording and replaying operation sequences
- :ref:`tutorial-performance` -- Understanding PackedGemm and profiling
