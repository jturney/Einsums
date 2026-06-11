..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-tensors:

***********************
Tutorial: Tensors 101
***********************

This tutorial introduces the core data structure in Einsums: the ``Tensor``.
You will learn how to create, inspect, and manipulate tensors.

Prerequisites
=============

Include Einsums and set up the namespace aliases used throughout these tutorials:

.. code-block:: cpp

    #include <Einsums/Tensor/Tensor.hpp>
    #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
    #include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
    #include <Einsums/TensorUtilities/CreateIdentity.hpp>
    #include <Einsums/Print.hpp>

    using namespace einsums;

Creating Tensors
================

A ``Tensor<T, Rank>`` is a dense, contiguous, n-dimensional array of type ``T``
with ``Rank`` dimensions. The first argument is always a name (used for debugging
and profiling).

.. code-block:: cpp

    // A 10x10 matrix of doubles
    auto A = Tensor<double, 2>("A", 10, 10);

    // A rank-3 tensor of floats
    auto B = Tensor<float, 3>("B", 5, 6, 7);

    // A vector of complex doubles
    auto v = Tensor<std::complex<double>, 1>("v", 100);

Convenience creators avoid repetitive initialization:

.. code-block:: cpp

    // All zeros
    auto Z = create_zero_tensor<double>("Z", 4, 4);

    // Random values in [-1, 1]
    auto R = create_random_tensor<double>("R", 4, 4);

    // Identity matrix
    auto I = create_identity_tensor<double>("I", 4, 4);

Element Access
==============

Use ``operator()`` with one index per dimension:

.. code-block:: cpp

    auto A = Tensor<double, 2>("A", 3, 3);
    A(0, 0) = 1.0;
    A(1, 2) = 3.14;

    double val = A(1, 2);  // 3.14

For a rank-1 tensor (vector):

.. code-block:: cpp

    auto v = Tensor<double, 1>("v", 5);
    v(0) = 10.0;
    v(4) = 20.0;

Querying Shape
==============

.. code-block:: cpp

    auto A = Tensor<double, 3>("A", 3, 4, 5);

    size_t rank = A.Rank;       // 3 (compile-time constant)
    size_t d0   = A.dim(0);     // 3
    size_t d1   = A.dim(1);     // 4
    size_t d2   = A.dim(2);     // 5
    size_t s0   = A.stride(0);  // 20 (= 4 * 5)
    size_t s1   = A.stride(1);  // 5
    size_t s2   = A.stride(2);  // 1
    size_t n    = A.size();     // 60 (= 3 * 4 * 5)

    std::string name = A.name(); // "A"

Raw Data Pointer
================

For interoperability with C libraries or BLAS:

.. code-block:: cpp

    double *ptr = A.data();  // Pointer to first element

Filling and Zeroing
====================

.. code-block:: cpp

    A.zero();          // Set all elements to 0
    A.set_all(3.14);   // Set all elements to 3.14

Printing
========

Einsums provides formatted output via ``println``:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 3, 3);
    println(A);
    // Prints:
    // Name: A
    //   Dims: 3x3
    //   [data...]

You can print to a ``std::FILE *`` or a stream using ``fprintln``. To print to a string, use ``fprintln`` with a ``std::ostringstream``.

Copying
=======

Tensors support copy construction and assignment:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = Tensor<double, 2>(A);   // Deep copy
    auto C = A;                       // Also deep copy (copy assignment)

    // Modify B without affecting A
    B(0, 0) = 999.0;
    // A(0, 0) is unchanged

Supported Types
===============

Einsums tensors work with:

- ``float``, ``double``
- ``std::complex<float>``, ``std::complex<double>``
- Integer types (``int``, ``int64_t``, etc.) for certain operations

The most common choice is ``double`` for scientific computing.

What's Next
===========

- :ref:`tutorial-einsum` -- Tensor contractions with Einstein summation
- :ref:`tutorial-views` -- Non-owning views and slicing
- :ref:`tutorial-linalg` -- Linear algebra operations
