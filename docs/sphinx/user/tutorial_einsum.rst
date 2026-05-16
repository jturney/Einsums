..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-einsum:

*****************************
Tutorial: Einstein Summation
*****************************

The ``einsum`` function is the heart of Einsums. It expresses tensor contractions
using Einstein summation convention: any index that appears in both input tensors
but not in the output is summed over (contracted).

Setup
=====

.. code-block:: cpp

    #include <Einsums/TensorAlgebra.hpp>
    #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
    #include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

The ``index`` namespace provides pre-defined index labels ``i``, ``j``, ``k``,
``l``, ``m``, ``n``, ``o``, ``p``, ``q``, ``r``, ``s``, ``t``.

Index Notation
==============

Each index is a compile-time struct. You pass them in ``Indices{...}`` tuples to
specify the contraction pattern:

.. code-block:: cpp

    einsum(Indices{i, j},     // C indices (output)
           &C,                 // output tensor
           Indices{i, k},     // A indices
           A,                  // input tensor A
           Indices{k, j},     // B indices
           B);                 // input tensor B

The rule: **index ``k`` appears in A and B but not in C, so it is summed over**.
This computes :math:`C_{ij} = \sum_k A_{ik} B_{kj}` (matrix multiplication).

Matrix Multiplication (GEMM)
============================

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 10, 5);
    auto B = create_random_tensor<double>("B", 5, 8);
    auto C = create_zero_tensor<double>("C", 10, 8);

    einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
    // C = A * B  (dispatches to BLAS GEMM)

Dot Product
===========

When the output is a scalar (rank 0), use ``Indices{}`` for C:

.. code-block:: cpp

    auto x = create_random_tensor<double>("x", 100);
    auto y = create_random_tensor<double>("y", 100);
    double result = 0.0;

    einsum(Indices{}, &result, Indices{i}, x, Indices{i}, y);
    // result = sum_i x_i * y_i

Outer Product
=============

When no indices are shared between A and B, you get an outer product:

.. code-block:: cpp

    auto x = create_random_tensor<double>("x", 4);
    auto y = create_random_tensor<double>("y", 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    einsum(Indices{i, j}, &C, Indices{i}, x, Indices{j}, y);
    // C_ij = x_i * y_j  (dispatches to BLAS GER)

Transpose
=========

The ``permute`` function reorders indices:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 4, 6);
    auto B = create_zero_tensor<double>("B", 6, 4);

    permute(0.0, Indices{j, i}, &B, 1.0, Indices{i, j}, A);
    // B = A^T

The signature is ``permute(beta, C_indices, &C, alpha, A_indices, A)``
which computes :math:`C = \beta C + \alpha \text{permute}(A)`.

Scaling with Prefactors
=======================

``einsum`` supports scaling prefactors for accumulation:

.. code-block:: cpp

    // C = 0.0 * C + 1.0 * A * B  (overwrite C)
    einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k}, A, Indices{k, j}, B);

    // C += A * B  (accumulate into C)
    einsum(1.0, Indices{i, j}, &C, 1.0, Indices{i, k}, A, Indices{k, j}, B);

    // C = 2.0 * C + 0.5 * A * B
    einsum(2.0, Indices{i, j}, &C, 0.5, Indices{i, k}, A, Indices{k, j}, B);

When prefactors are omitted, the defaults are ``c_prefactor = 0`` and
``ab_prefactor = 1`` (overwrite mode).

Higher-Rank Contractions
========================

Einsum handles arbitrary tensor ranks:

.. code-block:: cpp

    // Rank-3 contraction: C_il = sum_jk A_ijk * B_jkl
    auto A = create_random_tensor<double>("A", 4, 5, 3);
    auto B = create_random_tensor<double>("B", 5, 3, 6);
    auto C = create_zero_tensor<double>("C", 4, 6);

    einsum(Indices{i, l}, &C, Indices{i, j, k}, A, Indices{j, k, l}, B);

    // Rank-4 with batch: C_bij = sum_k A_bik * B_bkj
    auto A4 = create_random_tensor<double>("A4", 2, 4, 3);
    auto B4 = create_random_tensor<double>("B4", 2, 3, 5);
    auto C4 = create_zero_tensor<double>("C4", 2, 4, 5);

    einsum(Indices{b, i, j}, &C4, Indices{b, i, k}, A4, Indices{b, k, j}, B4);

Dispatch and Performance
========================

Einsums automatically selects the best algorithm for each contraction:

1. **BLAS specialization** (fastest): DOT, GER, GEMV, GEMM for simple patterns
2. **PackedGemm** (fast): BLIS-style cache-blocked packing for complex patterns
   including multi-M, multi-N, multi-K, and batch dimensions
3. **Generic algorithm** (fallback): Nested loops for patterns that don't map to
   BLAS (e.g., Hadamard products with repeated indices)

You don't need to think about dispatch -- Einsums chooses automatically. The
profiler will show you which algorithm was selected (see :ref:`tutorial-performance`).

What's Next
===========

- :ref:`tutorial-views` -- Slicing tensors for submatrix operations
- :ref:`tutorial-linalg` -- Eigendecomposition, SVD, and linear solvers
- :ref:`tutorial-compute-graph` -- Capturing and replaying computation sequences
