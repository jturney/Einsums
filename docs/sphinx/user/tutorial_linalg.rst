..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-linalg:

*****************************
Tutorial: Linear Algebra
*****************************

Einsums wraps BLAS and LAPACK routines into a convenient C++ API through the
``linear_algebra`` namespace. This tutorial covers the most commonly used
operations.

Setup
=====

.. code-block:: cpp

    #include <Einsums/LinearAlgebra.hpp>
    #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
    #include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
    #include <Einsums/TensorUtilities/CreateIdentity.hpp>

    using namespace einsums;
    using namespace einsums::linear_algebra;

Scaling: ``scale``
==================

Multiply every element by a scalar:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 4, 4);
    scale(2.0, &A);  // A *= 2.0

AXPY: ``axpy``
==============

:math:`\mathbf{Y} = \mathbf{Y} + \alpha \mathbf{X}` (BLAS Level 1):

.. code-block:: cpp

    auto X = create_random_tensor<double>("X", 100);
    auto Y = create_zero_tensor<double>("Y", 100);

    axpy(1.0, X, &Y);   // Y += X
    axpy(-0.5, X, &Y);  // Y -= 0.5 * X

Matrix Multiply: ``gemm``
=========================

:math:`\mathbf{C} = \alpha \mathbf{A B} + \beta \mathbf{C}` (BLAS Level 3):

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 10, 5);
    auto B = create_random_tensor<double>("B", 5, 8);
    auto C = create_zero_tensor<double>("C", 10, 8);

    gemm<false, false>(1.0, A, B, 0.0, &C);  // C = A * B

The template parameters control transposition:

.. code-block:: cpp

    gemm<true, false>(1.0, A, B, 0.0, &C);   // C = A^T * B
    gemm<false, true>(1.0, A, B, 0.0, &C);   // C = A * B^T

Symmetric Eigendecomposition: ``syev``
======================================

Diagonalize a symmetric matrix :math:`\mathbf{AU} = \mathbf{U\Lambda}`, where :math:`\mathbf{A} = \mathbf{A}^T` and :math:`\mathbf{UU}^H = \mathbf{I}`:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 5, 5);
    // Make symmetric: A = 0.5 * (A + A^T)
    // ... (see tutorial_einsum for permute)

    auto [eigenvectors, eigenvalues] = syev(A);
    // or syev(&A, &eivenvalues, &eigenvectors);
    // eigenvectors: 5x5 matrix (columns are eigenvectors)
    // eigenvalues: 5-element vector (ascending order)

    println("Eigenvalues: ", eigenvalues);

Note that the input matrix is overwritten with data required to perform the eigendecomposition.

Singular Value Decomposition: ``svd``
=====================================

Perform singular value decomposition :math:`\mathbf{A} = \mathbf{U \Sigma V}^T`:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 6, 4);
    auto [U, sigma, Vt] = svd(A);
    // U: 6x6 unitary matrix
    // sigma: 4-element vector of singular values
    // Vt: 4x4 unitary matrix (V transposed)

Linear Solve: ``gesv``
======================

Solve :math:`\mathbf{AX} = \mathbf{B}` for :math:`\mathbf{X}`. On exit, :math:`\mathbf{A}` will be overwritten with its LU
factorization, where the diagonal elements of the L factor are all 1, and the diagonal elements of :math:`\mathbf{A}` are the diagonal elements of the U factor,
and :math:`\mathbf{B}` will be overwritten with the value of :math:`\mathbf{X}`:

.. code-block:: cpp

    auto A = Tensor<double, 2>("A", 3, 3);
    auto B = Tensor<double, 2>("B", 3, 2);

    // Fill A (well-conditioned) and B...
    A(0, 0) = 4.0; A(0, 1) = 1.0; A(0, 2) = 0.0;
    A(1, 0) = 1.0; A(1, 1) = 5.0; A(1, 2) = 1.0;
    A(2, 0) = 0.0; A(2, 1) = 1.0; A(2, 2) = 3.0;
    B(0, 0) = 1.0; B(0, 1) = 2.0;
    B(1, 0) = 3.0; B(1, 1) = 4.0;
    B(2, 0) = 5.0; B(2, 1) = 6.0;

    int info = gesv(&A, &B);
    // B now contains the solution X
    // A is overwritten with LU factors

.. warning::

   ``gesv`` destroys both ``A`` and ``B``. Make copies if you need the
   originals.

Matrix Inverse: ``invert``
==========================

Compute the matrix inverse. That is, find a matrix :math:`\mathbf{A}^{-1}` such that :math:`\mathbf{AA}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}`:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 4, 4);
    auto A_inv = invert(A);
    // A_inv * A ≈ I

Determinant: ``det``
====================

Compute the determinant of a matrix. This is done using the LU factorization method.

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 3, 3);
    double d = det(A);
    println("det(A) = {}", d);

Norm: ``norm``
==============

Compute an induced matrix norm.

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 5, 5);
    double n = norm(A);  // Frobenius norm

Dot Product: ``dot``
====================

Compute the programmer's dot product between two vectors. That is, compute :math:`\sum_i x_iy_i`.
Use ``true_dot`` to compute the mathematician's dot product, :math:`sum_i x_i^*y_i`. For real numbers,
the two definitions are the same. For complex numbers, the two definitions will differ.

.. code-block:: cpp

    auto x = create_random_tensor<double>("x", 100);
    auto y = create_random_tensor<double>("y", 100);
    double d = dot(x, y);  // x . y

Rank-1 Update: ``ger``
======================

:math:`\mathbf{A} = \mathbf{A} + \alpha \mathbf{x y}^T`:

.. code-block:: cpp

    auto x = create_random_tensor<double>("x", 4);
    auto y = create_random_tensor<double>("y", 5);
    auto A = create_zero_tensor<double>("A", 4, 5);

    ger(1.0, x, y, &A);  // A = x * y^T

If you need :math:`\mathbf{A} = \mathbf{A} + \alpha \mathbf{x}\mathbf{y}^H`, use ``gerc``.

QR Decomposition: ``qr``
=========================

Compute the QR decomposition of a matrix. That is, find the matrices :math:`\mathbf{Q}` and :math:`\mathbf{R}` that satisfy :math:`\mathbf{A} = \mathbf{QR}`,
where :math:`\mathbf{Q}` is a unitary matrix and :math:`\mathbf{R}` is an upper-triangular matrix.

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", 6, 4);
    auto [Q, R] = qr(A);
    // Q: 6x4 orthogonal
    // R: 4x4 upper triangular

What's Next
===========

- :ref:`tutorial-views` -- Working with submatrices and slices
- :ref:`tutorial-compute-graph` -- Capturing and optimizing sequences of operations
- :ref:`tutorial-performance` -- Performance tuning
