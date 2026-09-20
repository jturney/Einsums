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

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/LinearAlgebra.hpp>
            #include <Einsums/Tensor/RuntimeTensor.hpp>
            #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
            #include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
            #include <Einsums/TensorUtilities/CreateIdentity.hpp>

            using namespace einsums;
            using namespace einsums::linear_algebra;

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import numpy as np
            import einsums
            from einsums import linalg

Most of these routines take :cpp:type:`einsums::RuntimeTensor`, whose rank travels with the
value. Three do not: in C++, ``det``, ``svd`` and ``qr`` are constrained to ``MatrixConcept``,
which is a compile-time rank of two, so they want a statically ranked
:cpp:type:`einsums::Tensor` and are written that way below.

That constraint is a C++ one. Python has a single tensor type, so the same three take an ordinary
tensor there and the Python tabs are simply shorter.

Scaling: ``scale``
==================

Multiply every element by a scalar:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto A = create_random_tensor<double>("A", {4, 4});
            scale(2.0, &A);  // A *= 2.0

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [4, 4])
            linalg.scale(2.0, A)   # A *= 2.0

AXPY: ``axpy``
==============

:math:`\mathbf{Y} = \mathbf{Y} + \alpha \mathbf{X}` (BLAS Level 1):

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto X = create_random_tensor<double>("X", {100});
            auto Y = create_zero_tensor<double>("Y", {100});

            axpy(1.0, X, &Y);   // Y += X
            axpy(-0.5, X, &Y);  // Y -= 0.5 * X

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            X = einsums.create_random_tensor("X", [100])
            Y = einsums.zeros([100], name="Y")

            linalg.axpy(1.0, X, Y)    # Y += X
            linalg.axpy(-0.5, X, Y)   # Y -= 0.5 * X

Matrix Multiply: ``gemm``
=========================

:math:`\mathbf{C} = \alpha \mathbf{A B} + \beta \mathbf{C}` (BLAS Level 3):

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto A = create_random_tensor<double>("A", {10, 5});
            auto B = create_random_tensor<double>("B", {5, 8});
            auto C = create_zero_tensor<double>("C", {10, 8});

            gemm<false, false>(1.0, A, B, 0.0, &C);  // C = A * B

            // Template parameters control transposition
            gemm<true, false>(1.0, At, B, 0.0, &C);  // C = A^T * B
            gemm<false, true>(1.0, A, Bt, 0.0, &C);  // C = A * B^T

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [10, 5])
            B = einsums.create_random_tensor("B", [5, 8])
            C = einsums.zeros([10, 8], name="C")

            linalg.gemm(1.0, A, B, 0.0, C)           # C = A * B

            # Keyword arguments control transposition
            linalg.gemm(1.0, At, B, 0.0, C, trans_a=True)   # C = A^T * B
            linalg.gemm(1.0, A, Bt, 0.0, C, trans_b=True)   # C = A * B^T

Symmetric Eigendecomposition: ``syev``
======================================

Diagonalize a symmetric matrix :math:`\mathbf{AU} = \mathbf{U\Lambda}`, where :math:`\mathbf{A} = \mathbf{A}^T` and :math:`\mathbf{UU}^H = \mathbf{I}`:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            // syev works in place: A is overwritten with the eigenvectors and w receives
            // the eigenvalues, so pass a matrix you do not need to keep.
            auto A = create_random_definite<double>("A", 5);
            auto w = create_zero_tensor<double>("w", {5});

            syev(&A, &w);
            // A now holds the eigenvectors, w the eigenvalues

            // There is also a returning form, but it is constrained to a compile-time rank
            // of two, so it takes a statically ranked Tensor:
            auto S = create_random_definite<double>("S", 5);   // Tensor<double, 2>
            auto [vectors, values] = syev(S);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            # syev works in place: A is overwritten with the eigenvectors and w receives
            # the eigenvalues, so pass a matrix you do not need to keep.
            A = einsums.create_random_definite("A", 5)
            w = einsums.zeros([5], name="w")

            linalg.syev(A, w)
            # A now holds the eigenvectors, w the eigenvalues
            # w is in ascending order

Note that the input matrix is overwritten with data required to perform the eigendecomposition.

Singular Value Decomposition: ``svd``
=====================================

Perform singular value decomposition :math:`\mathbf{A} = \mathbf{U \Sigma V}^T`:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            // svd is constrained to MatrixConcept, a compile-time rank of two, so this one
            // takes a statically ranked Tensor rather than a RuntimeTensor.
            auto A = create_random_tensor<double>("A", 6, 4);
            auto [U, sigma, Vt] = svd(A);
            // U: optional 6x6 unitary matrix
            // sigma: 4-element vector of singular values
            // Vt: optional 4x4 unitary matrix (V transposed)

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [6, 4])
            U, sigma, Vt = linalg.svd(A)
            # sigma: 4 singular values

Linear Solve: ``gesv``
======================

Solve :math:`\mathbf{AX} = \mathbf{B}` for :math:`\mathbf{X}`. On exit, :math:`\mathbf{A}` will be overwritten with its LU
factorization, where the diagonal elements of the L factor are all 1, and the diagonal elements of :math:`\mathbf{A}` are the diagonal elements of the U factor,
and :math:`\mathbf{B}` will be overwritten with the value of :math:`\mathbf{X}`:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            RuntimeTensor<double> A("A", {3, 3});
            RuntimeTensor<double> B("B", {3, 2});

            // Fill A (well-conditioned) and B
            A(0, 0) = 4.0; A(0, 1) = 1.0; A(0, 2) = 0.0;
            A(1, 0) = 1.0; A(1, 1) = 5.0; A(1, 2) = 1.0;
            A(2, 0) = 0.0; A(2, 1) = 1.0; A(2, 2) = 3.0;
            B(0, 0) = 1.0; B(0, 1) = 2.0;
            B(1, 0) = 3.0; B(1, 1) = 4.0;
            B(2, 0) = 5.0; B(2, 1) = 6.0;

            gesv(&A, &B);
            // B now contains the solution X
            // A is overwritten with LU factors

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.zeros([3, 3], name="A")
            B = einsums.zeros([3, 2], name="B")

            # Fill A (well-conditioned) and B
            np.asarray(A)[...] = [[4.0, 1.0, 0.0],
                                  [1.0, 5.0, 1.0],
                                  [0.0, 1.0, 3.0]]
            np.asarray(B)[...] = [[1.0, 2.0],
                                  [3.0, 4.0],
                                  [5.0, 6.0]]

            linalg.gesv(A, B)
            # B now contains the solution X
            # A is overwritten with LU factors

``np.asarray`` on a tensor is a zero-copy view, so filling through it writes the tensor itself.
:ref:`howto-from-numpy` covers that boundary, including the direction that does copy.

.. warning::

   ``gesv`` destroys both ``A`` and ``B``. Make copies if you need the
   originals.

Matrix Inverse: ``invert``
==========================

Compute the matrix inverse. That is, find a matrix :math:`\mathbf{A}^{-1}` such that :math:`\mathbf{AA}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}`:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            // invert works in place and returns nothing.
            auto A = create_random_definite<double>("A", 4);
            invert(&A);
            // A now holds its own inverse

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            # invert works in place and returns nothing.
            A = einsums.create_random_definite("A", 4)
            linalg.invert(A)
            # A now holds its own inverse

Determinant: ``det``
====================

Compute the determinant of a matrix. This is done using the LU factorization method.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            // det carries the same MatrixConcept constraint as svd, so it takes a statically
            // ranked Tensor.
            auto A = create_random_tensor<double>("A", 3, 3);
            double d = det(A);
            println("det(A) = {}", d);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [3, 3])
            d = linalg.det(A)
            print(f"det(A) = {d}")

Norm: ``norm``
==============

Compute an induced matrix norm.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto A = create_random_tensor<double>("A", {5, 5});
            double n = norm(Norm::FROBENIUS, A);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [5, 5])
            n = linalg.norm(linalg.Norm.FROBENIUS, A)

The norm is named rather than defaulted. ``Norm::MAXABS``, ``Norm::ONE``, ``Norm::INFTY``,
``Norm::FROBENIUS`` and ``Norm::TWO`` are the choices.

Dot Product: ``dot``
====================

Compute the programmer's dot product between two vectors. That is, compute :math:`\sum_i x_iy_i`.
Use ``true_dot`` to compute the mathematician's dot product, :math:`sum_i x_i^*y_i`. For real numbers,
the two definitions are the same. For complex numbers, the two definitions will differ.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto x = create_random_tensor<double>("x", {100});
            auto y = create_random_tensor<double>("y", {100});
            double d = dot(x, y);  // x . y

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            x = einsums.create_random_tensor("x", [100])
            y = einsums.create_random_tensor("y", [100])
            d = linalg.dot(x, y)   # x . y

Rank-1 Update: ``ger``
======================

:math:`\mathbf{A} = \mathbf{A} + \alpha \mathbf{x y}^T`:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto x = create_random_tensor<double>("x", {4});
            auto y = create_random_tensor<double>("y", {5});
            auto A = create_zero_tensor<double>("A", {4, 5});

            ger(1.0, x, y, &A);  // A = x * y^T

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            x = einsums.create_random_tensor("x", [4])
            y = einsums.create_random_tensor("y", [5])
            A = einsums.zeros([4, 5], name="A")

            linalg.ger(1.0, x, y, A)   # A = x * y^T

If you need :math:`\mathbf{A} = \mathbf{A} + \alpha \mathbf{x}\mathbf{y}^H`, use ``gerc``.

QR Decomposition: ``qr``
=========================

Compute the QR decomposition of a matrix. That is, find the matrices :math:`\mathbf{Q}` and :math:`\mathbf{R}` that satisfy :math:`\mathbf{A} = \mathbf{QR}`,
where :math:`\mathbf{Q}` is a unitary matrix and :math:`\mathbf{R}` is an upper-triangular matrix.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            // qr carries the same MatrixConcept constraint as det and svd.
            auto A = create_random_tensor<double>("A", 6, 4);
            auto [Q, R] = qr(A);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [6, 4])
            Q, R = linalg.qr(A)
            # Q: 6x4 orthogonal, R: 4x4 upper triangular

.. _tutorial-linalg-graph:

The Same Operations, Captured
=============================

Everything above runs immediately. A sequence of them can be recorded into a graph instead,
optimized once and replayed, which is what an iterative method wants:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/ComputeGraph/Graph.hpp>
            #include <Einsums/ComputeGraph/Operations.hpp>

            namespace cg = einsums::compute_graph;

            auto A = create_random_tensor<double>("A", {64, 64});
            auto B = create_random_tensor<double>("B", {64, 64});

            cg::Graph graph("update");
            auto     &C = graph.create_runtime_tensor<double>("C", {64, 64}, /*intermediate=*/false);

            {
                cg::CaptureGuard guard(graph);
                cg::gemm<false, false>(1.0, A, B, 0.0, &C);
                cg::scale(0.5, &C);
            }

            graph.optimize();

            for (int iter = 0; iter < 100; ++iter) {
                graph.execute();
            }

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import einsums.graph as cg

            A = einsums.create_random_tensor("A", [64, 64])
            B = einsums.create_random_tensor("B", [64, 64])

            graph = cg.Graph("update")
            C = graph.create_tensor("C", [64, 64], intermediate=False)

            with cg.capture(graph):
                linalg.gemm(1.0, A, B, 0.0, C)
                linalg.scale(0.5, C)

            graph.optimize()

            for _ in range(100):
                graph.execute()

In C++ the ``cg::`` spellings are the graph-aware wrappers: inside a capture they record, and
outside one they run eagerly, so the same call serves both. Python has only the one spelling,
which behaves the same way.

Two rules carry over from the rest of the library. A graph-owned tensor you read after
``execute()`` needs ``intermediate=false``, or the optimizer removes the operation that fills it.
And the returning forms throw during capture, because a recorded operation has no value to hand
back until the graph runs: use ``cg::dot(&result, u, v)`` rather than ``dot(u, v)``, and the
output-argument spellings of the decompositions.

What's Next
===========

- :ref:`tutorial-views` -- Working with submatrices and slices
- :ref:`tutorial-compute-graph` -- Capturing and optimizing sequences of operations
- :ref:`tutorial-performance` -- Performance tuning
