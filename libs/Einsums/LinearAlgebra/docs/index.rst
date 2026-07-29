..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_LinearAlgebra:

Linear Algebra
==============

This module contains the user-facing definitions of various linear algebra routines.

See the :ref:`API reference <modules_Einsums_LinearAlgebra_api>` of this module for more
details.

Public API
----------

Here are some of the public symbols that are available to use. More can be found in the :ref:`API reference <modules_Einsums_LinearAlgebra_api>`.

- :cpp:func:`~einsums::linear_algebra::sum_square` - accumulates the scaled sum of squares of a tensor's elements.
- :cpp:func:`~einsums::linear_algebra::gemm` - general matrix multiplication, with transposes as template booleans, as runtime characters, or in a form that returns the product as a new tensor.
- :cpp:func:`~einsums::linear_algebra::symm_gemm` - computes the double product :math:`OP(B)^T OP(A) OP(B) = C`.
- :cpp:func:`~einsums::linear_algebra::gemv` - general matrix-vector multiplication.
- :cpp:func:`~einsums::linear_algebra::syev` - eigendecomposition of a real symmetric matrix.
- :cpp:func:`~einsums::linear_algebra::heev` - eigendecomposition of a complex Hermitian matrix.
- :cpp:func:`~einsums::linear_algebra::geev` - eigendecomposition of a general matrix.
- :cpp:func:`~einsums::linear_algebra::gesv` - solves a system of linear equations.
- :cpp:func:`~einsums::linear_algebra::scale` - scales a tensor by a scalar value.
- :cpp:func:`~einsums::linear_algebra::scale_row` - scales a row of a matrix.
- :cpp:func:`~einsums::linear_algebra::scale_column` - scales a column of a matrix.
- :cpp:func:`~einsums::linear_algebra::pow` - raises a matrix to a power through its eigendecomposition.
- :cpp:func:`~einsums::linear_algebra::dot` - dot product of two or three tensors, without conjugation.
- :cpp:func:`~einsums::linear_algebra::true_dot` - dot product that conjugates the first tensor if complex.
- :cpp:func:`~einsums::linear_algebra::axpy` - scaled tensor addition :math:`\mathbf{Y} = \alpha\mathbf{X} + \mathbf{Y}`.
- :cpp:func:`~einsums::linear_algebra::axpby` - scaled tensor addition :math:`\mathbf{Y} = \alpha\mathbf{X} + \beta\mathbf{Y}`.
- :cpp:func:`~einsums::linear_algebra::ger` - rank-1 update :math:`\mathbf{A} = \alpha\mathbf{XY}^T + \mathbf{A}`.
- :cpp:func:`~einsums::linear_algebra::gerc` - rank-1 update conjugating the right vector, :math:`\mathbf{A} = \alpha\mathbf{XY}^H + \mathbf{A}`.
- :cpp:func:`~einsums::linear_algebra::invert` - inverts a matrix in place using getrf and getri.
- :cpp:func:`~einsums::linear_algebra::direct_product` - element-wise (Hadamard) product with accumulation.
- :cpp:func:`~einsums::linear_algebra::det` - determinant of a matrix.
