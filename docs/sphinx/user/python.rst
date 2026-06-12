..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

*****************
Einsums in Python
*****************

Einsums ships first-class Python bindings (``import einsums``). The Python
surface is generated directly from the C++ headers, so it tracks the library
as it evolves; see the :ref:`Python API reference <api_python>` for the full
list of modules, classes, and functions.

This page covers the basics: creating tensors, contracting them, and moving
data to and from NumPy.

Running Einsums
---------------

To use Einsums, import it:

.. code-block:: python

    import einsums

Einsums interoperates with anything that implements the Python buffer
protocol, most importantly :code:`numpy.ndarray`, and provides its own
runtime tensor types for the C++ side of the library. The simplest way to
multiply two matrices is the ``@`` operator, which dispatches to a BLAS
``gemm``, or is recorded into the active graph when used inside
``cg.capture``:

.. code-block:: python

    import einsums
    import numpy as np

    A = einsums.create_random_tensor("A", [3, 3])
    B = einsums.create_random_tensor("B", [3, 3])

    C = A @ B                      # matrix product, via einsums.linalg.gemm
    print(np.asarray(C))           # zero-copy view of the result as a NumPy array

For explicit index contractions, :py:func:`einsums.einsum` takes a spec of
the form ``"out <- lhs ; rhs"`` and writes into a pre-allocated output. The
matrix product above is ``"ij <- ik ; kj"``:

.. code-block:: python

    C = einsums.create_zero_tensor("C", [3, 3], dtype="float64")
    einsums.einsum("ij <- ik ; kj", C, A, B)

and the lower-level :py:func:`einsums.linalg.gemm` is available when you want
to control the scalar prefactors directly:

.. code-block:: python

    # C = alpha * A @ B + beta * C
    einsums.linalg.gemm(1.0, A, B, 0.0, C)

Creating tensors
----------------

Einsums exposes one runtime tensor class per scalar type, named by the
BLAS-style letter:

* ``RuntimeTensorF``: 32-bit real, matching ``numpy.single`` / ``float32``
* ``RuntimeTensorD``: 64-bit real, matching Python ``float`` / ``numpy.double``
* ``RuntimeTensorC``: 64-bit complex, matching ``numpy.complex64``
* ``RuntimeTensorZ``: 128-bit complex, matching Python ``complex`` / ``numpy.complex128``

plus the matching ``RuntimeTensorView{F,D,C,Z}`` view types and tiled
variants. Extended and half precision are not provided. The usual
constructors live at the package top level:

.. code-block:: python

    A = einsums.create_random_tensor("A", [3, 3])          # filled with random data
    C = einsums.create_zero_tensor("C", [3, 3], dtype="float64")
    M = einsums.array(np.eye(3))                            # copy a NumPy array in

Slicing a tensor produces a zero-copy view rather than a copy:

.. code-block:: python

    A_view = A[0:2, 0:2]           # a RuntimeTensorViewD aliasing A's storage
    print(type(A))                 # <class 'einsums.RuntimeTensorD'>
    print(type(A_view))            # <class 'einsums.RuntimeTensorViewD'>

NumPy ergonomics
----------------

Runtime tensors and views carry the NumPy-style attributes array users
expect, including ``.shape``, ``.ndim``, ``.dtype``, ``.T``, ``len(t)``, ``t.copy()``,
``t.transpose(...)``, reductions (``sum``/``mean``/``max``), the ``@``
operator, and the arithmetic operators, which are built on top of the buffer protocol
so ``np.asarray(t)`` is zero-copy. These are documented on the
:ref:`tensor ergonomics page <einsums_tensor_ergonomics>`.

.. code-block:: python

    A.shape            # (3, 3)
    A.T                # zero-copy transpose view
    np.asarray(A)      # zero-copy NumPy view; np.array(A) to force a copy

.. note::

   ``@`` and the arithmetic operators stay on Einsums code paths and dispatch
   to :py:mod:`einsums.linalg` rather than falling through to
   NumPy, so they compose inside a captured :ref:`ComputeGraph
   <modules_Einsums_ComputeGraph>` workflow.
