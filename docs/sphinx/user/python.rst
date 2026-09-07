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

Capturing and optimizing
------------------------

``einsums.graph`` is the Python surface of the ComputeGraph. It captures a
sequence of operations, runs the optimization passes over the result, and
replays it:

.. code-block:: python

    import einsums.graph as cg

    g = cg.Graph("mp2")
    with cg.capture(g):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
        ...

    g.apply(cg.default_pass_manager())
    g.execute()

Beside ``default_pass_manager`` are ``analysis_pass_manager``,
``structural_pass_manager``, ``resource_pass_manager`` and
``tuning_pass_manager``, which are the phase-filtered views of the same
pipeline; the last two are what a caller runs over a graph loaded from a file.
The module builds each in place through the ``populate_*`` methods, because a
``PassManager`` holds passes it owns and cannot be returned by value.

The whole optimizer surface is reachable from Python, and the pieces a program
touches are these:

``cg.annotate(tensor, spaces=..., tag=..., graph=...)``
    Say what a tensor's axes range over and what the tensor *is*. Spaces are
    given by registered name, and a tag as a name or as a mapping carrying
    ``"name"`` plus attributes. One call, because a caller with both should not
    have two chances to make one and forget the other.

``cg.global_space_registry()`` and ``cg.private_space_registry(graph)``
    Where spaces are declared. The default is process-global, which is what
    makes a saved graph's space names resolvable when it is loaded. A program
    that does not share files takes a registry of its own, which is one line;
    a registry refuses a second declaration of one name with different content
    rather than overwriting it.

``cg.FactorizationRegistry`` and ``cg.FactorizationPass``
    Register a provider on a tag and let one generic pass substitute and
    re-associate. ``cg.MetricFitFactorization``, ``cg.ThcFactorization`` and
    ``cg.NaturalAuxiliaryFactorization`` are the providers that ship;
    ``cg.LaplaceTransform``, ``cg.BasisTruncation`` and ``cg.AxisTiling`` are
    passes a program adds directly.

``g.approximations()`` and ``g.approximation_tolerance(output)``
    What a lossy rewrite recorded: the tolerance it was asked for, the bound it
    states, the units that bound is in, and whether the number was measured or
    asserted. ``einsums.testing.assert_close(..., graph=g, output=...)`` reads
    them, so a comparison against an approximated result does not need a
    tolerance picked by hand.

``cg.save_graph``, ``cg.load_graph`` / ``cg.load_graph_into`` and ``cg.bind``
    A graph's structure, interface and approximation records, written to a file
    and bound to a fresh set of tensors on the other side. ``cg.bind`` takes the
    whole mapping at once, because a dim symbol is a constraint across slots.

A provider cannot yet be *authored* in Python: ``FactorizationProvider`` is
exposed so the registry can hand one back, and a Python subclass of it does not
dispatch back into Python.

The narrative for all of it, with a runnable tour, is
:ref:`tutorial-optimizer`.
