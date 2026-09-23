..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-tensors:

***********************
Tutorial: Tensors 101
***********************

This tutorial introduces the data structure the rest of Einsums is built on, and how to create,
inspect and manipulate one.

Every example is given in both languages. Pick a tab and the rest of the page, and the other
tutorials, stay in that language.

Prerequisites
=============

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/Tensor/RuntimeTensor.hpp>
            #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
            #include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
            #include <Einsums/TensorUtilities/CreateIdentity.hpp>
            #include <Einsums/Print.hpp>

            using namespace einsums;

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import numpy as np
            import einsums

Creating Tensors
================

:cpp:type:`einsums::RuntimeTensor` is a dense, contiguous, n-dimensional array. The first
argument is a name, which is not decoration: it is how the tensor is identified in profiler
output and in the reports the graph optimizer prints, so a meaningful one pays for itself the
first time you read either.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            // A 10x10 matrix of doubles
            RuntimeTensor<double> A("A", {10, 10});

            // A rank-3 tensor of floats
            RuntimeTensor<float> B("B", {5, 6, 7});

            // A vector of complex doubles
            RuntimeTensor<std::complex<double>> v("v", {100});

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            # A 10x10 matrix of doubles
            A = einsums.zeros([10, 10], name="A")

            # A rank-3 tensor of floats
            B = einsums.zeros([5, 6, 7], dtype="float32", name="B")

            # A vector of complex doubles
            v = einsums.zeros([100], dtype="complex128", name="v")

The rank is the length of the dimension list, and it is carried by the object rather than by its
type. That is what lets one function handle tensors of several ranks, and it is the type the
Python bindings and the :doc:`ComputeGraph <tutorial_compute_graph>` are built around. In C++ the
element type is a template argument; in Python it is the ``dtype`` string.

Convenience creators avoid repetitive initialization. Each takes the dimensions as a list:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto Z = create_zero_tensor<double>("Z", {4, 4});      // all zeros
            auto R = create_random_tensor<double>("R", {4, 4});    // random in [-1, 1]
            auto I = create_identity_tensor<double>("I", {4, 4});  // ones on the diagonal

        Each also has a form taking the extents as separate arguments, which returns a statically
        ranked :cpp:type:`einsums::Tensor` instead. ``create_zero_tensor<double>("Z", 4, 4)`` and
        ``create_zero_tensor<double>("Z", {4, 4})`` are therefore different types, so write the
        list when you want the type this page is about.

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            Z = einsums.create_zero_tensor("Z", [4, 4])      # all zeros
            R = einsums.create_random_tensor("R", [4, 4])    # random in [-1, 1]
            I = einsums.create_identity_tensor("I", [4, 4])  # ones on the diagonal

            # The numpy-shaped spellings are there too
            Z2 = einsums.zeros([4, 4])
            O  = einsums.ones([4, 4])
            I2 = einsums.eye(4)

        Python has only the runtime-rank form, so the C++ distinction between a list and separate
        extents does not arise.

An identity does not have to be square. Ones sit where every index agrees, so the diagonal is as
long as the shortest axis and ``{6, 3}`` puts ones at ``(0,0)``, ``(1,1)`` and ``(2,2)``.

Element Access
==============

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            RuntimeTensor<double> A("A", {3, 3});
            A(0, 0) = 1.0;
            A(1, 2) = 3.14;

            double val = A(1, 2);  // 3.14

            RuntimeTensor<double> v("v", {5});
            v(0) = 10.0;
            v(4) = 20.0;

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.zeros([3, 3], name="A")
            A[0, 0] = 1.0
            A[1, 2] = 3.14

            val = A[1, 2]   # 3.14

            v = einsums.zeros([5], name="v")
            v[0] = 10.0
            v[4] = 20.0

            # Negative indices wrap, as elsewhere in Python
            assert A[-1, -1] == A[2, 2]

Element access is convenient and is not how you should move bulk data. A loop over single
elements pays an index computation each time; the operations in :doc:`tutorial_einsum` hand whole
tensors to BLAS instead.

Querying Shape
==============

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            RuntimeTensor<double> A("A", {3, 4, 5});

            size_t rank = A.rank();     // 3
            size_t d0   = A.dim(0);     // 3
            size_t d1   = A.dim(1);     // 4
            size_t d2   = A.dim(2);     // 5
            size_t n    = A.size();     // 60 (= 3 * 4 * 5)

            std::string name = A.name(); // "A"

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.zeros([3, 4, 5], name="A")

            rank = A.rank()      # 3, a method
            d0   = A.dim(0)      # 3, a method
            n    = A.size        # 60, a property
            name = A.name        # "A", a property

            shape = np.asarray(A).shape   # (3, 4, 5)

The two languages differ here in a way worth knowing. C++ spells all of these as calls; Python
makes ``size`` and ``name`` properties while ``rank()``, ``dim()`` and ``stride()`` stay methods.
There is no ``dims`` or ``strides`` on the Python side, so reach for ``np.asarray(t).shape`` when
you want the whole shape at once.

``rank()`` is a call rather than a compile-time constant, which is the one visible cost of the
rank travelling with the value. In exchange, a function taking a tensor accepts a matrix and a
rank-4 tensor without being a template.

Filling and Zeroing
====================

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            A.zero();          // Set all elements to 0
            A.set_all(3.14);   // Set all elements to 3.14

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A.zero()           # Set all elements to 0
            A.set_all(3.14)    # Set all elements to 3.14

Printing
========

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto A = create_random_tensor<double>("A", {3, 3});
            println(A);
            // Name: A
            //   Dims: 3x3
            //   [data...]

        You can print to a ``std::FILE *`` or a stream using ``fprintln``. To print to a string,
        use ``fprintln`` with a ``std::ostringstream``.

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [3, 3])
            print(A)
            # RuntimeTensorD(name='A', shape=(3, 3), dtype=float64)

            # For the values themselves, go through numpy
            print(np.asarray(A))

        ``print`` gives the summary rather than the elements, which is what you want when a
        tensor is large. ``np.asarray`` is a zero-copy view, so printing it costs nothing extra.

Copying
=======

This is the one place where the two languages behave differently enough to catch you out.

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            auto A = create_random_tensor<double>("A", {4, 4});
            auto B = A;        // DEEP COPY

            B(0, 0) = 999.0;   // A(0, 0) is unchanged

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            A = einsums.create_random_tensor("A", [4, 4])

            B = A              # NOT a copy: another name for the same tensor
            B[0, 0] = 999.0    # A[0, 0] is now 999.0 as well

            C = einsums.array(A)   # this copies
            C[1, 1] = 111.0        # A[1, 1] is unchanged

        Assignment binds a name, as everywhere in Python. ``einsums.array`` is the copy, matching
        ``numpy.array``. ``copy.deepcopy`` is not supported and raises ``TypeError``.

A deep copy of a large tensor is a large allocation and a full traversal. When you want a
sub-block rather than a duplicate, take a view instead, which copies nothing: see
:doc:`tutorial_views`.

Supported Types
===============

- ``float``, ``double``
- ``std::complex<float>``, ``std::complex<double>``
- Integer types for certain operations

``double`` is the common choice for scientific computing, and is what both languages default to.
In Python these are the ``dtype`` strings ``"float32"``, ``"float64"``, ``"complex64"`` and
``"complex128"``.

Where a routine is written once for several of these, prefer to test it over all of them rather
than over ``double`` alone; a float64-only test has hidden real bugs before. ``einsums.testing``
provides ``ALL_DTYPES`` for exactly that.

.. _tutorial-tensors-graph:

The Same Tensors, Captured
==========================

Everything above runs eagerly: each call does its work and returns. Einsums can also *record*
operations into a graph, optimize the recording, and replay it, which is where fusion, memory
planning and batching come from. None of that is available to an eager call.

The tensors are the same objects. What changes is that the graph owns the ones it creates for
you, and it has to be told which of those you will read afterwards:

.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <Einsums/ComputeGraph/Graph.hpp>
            #include <Einsums/ComputeGraph/Operations.hpp>

            namespace cg = einsums::compute_graph;

            auto A = create_random_tensor<double>("A", {4, 4});
            auto B = create_random_tensor<double>("B", {4, 4});

            cg::Graph graph("example");

            // intermediate = false: a result this code reads after execute(), not scratch.
            auto &C = graph.create_runtime_tensor<double>("C", {4, 4}, /*intermediate=*/false);

            {
                cg::CaptureGuard guard(graph);
                cg::einsum("ik;kj->ij", &C, A, B);
            }

            graph.optimize();
            graph.execute();       // replay as often as you like

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            import einsums.graph as cg

            A = einsums.create_random_tensor("A", [4, 4])
            B = einsums.create_random_tensor("B", [4, 4])

            graph = cg.Graph("example")

            # intermediate=False: a result this code reads after execute(), not scratch.
            C = graph.create_tensor("C", [4, 4], intermediate=False)

            with cg.capture(graph):
                einsums.einsum("ik;kj->ij", C, A, B)

            graph.optimize()
            graph.execute()        # replay as often as you like

``A`` and ``B`` are ordinary tensors you own, and the graph reads them. ``C`` is owned by the
graph, and the ``intermediate`` flag is what says it is an answer rather than working space.
Leave it off and the optimizer will remove the contraction that fills it, because from inside the
graph nothing reads it.

That flag is the one mistake in this tutorial that can pass without an error, so it is worth
meeting here rather than later. :ref:`howto-graphs-intermediate` explains what the optimizer
reports when it happens.

What's Next
===========

- :ref:`tutorial-einsum` for tensor contractions
- :ref:`tutorial-views` for looking at part of a tensor without copying it
- :ref:`tutorial-linalg` for decompositions and solves
- :ref:`tutorial-compute-graph` for what capturing buys once the operations get interesting
