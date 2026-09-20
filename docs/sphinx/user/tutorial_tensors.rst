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

Prerequisites
=============

.. code-block:: cpp

    #include <Einsums/Tensor/RuntimeTensor.hpp>
    #include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
    #include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
    #include <Einsums/TensorUtilities/CreateIdentity.hpp>
    #include <Einsums/Print.hpp>

    using namespace einsums;

Creating Tensors
================

:cpp:type:`einsums::RuntimeTensor` is a dense, contiguous, n-dimensional array. The first
argument is a name, which is not decoration: it is how the tensor is identified in profiler
output and in the reports the graph optimizer prints, so a meaningful one pays for itself the
first time you read either.

.. code-block:: cpp

    // A 10x10 matrix of doubles
    RuntimeTensor<double> A("A", {10, 10});

    // A rank-3 tensor of floats
    RuntimeTensor<float> B("B", {5, 6, 7});

    // A vector of complex doubles
    RuntimeTensor<std::complex<double>> v("v", {100});

The rank is the length of the dimension list, and it is carried by the object rather than by its
type. That is what lets one function handle tensors of several ranks, and it is the type the
Python bindings and the :doc:`ComputeGraph <tutorial_compute_graph>` are built around.

Convenience creators avoid repetitive initialization. Each takes the dimensions as a list and
returns a ``RuntimeTensor``:

.. code-block:: cpp

    // All zeros
    auto Z = create_zero_tensor<double>("Z", {4, 4});

    // Random values in [-1, 1]
    auto R = create_random_tensor<double>("R", {4, 4});

    // Ones on the diagonal
    auto I = create_identity_tensor<double>("I", {4, 4});

Each also has a form taking the extents as separate arguments, which returns a statically ranked
:cpp:type:`einsums::Tensor` instead. ``create_zero_tensor<double>("Z", 4, 4)`` and
``create_zero_tensor<double>("Z", {4, 4})`` are therefore different types, so write the list when
you want the type this page is about.

An identity does not have to be square. Ones sit where every index agrees, so the diagonal is as
long as the shortest axis and ``{6, 3}`` puts ones at ``(0,0)``, ``(1,1)`` and ``(2,2)``.

Element Access
==============

Use ``operator()`` with one index per dimension:

.. code-block:: cpp

    RuntimeTensor<double> A("A", {3, 3});
    A(0, 0) = 1.0;
    A(1, 2) = 3.14;

    double val = A(1, 2);  // 3.14

For a rank-1 tensor:

.. code-block:: cpp

    RuntimeTensor<double> v("v", {5});
    v(0) = 10.0;
    v(4) = 20.0;

Element access is convenient and is not how you should move bulk data. A loop over
``operator()`` pays an index computation per element; the operations in
:doc:`tutorial_einsum` hand whole tensors to BLAS instead.

Querying Shape
==============

.. code-block:: cpp

    RuntimeTensor<double> A("A", {3, 4, 5});

    size_t rank = A.rank();     // 3
    size_t d0   = A.dim(0);     // 3
    size_t d1   = A.dim(1);     // 4
    size_t d2   = A.dim(2);     // 5
    size_t n    = A.size();     // 60 (= 3 * 4 * 5)

    std::string name = A.name(); // "A"

``rank()`` is a call rather than a compile-time constant, which is the one visible cost of the
rank travelling with the value. In exchange, a function taking ``RuntimeTensor<double> const &``
accepts a matrix and a rank-4 tensor without being a template.

Raw Data Pointer
================

For interoperability with C libraries or BLAS:

.. code-block:: cpp

    double *ptr = A.data();  // Pointer to the first element

Tensors are column-major at construction, so the first index varies fastest. Code that walks
this pointer itself has to respect that; code that stays inside the tensor API does not.

Filling and Zeroing
====================

.. code-block:: cpp

    A.zero();          // Set all elements to 0
    A.set_all(3.14);   // Set all elements to 3.14

Printing
========

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", {3, 3});
    println(A);
    // Prints:
    // Name: A
    //   Dims: 3x3
    //   [data...]

You can print to a ``std::FILE *`` or a stream using ``fprintln``. To print to a string, use
``fprintln`` with a ``std::ostringstream``.

Copying
=======

Tensors copy deeply:

.. code-block:: cpp

    auto A = create_random_tensor<double>("A", {4, 4});
    auto B = A;        // deep copy

    B(0, 0) = 999.0;   // A(0, 0) is unchanged

A deep copy of a large tensor is a large allocation and a full traversal. When you want a
sub-block rather than a duplicate, take a view instead, which copies nothing: see
:doc:`tutorial_views`.

Supported Types
===============

- ``float``, ``double``
- ``std::complex<float>``, ``std::complex<double>``
- Integer types for certain operations

``double`` is the common choice for scientific computing. Where a routine is written once for
several of these, prefer to test it over all of them rather than over ``double`` alone; a
float64-only test has hidden real bugs before.

.. _tutorial-tensors-graph:

The Same Tensors, Captured
==========================

Everything above runs eagerly: each call does its work and returns. Einsums can also *record*
operations into a graph, optimize the recording, and replay it, which is where fusion, memory
planning and batching come from. None of that is available to an eager call.

The tensors are the same objects. What changes is that the graph owns the ones it creates for
you, and it has to be told which of those you will read afterwards:

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

    println(C);

``A`` and ``B`` are ordinary tensors you own, and the graph reads them. ``C`` is owned by the
graph, and the ``intermediate=false`` is what says it is an answer rather than working space.
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
