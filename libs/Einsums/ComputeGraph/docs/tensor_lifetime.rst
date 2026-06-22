.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

==============================
Tensor Lifetime Management
==============================

The ComputeGraph captures operations as type-erased lambdas that reference tensors
through TensorSlot indirection. All tensors used in the graph must outlive the graph
itself. This page explains the problem and the solutions.

The Problem
===========

When you create a tensor inside a capture block, it is destroyed when the block ends.
But the captured lambda still holds a reference to it:

.. code-block:: cpp

   cg::Graph graph("broken");
   {
       cg::CaptureGuard guard(graph);
       auto tmp = create_zero_tensor<double>("tmp", N, N);  // Created here
       cg::einsum(..., &tmp, ...);   // Lambda captures slot pointing to tmp
   }
   // tmp is DESTROYED here — slot has a dangling pointer!
   graph.execute();   // CRASH or undefined behavior

Solution 1: Graph-Owned Tensors (Recommended)
==============================================

Use ``graph.create_tensor()`` to create tensors owned by the graph. They live
until the graph is destroyed. An Alloc node is also inserted into the graph,
making the allocation visible to the MemoryPlanning optimization pass:

.. code-block:: cpp

   cg::Graph graph("safe");
   auto &tmp = graph.create_zero_tensor<double, 2>("tmp", N, N);  // Graph-owned!

   {
       cg::CaptureGuard guard(graph);
       cg::einsum(..., &tmp, ...);   // Safe — tmp outlives the graph
   }
   graph.execute();   // Works correctly

The API:

.. code-block:: cpp

   // Uninitialized tensor (with Alloc node)
   auto &T = graph.create_tensor<double, 2>("name", rows, cols);

   // Zero-initialized tensor (with Alloc node)
   auto &T = graph.create_zero_tensor<double, 2>("name", rows, cols);

Template parameters
^^^^^^^^^^^^^^^^^^^

- ``T``: Element type (``double``, ``float``, ``std::complex<double>``, etc.)
- ``Rank``: Number of dimensions (must be explicitly specified)
- ``Dims...``: Dimension sizes (deduced from arguments)

Solution 2: Outer-Scope Declaration
====================================

Declare tensors in a scope that outlives the graph:

.. code-block:: cpp

   auto tmp = create_zero_tensor<double>("tmp", N, N);  // Outer scope
   cg::Graph graph("safe");
   {
       cg::CaptureGuard guard(graph);
       cg::einsum(..., &tmp, ...);   // Safe — tmp in outer scope
   }
   graph.execute();   // Works

This is the approach for Pipeline intermediates. Because a Pipeline does not own
tensors, declare shared intermediates in the outer scope:

.. code-block:: cpp

   auto tmp = create_zero_tensor<double>("tmp", N, N);  // Outer scope
   cg::Pipeline pipeline("scf");
   // tmp is used across multiple stages

Runtime Validation
==================

As a safety net, ``execute()`` validates tensor
pointers before running any operations. If a tensor appears to have been
destroyed, a descriptive ``std::runtime_error`` is thrown instead of a segfault:

.. code-block:: text

   Graph 'broken': tensor 'tmp' (id=2) appears to have been destroyed.
   Ensure all tensors outlive the graph, or use graph.create_tensor()
   for intermediates.

The validation uses a name-hash canary: at registration time, the tensor's name
hash is recorded. Before execution, the hash is re-computed and compared. If
the memory has been reused or corrupted, the hashes won't match.

.. note::

   This is a best-effort check. It catches most use-after-free cases but is
   not guaranteed to detect all memory corruption.

When to Use What
=================

========================================= ==========================================
Scenario                                  Recommendation
========================================= ==========================================
Intermediate only used within graph       ``graph.create_tensor()``
Intermediate shared across pipeline       Declare in outer scope
stages
Input/output tensors (user's data)        Declare in outer scope
Return-value ops (syev, svd, qr)          Use in-place forms (``syev(&A, &W)``)
                                          with pre-allocated tensors
========================================= ==========================================
