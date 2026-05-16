.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

============
Control Flow
============

The ComputeGraph supports two kinds of control flow nodes within a flat Graph:

- **Conditional nodes**: If-then-else based on a runtime predicate
- **Loop nodes**: While loop with convergence-based early exit

These are nodes IN the graph, not Pipeline stages. They coexist with regular
operations and respect data dependencies.

Conditional Nodes
=================

.. code-block:: cpp

   // Lambda form (preferred):
   graph.add_conditional("check",
       [&]() { return value(0) > threshold; },   // predicate
       [&]() { cg::scale(0.5, &value); },         // then
       [&]() { cg::scale(2.0, &value); }           // else (optional)
   );

   // Graph-returning form (for programmatic branch building):
   auto [then_g, else_g] = graph.add_conditional("check", predicate);
   { cg::CaptureGuard g(then_g); cg::scale(0.5, &value); }
   { cg::CaptureGuard g(else_g); cg::scale(2.0, &value); }

The predicate is a ``std::function<bool()>`` evaluated at execution time.
It can inspect tensor values, external variables, or any other state.

If the else-branch is omitted, it's a no-op when the predicate is false.

Loop Nodes
==========

.. code-block:: cpp

   // Lambda form (preferred):
   graph.add_loop("converge", 100,
       [&](size_t iter) { return energy_diff > 1e-8; },   // condition
       [&]() { cg::einsum(...); cg::scale(...); }          // body
   );

   // Graph-returning form:
   auto &body = graph.add_loop("converge", 100, condition);
   { cg::CaptureGuard g(body); cg::einsum(...); }

The loop body always executes at least once. The condition is checked after
each iteration:

- Return ``true`` → continue iterating
- Return ``false`` → exit the loop

The ``max_iterations`` parameter is a safety limit.

Mixing Control Flow with Regular Operations
============================================

.. code-block:: cpp

   cg::Graph graph("mixed");

   // Regular operation
   { cg::CaptureGuard g(graph); cg::einsum("ij <- ik ; kj", &C, A, B); }

   // Loop node
   auto &body = graph.add_loop("refine", 50, condition);
   { cg::CaptureGuard g(body); cg::scale(0.9, &C); }

   // Another regular operation
   { cg::CaptureGuard g(graph); cg::scale(factor, &C); }

   graph.execute();

The graph executes: einsum → loop (up to 50 iterations) → scale.

Conditional/Loop Nodes vs Pipeline
===================================

========================================= ===========================================
Pipeline                                  Graph Control Flow Nodes
========================================= ===========================================
Stages execute sequentially               Nodes in DAG, respect dependencies
Each stage is a separate Graph            Subgraphs embedded as node descriptors
``pipeline.add_stage()``                  ``graph.add_conditional()``
``pipeline.add_loop()``                   ``graph.add_loop()``
Best for multi-phase workflows            Best for in-graph branching/iteration
========================================= ===========================================
