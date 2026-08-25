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

Predicates as Data
==================

A ``std::function`` predicate is opaque: nothing can inspect it, and nothing can
write it to a file.
``PredExpr`` is the same condition expressed as data, and both ``add_conditional``
and ``add_loop`` accept one wherever they accept a callable.

.. code-block:: cpp

   // A comparison over pipeline parameters.
   auto [then_g, else_g] = graph.add_conditional(
       "converged", cg::PredExpr::compare(cg::BoundExpr{"delta"}, cg::CmpOp::Lt, cg::BoundExpr{1}));

   // A comparison against the index of the iteration that just finished.
   auto &body = graph.add_loop("iter", 100, cg::PredExpr::iteration(cg::CmpOp::Lt, cg::BoundExpr{9}));

   // One slot of a GateFlags array. This is what add_conditional_flag builds.
   auto [gated, _] = graph.add_conditional("block", cg::PredExpr::flag(flags, block_index));

The arms are a literal, a comparison over two ``BoundExpr`` operands, a
comparison against the loop iteration index, a gate-flag load, and a callback.
Only the callback arm holds a closure, and it is the only one that stops a node
from being reconstructed from its descriptor alone; ``Graph::serializability_report()``
names such a node individually.
A conditional or a loop whose predicate names a parameter is ordered against
whatever writes that parameter, the same way a view with parametric bounds is.

The iteration index is deliberately NOT a reserved parameter name.
A ``ParamTable`` is a flat namespace shared by the whole graph, so a loop-private
counter placed in it would collide between nested loops and would be readable by
nodes that have no business seeing it.

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
