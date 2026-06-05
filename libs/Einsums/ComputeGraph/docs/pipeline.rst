.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

========
Pipeline
========

The ``Pipeline`` class models multi-stage computational workflows as a linear
sequence of stages, where each stage is either a one-shot subgraph or an
iterative loop with convergence-based early exit.

Why Pipeline?
=============

Real computational workflows have structure beyond a flat DAG:

1. **Setup**: Read inputs, build orthogonalization matrices, initialize
2. **Iteration**: SCF/CC loop with convergence check
3. **Post-processing**: Compute properties, write outputs

A Pipeline captures this structure, allowing optimization passes to be applied
to each stage independently and providing clean profiler regions for each phase.

Creating a Pipeline
===================

.. code-block:: cpp

   namespace cg = einsums::compute_graph;

   // Intermediates in outer scope
   auto tmp = create_zero_tensor<double>("tmp", N, N);

   cg::Pipeline pipeline("my_workflow");

One-Shot Stages
===============

.. code-block:: cpp

   // Lambda form (preferred):
   pipeline.add_stage("setup", [&]() {
       cg::einsum(...);
       cg::scale(...);
   });

   // Graph-returning form:
   auto &stage = pipeline.add_stage("setup");
   { cg::CaptureGuard g(stage); cg::einsum(...); }

Loop Stages
===========

.. code-block:: cpp

   // Lambda form (preferred):
   pipeline.add_loop("scf_iterations", 200,
       [&](size_t iter) { return delta > 1e-8; },   // condition
       [&]() { cg::einsum(...); }                     // body
   );

   // Graph-returning form:
   auto &loop_body = pipeline.add_loop(
           "scf_iterations",     // Stage name
           200,                  // Max iterations (safety limit)
           [&](size_t iter) -> bool {
               // Called AFTER each iteration
               double delta = std::abs(energy - energy_old);
               if (delta < 1e-8) {
                   return false;   // Converged → stop
               }
               return true;       // Continue iterating
           });
       cg::CaptureGuard guard(loop_body);
       cg::einsum(...);   // Captured into the loop body
   }

The loop body graph executes repeatedly. After each iteration, the condition
function is called. It can inspect tensor values to check convergence.

Optimization
============

.. code-block:: cpp

   cg::passes::ScaleAbsorption fuse;
   cg::passes::MemoryPlanning  mem;
   cg::PassManager pm;
   pm.add<cg::passes::ScaleAbsorption>().add<cg::passes::MemoryPlanning>();
   pipeline.apply(pm);

   mem.print_report(std::cout);

Execution
=========

.. code-block:: cpp

   pipeline.execute();   // All stages, with profiler instrumentation

Each stage and loop iteration is wrapped in profiler regions automatically.

Execution with Executor
========================

Pipelines support custom executors for parallel execution:

.. code-block:: cpp

   cg::OpenMPExecutor omp;
   pipeline.execute(omp);      // Each stage uses OpenMP executor

Tensor Lifetime
===============

Tensors referenced by any stage must outlive the pipeline. Declare
shared intermediates in the outer scope:

.. code-block:: cpp

   // WRONG: tmp dies at the end of the block
   {
       auto &stage = pipeline.add_stage("setup");
       cg::CaptureGuard guard(stage);
       auto tmp = create_zero_tensor<double>("tmp", N, N);  // DANGER!
       cg::einsum(..., &tmp, ...);   // Dangling reference on replay!
   }

   // RIGHT: outer scope, outlives all stages
   auto tmp = create_zero_tensor<double>("tmp", N, N);
   {
       auto &stage = pipeline.add_stage("setup");
       cg::CaptureGuard guard(stage);
       cg::einsum(..., &tmp, ...);   // Safe
   }

Complete Example
================

.. code-block:: cpp

   // Intermediates in outer scope
   auto F_ort = create_zero_tensor<double>("F_ort", N, N);
   auto tmp   = create_zero_tensor<double>("tmp", N, N);

   cg::Pipeline pipeline("hartree_fock");

   // Setup stage
   { auto &s = pipeline.add_stage("setup"); cg::CaptureGuard g(s);
     cg::permute("ij <- ij", 0.0, &F, 1.0, H);
   }

   // SCF loop
   { auto &body = pipeline.add_loop("scf", 200, [&](size_t iter) {
         return std::abs(energy - energy_old) > 1e-8;
     });
     cg::CaptureGuard g(body);
     cg::einsum(...);
     cg::syev(&F_ort, &epsilon);
     cg::einsum(...);
   }

   // Post-processing
   { auto &s = pipeline.add_stage("post"); cg::CaptureGuard g(s);
     cg::scale(factor, &result);
   }

   cg::PassManager pm;
   pm.add<cg::passes::ScaleAbsorption>();
   pipeline.apply(pm);
   pipeline.execute();
