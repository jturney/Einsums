.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _computegraph_pipeline:

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

A Pipeline captures this structure. Optimization passes are applied to each
stage independently, and each stage gets its own profiler region.

Creating a Pipeline
===================

.. code-block:: cpp

   namespace cg = einsums::compute_graph;

   cg::Workspace ws("calculation");
   auto &A = ws.declare_random_tensor<double, 2>("A", n, n);

   cg::Pipeline pipeline("my_workflow");
   pipeline.set_workspace(ws);

A pipeline should own its storage scope explicitly: ``run()`` throws if no
workspace is set, because it materializes the workspace as part of the call.
Tensors shared across stages can also be declared on the pipeline itself, in
which case they are freed when the pipeline is destroyed.

One-Shot Stages
===============

.. code-block:: cpp

   // Lambda form (preferred):
   pipeline.add_stage("setup", [&]() {
       cg::einsum("ij <- ik ; kj", &C, A, B);
       cg::scale(2.0, &C);
   });

   // The body may instead take the stage Graph, for stage-scoped scratch:
   pipeline.add_stage("setup", [&](cg::Graph &g) {
       auto &tmp = g.scratch<double, 2>("tmp", n, n);
       cg::einsum("ij <- ik ; kj", &tmp, A, B);
   });

   // Graph-returning form, when you need the Graph for something else:
   auto &stage = pipeline.add_stage("setup");
   { cg::CaptureGuard g(stage); cg::einsum("ij <- ik ; kj", &C, A, B); }

Loop Stages
===========

The condition is evaluated **after** each iteration, so the body always runs at
least once. It returns ``true`` to continue and ``false`` to stop, and
``max_iterations`` is a safety limit, not a target.

.. code-block:: cpp

   // Lambda form (preferred):
   pipeline.add_loop("scf_iterations", 200,
       [&](size_t iter) { return std::abs(energy - energy_old) > 1e-8; },
       [&]() { cg::einsum("ij <- ijkl ; kl", &F, eri, D); }
   );

   // Graph-returning form:
   {
       auto &loop_body = pipeline.add_loop("scf_iterations", 200,
           [&](size_t iter) -> bool {
               return std::abs(energy - energy_old) > 1e-8;
           });
       cg::CaptureGuard guard(loop_body);
       cg::einsum("ij <- ijkl ; kl", &F, eri, D);
   }

The condition can inspect tensor values to check convergence, and it can also
update runtime parameters (see `Runtime parameters`_) that the next iteration
reads.

Runtime parameters
==================

A pipeline carries a table of integer-valued, name-keyed scalars resolved at
execute time. They are used as dynamic slice bounds for ``View`` ops, dynamic
loop limits, and anything else whose value is not known at capture:

.. code-block:: cpp

   pipeline.set_param("batch_start", 0);
   ...
   pipeline.get_param("batch_start");
   pipeline.get_param_or("batch_start", 0);

Parameters are read every time a node executes, so an update between iterations,
from a ``LoopCondition``, an external callback, or a ``WriteParam`` node, takes
effect on the next read.

Optimization passes treat parameter values as opaque. They may inspect
``BoundExpr::is_param()`` structurally, but must not branch on the runtime value.

Optimization
============

``Pipeline::apply`` runs the pass manager on every one-shot stage's Graph and
every loop stage's body Graph, and returns true if any stage was modified:

.. code-block:: cpp

   auto pm = cg::PassManager::create_default();
   pipeline.apply(pm);

   // Or a hand-built list:
   cg::PassManager pm2;
   pm2.add<cg::passes::ScaleAbsorption>()
      .add<cg::passes::MemoryPlanning>();
   pipeline.apply(pm2);

.. important::

   Install a stage's executor **before** ``apply()``. ``ScratchPrivatization``
   rewrites only graphs that carry one, on the reasoning that under the built-in
   sequential replay its clones cost cache locality and buy nothing:

   .. code-block:: cpp

      auto &body = pipeline.add_loop("scf", 200, cond);
      body.set_executor(std::make_shared<cg::DataflowExecutor>());
      pipeline.apply(pm);   // now the loop body's false dependencies are broken

Execution
=========

.. code-block:: cpp

   pipeline.execute();          // all stages, with profiler instrumentation

   cg::OpenMPExecutor omp;
   pipeline.execute(omp);       // each stage uses the given executor

Each stage and loop iteration is wrapped in a profiler region automatically.
``LoopNode::last_iteration_count`` records how many iterations actually ran.

Build, optimize, and run in one call
------------------------------------

``run()`` is the usual entry point. It applies the default pass manager,
materializes the associated workspace, and executes:

.. code-block:: cpp

   pipeline.run();          // default passes
   pipeline.run(my_pm);     // a caller-supplied pass manager

``make_pipeline`` and ``cg::run`` wrap the construction too:

.. code-block:: cpp

   cg::Workspace ws("example");
   auto &A = ws.declare_random_tensor<double, 2>("A", 8, 5);
   auto &B = ws.declare_random_tensor<double, 2>("B", 5, 6);
   auto &C = ws.declare_zero_tensor<double, 2>("C", 8, 6);

   cg::run("compute", ws, [&](cg::Pipeline &p) {
       p.add_stage("multiply", [&] {
           cg::einsum("ij <- ik ; kj", &C, A, B);
       });
   });
   // C is now populated.

``cg::run`` destroys the pipeline when the call returns, so any result the caller
wants to read must be declared on the workspace, which outlives the call. Use
``cg::make_pipeline`` instead when you want to keep the pipeline around to
inspect or re-execute.

Inspecting a pipeline
=====================

.. code-block:: cpp

   pipeline.name();               // pipeline name
   pipeline.num_stages();         // stage count
   pipeline.stage_name(i);        // name of stage i
   pipeline.stage_graph(i);       // Graph* for stage i (loop stages give the body)
   pipeline.workspace();          // associated Workspace*, or nullptr

``stage_graph(i)`` is how you reach a single stage to apply a pass to it alone,
read ``graph.explain()`` for it, or install a per-stage executor.

Tensor Lifetime
===============

Tensors referenced by any stage must outlive the pipeline. Declare shared
intermediates on the workspace, on the pipeline, or in the outer scope:

.. code-block:: cpp

   // WRONG: tmp dies at the end of the block
   {
       auto &stage = pipeline.add_stage("setup");
       cg::CaptureGuard guard(stage);
       auto tmp = create_zero_tensor<double>("tmp", N, N);  // DANGER!
       cg::einsum("ij <- ik ; kj", &tmp, A, B);   // Dangling reference on replay!
   }

   // RIGHT: pipeline-scoped, deferred, and visible to the memory passes
   auto &tmp = pipeline.declare_zero_tensor<double, 2>("tmp", N, N);
   pipeline.add_stage("setup", [&] {
       cg::einsum("ij <- ik ; kj", &tmp, A, B);
   });

For scratch that only lives within one stage, use the ``(Graph &)`` body form and
``g.scratch()``: the tensor is graph-owned, so ``FreeInsertion``,
``InplaceOptimization`` and ``MemoryPlanning``'s arena can all act on it. See
:doc:`workspace` for the three scoping levels and what each buys.

Complete Example
================

.. code-block:: cpp

   namespace cg = einsums::compute_graph;

   cg::Workspace ws("hartree_fock");
   auto &eri = ws.declare_tensor<double, 4>("ERI", nao, nao, nao, nao);

   cg::Pipeline pipeline("hartree_fock");
   pipeline.set_workspace(ws);

   auto &F     = pipeline.declare_zero_tensor<double, 2>("F", nao, nao);
   auto &D     = pipeline.declare_zero_tensor<double, 2>("D", nao, nao);
   auto &F_ort = pipeline.declare_zero_tensor<double, 2>("F_ort", nao, nao);

   pipeline.add_stage("setup", [&] {
       cg::permute("ij <- ij", 0.0, &F, 1.0, H);
   });

   pipeline.add_loop("scf", 200,
       [&](size_t iter) { return std::abs(energy - energy_old) > 1e-8; },
       [&] {
           cg::einsum("ij <- ijkl ; kl", &F, eri, D);
           cg::syev(&F_ort, &epsilon);
           cg::einsum("ij <- ik ; jk", &D, C_occ, C_occ);
       });

   pipeline.add_stage("post", [&] {
       cg::scale(factor, &F);
   });

   pipeline.run();   // default passes + materialize workspace + execute
