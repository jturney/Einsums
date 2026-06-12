.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

========
Executor
========

The Executor abstraction decouples graph structure from execution strategy.
Built-in executors:

- **SequentialExecutor**: Nodes run one at a time in topological order (default)
- **OpenMPExecutor**: Independent nodes run in parallel via OpenMP wavefront
- **DataflowExecutor**: Maximum overlap via TaskPool continuations (no barriers)
- **MPIExecutor**: All ranks execute the same graph; collective nodes synchronize

Using an Executor
=================

.. code-block:: cpp

   cg::Graph graph("example");
   // ... capture operations ...

   // Default: sequential
   graph.execute();

   // Explicit sequential
   cg::SequentialExecutor seq;
   graph.execute(seq);

   // Parallel: OpenMP wavefront
   cg::OpenMPExecutor omp;
   graph.execute(omp);

   // Pipeline with executor
   cg::Pipeline pipeline("scf");
   // ... add stages ...
   pipeline.execute(omp);

How OpenMPExecutor Works
========================

The OpenMP executor uses wavefront parallelism:

1. Assigns each node a level based on its dependencies:
   - Level 0: nodes with no predecessors (independent)
   - Level N: nodes whose latest predecessor is at level N-1
2. Executes levels sequentially, while nodes within a level run in parallel

The parallelism is detected automatically from data dependencies, so the user
does not need to restructure their code.

.. code-block:: cpp

   {
       cg::CaptureGuard guard(graph);
       cg::einsum("ij <- ik ; kj", &C, A, B);   // Level 0
       cg::einsum("ij <- ik ; kj", &F, D, E);   // Level 0 (independent!)
       cg::scale(2.0, &C);                       // Level 1 (depends on C)
       cg::axpy(1.0, C, &F);                     // Level 2 (depends on C and F)
   }
   // Level 0: C=A*B and F=D*E run in PARALLEL
   // Level 1: scale(C) runs after C is ready
   // Level 2: axpy runs after both C and F are ready

DataflowExecutor (TaskPool)
============================

The ``DataflowExecutor`` uses the TaskPool module for true dataflow execution.
Instead of wavefront barriers, each node is submitted as a TaskPool task
that runs when all its predecessors complete via continuation chaining:

.. code-block:: cpp

   cg::DataflowExecutor df;
   graph.execute(df);  // Maximum overlap, no barriers

**How it works:**

1. Each node gets a ``TaskHandle<void>`` representing its completion
2. Nodes with no predecessors are submitted immediately
3. Nodes with predecessors use ``when_all()`` + ``dataflow()`` to wait
   for all predecessors before executing
4. The work-stealing pool naturally load-balances across workers

**When to use:**

- Use ``DataflowExecutor`` when your graph has many independent nodes,
  async I/O nodes, and you want maximum overlap
- Use ``OpenMPExecutor`` for simpler graphs where wavefront parallelism
  is sufficient
- Use ``SequentialExecutor`` for debugging or when nodes internally use
  OpenMP/TaskPool

Async I/O Overlap
==================

The ``DataflowExecutor`` has special support for asynchronous I/O nodes
created with ``cg::read_async()`` / ``cg::write_async()``. These nodes have
two execution phases:

1. **Start**: Begins the I/O operation (e.g., submits an async read request)
2. **Finish**: Waits for the I/O to complete before consumers run

Between start and finish, independent compute nodes can execute, overlapping
I/O latency with useful computation:

.. code-block:: text

   Time ──────────────────────────────────────────────────►

   read/start ═══════ [I/O in progress] ═══════ read/finish
                  ↑                                  ↑
               compute A*B runs here              consumer
               (independent of read)              runs after
                                                  finish

To set up async I/O:

.. code-block:: cpp

   std::future<void> io_future;

   cg::Graph graph("pipeline");
   {
       cg::CaptureGuard guard(graph);

       // Async read: starts I/O immediately, finishes when consumer needs data
       cg::read_async("load integrals", "integrals.h5", "/eri", &ERI,
           [&]() { io_future = std::async(std::launch::async,
                      [&] { load_from_disk(ERI); }); },
           [&]() { io_future.get(); },
           [&]() { load_from_disk(ERI); }  // sync fallback
       );

       // Independent computation — overlaps with I/O
       cg::einsum("ik;kj->ij", &C, A, B);

       // Consumer of ERI — waits for read to finish
       cg::einsum("ikjl;kl->ij", &F, ERI, D);
   }

   // IOPrefetch moves the read to position 0, maximizing overlap window
   auto pm = cg::PassManager::create_default();
   graph.apply(pm);

   // DataflowExecutor enables the async overlap
   cg::DataflowExecutor df;
   graph.execute(df);

**Note**: ``SequentialExecutor`` and ``OpenMPExecutor`` call the synchronous
fallback lambda, so no overlap occurs. Use ``DataflowExecutor`` for async
I/O overlap.

**Graph-capturable parallel_for and parallel_reduce:**

``cg::parallel_for()`` and ``cg::parallel_reduce()`` capture into the graph
as nodes, with dependency tracking via declared tensor outputs. The graph's
topological sort ensures they run in the correct order relative to other
operations:

.. code-block:: cpp

   cg::Graph graph("fock_build");
   {
       cg::CaptureGuard guard(graph);

       // Node 1: parallel_for computes J and K (via TaskPool)
       cg::parallel_for("integrals", 0, n_pairs,
           [&](size_t pair) { compute_integrals(pair, J, K); },
           &J, &K);  // Declare outputs for dependency tracking

       // Node 2: F = H + 2*J - K (einsum/axpy — depends on J, K)
       cg::permute("ij <- ij", 0.0, &F, 1.0, H);
       cg::axpy(2.0, J, &F);
       cg::axpy(-1.0, K, &F);

       // Node 3: parallel_reduce for energy (depends on F)
       cg::parallel_reduce<double>("energy", 0, N*N, &energy,
           []() { return 0.0; },
           [&](size_t i, double &acc) { acc += D(i) * F(i); },
           [](double &g, double const &l) { g += l; },
           &D, &F);  // Declare inputs for dependency tracking
   }

   graph.execute();  // Automatically ordered: integrals → assembly → energy

The tensor arguments at the end of ``cg::parallel_for()`` and
``cg::parallel_reduce()`` declare which tensors the body reads/writes,
enabling the graph's topological sort to order these nodes correctly.

**Manual TaskPool usage inside nodes:**

You can also call TaskPool directly from within a graph node's lambda. In this
case there is no automatic dependency tracking. The ``parallel_for`` blocks
before returning, so the next node sees the completed results:

.. code-block:: cpp

   auto &pool = tp::TaskPool::get_singleton();
   pool.parallel_for("integral_batches", 0, num_pairs, [&](size_t pair) {
       compute_integrals(pair);
   });

The graph handles coarse-grained operation ordering; the TaskPool handles
fine-grained parallelism within each operation.

Writing a Custom Executor
=========================

.. code-block:: cpp

   class MyExecutor : public cg::Executor {
   public:
       std::string name() const override { return "MyExecutor"; }
       void execute(cg::Graph &graph) override {
           auto &nodes = graph.nodes();
           auto const &deps = graph.dependencies();
           // ... custom execution strategy ...
       }
   };
