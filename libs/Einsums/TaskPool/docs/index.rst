..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_TaskPool:

========
TaskPool
========

The TaskPool module provides fine-grained data-parallel task execution with
work-stealing, continuations, and dataflow dependency management. It is
designed for workloads like quantum chemistry integral generation where
thousands of independent tasks must run in parallel with automatic load
balancing.

Inspired by `HPX <https://github.com/STEllAR-GROUP/hpx>`_, the TaskPool
provides:

- A work-stealing thread pool built on lock-free Chase-Lev deques.
- Continuations through ``TaskHandle::then()`` for composing async pipelines.
- Fan-in through ``when_all()`` for waiting on multiple tasks.
- Dataflow through ``dataflow()`` for dependency-driven task scheduling.
- Data-parallel patterns: ``parallel_for()`` and ``parallel_reduce()``.
- Profiler integration with per-task regions and metrics export.
- ComputeGraph integration through ``DataflowExecutor``.

Getting Started
===============

.. code-block:: cpp

    #include <Einsums/TaskPool/TaskPool.hpp>

    namespace tp = einsums::task_pool;

    auto &pool = tp::TaskPool::get_singleton();

    // Submit a task
    auto handle = pool.submit("compute", []() { return 42; });
    int result = handle.get();  // 42

The TaskPool is a singleton that creates worker threads at first access.
Thread count defaults to ``std::hardware_concurrency()`` or
``OMP_NUM_THREADS``.

Submitting Tasks
================

Basic submit
------------

.. code-block:: cpp

    // Returns TaskHandle<int>
    auto h = pool.submit("my_task", []() { return compute_something(); });
    int val = h.get();  // Blocks until done

    // Void tasks
    auto v = pool.submit("side_effect", [&]() { update_state(); });
    v.wait();  // Blocks until done

Task groups
-----------

Submit multiple tasks with a collective barrier:

.. code-block:: cpp

    std::vector<std::function<void()>> tasks;
    for (int i = 0; i < 100; i++) {
        tasks.push_back([&, i]() { process_batch(i); });
    }
    auto group = pool.submit_group("batch_work", std::move(tasks));
    group.wait_all();  // All 100 tasks complete

Continuations
=============

Chain computations with ``.then()``:

.. code-block:: cpp

    auto result = pool.submit("step1", []() { return 10; })
                      .then("step2", [](int v) { return v * 2; })
                      .then("step3", [](int v) { return v + 3; });

    int val = result.get();  // (10 * 2) + 3 = 23

Each ``.then()`` submits the continuation to the TaskPool when the
predecessor completes. The continuation receives the predecessor's result.

Fan-In with when_all
====================

Wait for multiple tasks to complete:

.. code-block:: cpp

    std::vector<tp::TaskHandle<double>> tasks;
    for (int i = 0; i < 10; i++) {
        tasks.push_back(pool.submit("compute", [i]() {
            return heavy_computation(i);
        }));
    }

    auto combined = tp::when_all(std::move(tasks));
    std::vector<double> results = combined.get();

``when_all`` also supports void handles for synchronization without results.

Variadic when_all (mixed types)
-------------------------------

For handles of different types, use the variadic form which returns a
``TaskHandle<std::tuple<...>>``:

.. code-block:: cpp

    auto a = pool.submit("int_val", []() { return 42; });
    auto b = pool.submit("dbl_val", []() { return 3.14; });
    auto c = pool.submit("str_val", []() { return std::string("hello"); });

    auto combined = tp::when_all(a, b, c);
    auto [x, y, z] = combined.get();
    // x=42, y=3.14, z="hello"

This is fully asynchronous, with no thread blocking to wait for individual handles.

Dataflow
========

The most powerful pattern: a task runs only when all its input futures
are ready, without manual synchronization. The callable receives the
results of its input handles as arguments:

.. code-block:: cpp

    // Two independent computations
    auto matrix = pool.submit("compute_matrix", [&]() {
        return build_matrix(data);  // returns Matrix
    });

    auto vector = pool.submit("compute_vector", [&]() {
        return build_vector(data);  // returns Vector
    });

    // Runs when BOTH matrix and vector are ready (fully async, no blocking)
    auto result = pool.dataflow("solve",
        [](Matrix const &A, Vector const &b) {
            return solve(A, b);  // Returns Solution
        }, matrix, vector);

    auto solution = result.get();

Multi-stage pipelines using dataflow:

.. code-block:: cpp

    // Stage 1: compute screening bounds
    auto screening = pool.submit("screen", [&]() {
        return compute_bounds(basis);
    });

    // Stage 2: depends on screening (single typed input)
    auto integrals = pool.dataflow("integrals",
        [&](ScreeningData const &bounds) {
            return compute_integrals(bounds);
        }, screening);

    // Stage 3: depends on integrals
    auto fock = pool.dataflow("fock",
        [&](IntegralData const &ints) {
            return assemble_fock(ints);
        }, integrals);

    auto result = fock.get();  // Runs the whole pipeline

parallel_for
============

Data-parallel loop with automatic chunking and work-stealing:

.. code-block:: cpp

    std::vector<double> data(N);
    pool.parallel_for("fill", 0, N, [&data](size_t i) {
        data[i] = std::sin(static_cast<double>(i));
    });

The range ``[0, N)`` is split into chunks (default: 4x oversubscription
for load balance). The calling thread participates in work-stealing
while waiting.

Execution policies
------------------

Control parallel behavior:

.. code-block:: cpp

    using namespace tp::execution;

    pool.parallel_for(par, "parallel", 0, N, body);    // Parallel (default)
    pool.parallel_for(seq, "sequential", 0, N, body);  // Sequential (for debugging)

    // Custom chunk size
    pool.parallel_for(ParallelPolicy{.chunk_size = 64}, "chunked", 0, N, body);

Graph-capturable parallel_for
-----------------------------

``cg::parallel_for()`` captures a parallel_for as a ComputeGraph node
with automatic dependency tracking:

.. code-block:: cpp

    namespace cg = einsums::compute_graph;

    cg::Graph graph("example");
    {
        cg::CaptureGuard guard(graph);
        cg::parallel_for("fill", 0, N,
            [&data](size_t i) { data[i] = compute(i); },
            &data_tensor);  // Declare output for dependency ordering
        cg::einsum(...);    // Automatically ordered after parallel_for
    }

See the ComputeGraph documentation for details.

parallel_reduce
===============

Thread-local accumulation with sequential merge:

.. code-block:: cpp

    double dot = pool.parallel_reduce<double>(
        "dot_product", 0, N,
        []() { return 0.0; },                                  // init accumulator
        [&](size_t i, double &acc) { acc += a[i] * b[i]; },    // body
        [](double &global, double const &local) { global += local; }  // combine
    );

Each worker gets its own accumulator (cache-line aligned to prevent false
sharing). After all chunks complete, accumulators are merged sequentially.

ComputeGraph Integration
=========================

DataflowExecutor
----------------

The ``DataflowExecutor`` executes ComputeGraph nodes via TaskPool
continuations, giving maximum overlap between independent nodes:

.. code-block:: cpp

    namespace cg = einsums::compute_graph;

    cg::Graph graph("example");
    { /* capture operations */ }

    cg::DataflowExecutor df;
    graph.execute(df);  // Independent nodes run concurrently

This replaces the wavefront-based ``OpenMPExecutor`` with a true dataflow
model: each node is submitted when its predecessors complete, with no
barrier between levels.

TaskPool inside graph nodes
---------------------------

A graph node's executor lambda can use TaskPool internally:

.. code-block:: cpp

    cg::Graph graph("fock_pipeline");
    {
        cg::CaptureGuard guard(graph);
        // Node 1: standard einsum (coarse-grained)
        cg::einsum("ik;kj->ij", &F, H, D);
    }

    // After graph execution, use TaskPool for fine-grained work
    pool.parallel_for("integral_batches", 0, num_pairs, [&](size_t pair) {
        compute_integrals(pair, &F_local);
    });

The graph handles operation ordering; the pool handles parallelism within
each operation.

Profiler Integration
====================

TaskPool integrates with the Einsums profiler:

- Worker threads appear as ``taskpool-worker-N`` in the profiler timeline.
- Each task creates a named profiler region, visible in both the tree and the flame graph.
- Metrics are accessible through the ``"get_taskpool_metrics"`` server handler.

The imgui profile viewer's Gantt panel shows task execution across workers.
Task regions appear nested under the submitting thread's profiler tree.

Accessing metrics:

.. code-block:: cpp

    auto m = pool.snapshot_metrics();
    println("Tasks: {} submitted, {} completed", m.total_submitted, m.total_completed);
    println("Steals: {}", m.total_steals);
    for (size_t i = 0; i < m.per_worker_executed.size(); i++) {
        println("  Worker {}: {} executed, {} stolen",
                i, m.per_worker_executed[i], m.per_worker_stolen[i]);
    }

OpenMP Coexistence
==================

The TaskPool and OpenMP coexist in the same program:

- Use TaskPool for fine-grained task parallelism (integral batches)
- Use OpenMP for BLAS-internal vectorization (handled by the vendor)
- Use ComputeGraph for operation-level sequencing

TaskPool workers do not call ``omp_set_num_threads()``, which avoids
globally affecting OpenMP parallelism. Tasks submitted to the pool may
call BLAS routines that use OpenMP internally; the vendor BLAS manages
its own thread count.

External submissions (from non-worker threads) go through a shared
mutex-protected queue rather than directly into worker deques. This
ensures thread safety since the work-stealing deques are single-producer
(owner thread only). Workers check the external queue on every iteration.

Future: MPI Distribution
========================

Phase 2 will add distributed execution across MPI ranks:

- ``TaskDescriptor`` with placement hints and data dependencies
- ``DistributedTaskPool`` that ships tasks to remote ranks
- ``parallel_reduce_distributed()`` with ``MPI_Allreduce``
- Execution policy: ``execution::distributed{comm}``

Phase 1 prepares for this by:

- ``WorkerContext.rank`` field (0 in Phase 1)
- Value-typed ``TaskHandle`` (serializable for remote)
- ``parallel_reduce`` combiner takes ``const&`` (MPI compatible)

API Reference
=============

TaskPool
--------

.. cpp:class:: einsums::task_pool::TaskPool

   Singleton work-stealing thread pool.

   .. cpp:function:: static TaskPool& get_singleton()
   .. cpp:function:: size_t num_workers() const
   .. cpp:function:: template<typename F> auto submit(std::string name, F&& callable) -> TaskHandle<R>
   .. cpp:function:: TaskGroup submit_group(std::string name, std::vector<std::function<void()>> tasks)
   .. cpp:function:: template<typename F> void parallel_for(std::string name, size_t begin, size_t end, F&& body)
   .. cpp:function:: template<typename Acc, typename Body, typename Combine> Acc parallel_reduce(std::string name, size_t begin, size_t end, Acc init, Body body, Combine combine)
   .. cpp:function:: template<typename F, typename... Ts> auto dataflow(std::string name, F&& callable, TaskHandle<Ts>... inputs)
   .. cpp:function:: Metrics snapshot_metrics() const
   .. cpp:function:: void shutdown()

TaskHandle
----------

.. cpp:class:: template<typename T> einsums::task_pool::TaskHandle

   Future with continuation support.

   .. cpp:function:: T get()
   .. cpp:function:: void wait()
   .. cpp:function:: bool ready() const
   .. cpp:function:: template<typename F> auto then(F&& fn) -> TaskHandle<R>
   .. cpp:function:: template<typename F> auto then(std::string name, F&& fn) -> TaskHandle<R>

TaskGroup
---------

.. cpp:class:: einsums::task_pool::TaskGroup

   Collective barrier for a group of submitted tasks.

   .. cpp:function:: void wait_all()
   .. cpp:function:: bool ready() const
   .. cpp:function:: size_t size() const
   .. cpp:function:: size_t remaining() const

Free Functions
--------------

.. cpp:function:: template<typename T> TaskHandle<std::vector<T>> when_all(std::vector<TaskHandle<T>>)

   Fan-in for homogeneous handles: completes when all are ready.

.. cpp:function:: template<typename... Ts> TaskHandle<std::tuple<Ts...>> when_all(TaskHandle<Ts>...)

   Fan-in for heterogeneous handles: completes when all are ready.
   Returns a tuple of results in the same order as the inputs.

.. cpp:function:: template<typename T> TaskHandle<T> make_ready_handle(T value)

   Create an immediately-ready handle.
