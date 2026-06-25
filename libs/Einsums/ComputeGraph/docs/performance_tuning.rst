.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

===================
Performance Tuning
===================

This guide covers how to tune Einsums ComputeGraph for optimal performance.
The defaults work well for most cases; tuning matters at scale (large tensors,
many ranks, GPU offload).

Executor Selection
===================

Choose the executor based on your workload:

.. code-block:: cpp

   graph.execute();                    // Sequential (default) — simplest, no overhead
   cg::DataflowExecutor df;
   graph.execute(df);                  // Maximum overlap via TaskPool
   cg::OpenMPExecutor omp;
   graph.execute(omp);                 // OpenMP task parallelism
   cg::MPIExecutor mpi;
   graph.execute(mpi);                 // MPI-aware (same as Sequential on 1 rank)

When to use which
-----------------

- **Sequential**: Small graphs (<10 nodes), debugging, or when nodes are already
  internally parallel (OpenMP BLAS). Zero scheduling overhead.
- **DataflowExecutor**: Large graphs with independent branches, async I/O overlap,
  or communication overlap. Best for pipelines and multi-stage computations.
- **OpenMPExecutor**: Medium graphs where BLAS is single-threaded and you want
  node-level parallelism. Avoid if BLAS already uses all cores.
- **MPIExecutor**: Distributed computation. On single rank, identical to Sequential.

Pass Configuration
===================

Disabling Passes
-----------------

Disable specific passes via runtime config:

.. code-block:: bash

   ./my_program --einsums:pass-disable "ContractionPlanning,GEMMBatching"

Or in code:

.. code-block:: cpp

   // Build custom pipeline without specific passes
   cg::PassManager pm;
   pm.add<cg::passes::ConstantFolding>();
   pm.add<cg::passes::Reorder>();
   pm.add<cg::passes::Materialization>();
   // Omit ContractionPlanning, GEMMBatching, etc.
   graph.apply(pm);

Verbose Pass Logging
---------------------

See what each pass does:

.. code-block:: bash

   ./my_program --einsums:pass-verbose

This logs which passes modify the graph, how many nodes change, and timing.

Analysis Mode
--------------

Run passes without applying changes (dry run):

.. code-block:: bash

   ./my_program --einsums:pass-analyze

Distribution Tuning
====================

Distribution Threshold
-----------------------

The ``DistributionPlanning`` pass only distributes tensors above a size threshold.
The default is 64 MB. Tensors smaller than this are replicated on every rank.

.. code-block:: cpp

   // Lower threshold to distribute smaller tensors
   cg::PassManager pm;
   pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1024 * 1024);  // 1 MB

   // Disable distribution entirely
   pm.add<cg::passes::DistributionPlanning>(/*threshold=*/SIZE_MAX);

Guidelines
^^^^^^^^^^

- For small calculations (< 1000 basis functions): keep default (64 MB). Most
  tensors are small enough that replication is cheaper than communication.
- For large calculations: lower to 1-10 MB. More tensors get distributed, reducing
  per-rank memory.
- For debugging distribution: set threshold to 1 byte to force everything distributed.

SUMMA vs Outer-Product
-----------------------

On square grids (Pr == Pc), ``DistributionPlanning`` uses SUMMA (distributes
link indices too). On non-square grids, it falls back to outer-product.

.. code-block:: cpp

   // Force outer-product only (no SUMMA, even on square grids)
   pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);

When to disable SUMMA
^^^^^^^^^^^^^^^^^^^^^

- When K, the link dimension, is small relative to M and N, so the broadcast
  overhead exceeds the memory savings.
- When tensors are used in chains where the link dimension changes role, so SUMMA
  distribution may conflict. Conflict resolution handles most such cases.

Process Grid Shape
-------------------

The ``ProcessGrid`` auto-computes a near-square grid. For specific shapes:

.. code-block:: cpp

   // Force a specific grid shape
   comm::ProcessGrid grid(4, 2, comm::Communicator::world());  // 4 rows × 2 cols

For most workloads, the auto-computed near-square grid is optimal. Override when:

- Your problem has very asymmetric dimensions (e.g., M >> N)
- You want to control which dimension gets more parallelism

Memory Tuning
==============

FreeInsertion Threshold
------------------------

The ``FreeInsertion`` pass frees intermediate tensors above a size threshold.
The default is 1 MB. Smaller intermediates stay alive to avoid alloc/free
overhead in loops.

.. code-block:: cpp

   // Free intermediates > 10 MB
   pm.add<cg::passes::FreeInsertion>(/*min_bytes=*/10 * 1024 * 1024);

   // Free everything (maximum memory savings, more alloc overhead in loops)
   pm.add<cg::passes::FreeInsertion>(/*min_bytes=*/0);

   // Disable freeing (keep everything alive — fastest for tight loops)
   pm.add<cg::passes::FreeInsertion>(/*min_bytes=*/SIZE_MAX);

Memory Budget
--------------

The ``DataflowExecutor`` can enforce a memory budget:

.. code-block:: cpp

   cg::DataflowExecutor df;
   df.set_memory_budget(4ULL * 1024 * 1024 * 1024);  // 4 GB limit
   graph.execute(df);

When the budget would be exceeded, the executor delays Materialize nodes until
Free nodes release enough memory. This prevents OOM for graphs with many large
intermediates.

Guidelines
^^^^^^^^^^

- Set to ~80% of available RAM to leave room for BLAS workspace and OS overhead.
- If your graph never exceeds RAM, leave at 0 (unlimited) for best parallelism.
- The budget only gates Materialize nodes; it does not affect compute scheduling.

I/O Tuning
===========

Async I/O Overlap
------------------

Use ``DataflowExecutor`` + ``IOPrefetch`` pass for I/O-compute overlap:

.. code-block:: cpp

   auto pm = cg::PassManager::create_default();  // Includes IOPrefetch
   graph.apply(pm);

   cg::DataflowExecutor df;
   graph.execute(df);  // Reads overlap with compute

``IOPrefetch`` moves DiskRead nodes as early as possible in the schedule.
Combined with the DataflowExecutor's async support, reads happen concurrently
with independent compute.

.etn File Performance
----------------------

For maximum throughput:

- Use large tensors, above 1 MB per write call; small writes are overhead-dominated.
- Use ``ReadWrite`` mode for append, which avoids rewriting existing data.
- For distributed I/O, ``DistributedTensorFile`` uses POSIX pwrite with MPI offset
  coordination. It has no MPI-IO dependency and works reliably on all filesystems.
- For slice reads, the innermost dimension (dim 0 in column-major) is contiguous
  and fastest. Non-contiguous slices require multiple read calls.

Profiling
==========

Enable detailed profiling to find bottlenecks:

.. code-block:: bash

   # Text report
   ./my_program --einsums:profiler-report --einsums:profiler-detailed

   # JSON export (for visualization)
   ./my_program --einsums:profiler-save profile.json

   # TCP server (for live monitoring)
   ./my_program --einsums:profiler-port 19216

What to look for
----------------

- **SUMMA panels**: ``broadcast_A``, ``broadcast_B``, ``local_gemm``. If broadcasts
  dominate, tensors may be too small for SUMMA, so switch to outer-product.
- **MPI collectives**: ``allreduce``, ``broadcast``, ``barrier``. If communication
  dominates, reduce the number of distributed tensors or increase computation per
  communication.
- **FreeInsertion**: ``free(tensor_name)`` events. If alloc/free overhead is visible,
  increase the FreeInsertion threshold.
- **I/O**: ``read_etn``, ``write_etn``. If I/O is not overlapped with compute,
  ensure you are using DataflowExecutor with IOPrefetch.

Common Patterns
================

SCF Loop
---------

.. code-block:: cpp

   cg::PassManager pm;
   pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1024 * 1024);
   pm.add<cg::passes::Materialization>();
   pm.add<cg::passes::InputSlicing>();
   pm.add<cg::passes::FreeInsertion>(/*min_bytes=*/10 * 1024 * 1024);  // Free large intermediates
   pm.add<cg::passes::CommunicationInsertion>();
   graph.apply(pm);

   for (int iter = 0; iter < max_iter; iter++) {
       graph.execute();
       // FreeInsertion releases large intermediates after each iteration.
       // Materialize re-allocates them at the start of the next iteration.
       // Small intermediates (< 10 MB) stay alive across iterations.

       if (converged) break;
       // Checkpoint every 5 iterations
       if (iter % 5 == 0) {
           tensor_io::checkpoint::save("scf_checkpoint.etn", graph);
       }
   }

Large Integral Transformation
------------------------------

.. code-block:: cpp

   // Tight memory budget for large ERI tensors
   cg::DataflowExecutor df;
   df.set_memory_budget(8ULL * 1024 * 1024 * 1024);  // 8 GB

   auto pm = cg::PassManager::create_default();
   graph.apply(pm);
   graph.execute(df);  // Budget prevents OOM, Free nodes release intermediates ASAP
