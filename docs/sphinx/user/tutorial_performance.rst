..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _tutorial-performance:

*****************************
Tutorial: Performance Tuning
*****************************

This tutorial explains how Einsums dispatches tensor contractions, how to use
the profiler to find bottlenecks, and practical tips for writing fast code.

How Einsum Dispatch Works
=========================

When you call ``einsum()``, Einsums analyzes the index pattern at compile time
and selects the fastest algorithm:

1. BLAS specialization makes direct hardware-optimized calls:

   - ``DOT`` for the rank-0 output :math:`c = \sum_i A_i B_i`
   - ``GER`` for the outer product :math:`C_{ij} = \alpha x_i y_j`
   - ``GEMV`` for the matrix-vector product :math:`y_i = \sum_j A_{ij} x_j`
   - ``GEMM`` for the matrix-matrix product :math:`C_{ij} = \sum_k A_{ik} B_{kj}`

2. PackedGemm applies BLIS-style cache-blocked packing. It handles higher-rank
   contractions that can be flattened to GEMM:

   - Multi-K: :math:`C_{il} = \sum_{jk} A_{ijk} B_{jkl}`
   - Multi-M: :math:`C_{ijl} = \sum_k A_{ijk} B_{kl}`
   - Multi-N: :math:`C_{ijl} = \sum_k A_{ik} B_{kjl}`
   - Batch: :math:`C_{bij} = \sum_k A_{bik} B_{bkj}`
   - Combinations of the above

3. The generic algorithm uses nested loops. It is the fallback for patterns that
   don't map to GEMM, such as Hadamard products with repeated indices like
   :math:`C_i = A_{ii} B_{ii}`.

The dispatch is automatic. You would write the same ``einsum()`` call regardless.
The profiler tells you which path was taken.

Using the Profiler
==================

Einsums has a built-in profiler that tracks time spent in every operation.

Timing Report
-------------

.. code-block:: cpp

    #include <Einsums/Profile/Profile.hpp>

    int main(int argc, char *argv[]) {
        einsums::initialize(argc, argv);

        // ... your code ...

        einsums::finalize();
        // Profiler report is printed automatically on shutdown
    }

The report shows a tree of timed regions with inclusive/exclusive times,
call counts, and standard deviations.

Profile Viewer
--------------

Connect the imgui profile viewer for real-time visualization:

.. code-block:: bash

    # Build the viewer (in devtools/profiling/imgui_viewer/)
    cmake -B build && cmake --build build

    # Run your Einsums program (it listens on localhost:19216)
    ./my_program

    # Connect the viewer
    ./build/profile_viewer_imgui

The viewer provides:

- Tree view: hierarchical timing breakdown.
- Flame graph: visual call-stack proportions.
- Hotspots: flat list sorted by exclusive time.
- Gantt chart: timeline of events.
- Compute Graph: interactive DAG of captured graph nodes, under View > Compute Graph.

Custom Profiling Regions
------------------------

Add your own timed regions:

.. code-block:: cpp

    #include <Einsums/Profile/Profile.hpp>
    using namespace einsums;

    void my_function() {
        profile::Profiler::instance().push("my_function");

        // ... work ...

        profile::Profiler::instance().pop();
    }

Performance Tips
================

1. Prefer einsum over manual loops
-----------------------------------

Einsums automatically dispatches to BLAS when possible. Manual loops will
almost always be slower:

.. code-block:: cpp

    // SLOW: manual loop
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C(i, j) += A(i, k) * B(k, j);

    // FAST: dispatches to BLAS GEMM
    einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);

2. Use views instead of copying
--------------------------------

Views avoid memory allocation and data copying:

.. code-block:: cpp

    // SLOW: copies a submatrix
    auto block = Tensor<double, 2>("block", 3, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            block(i, j) = A(i + 2, j + 5);

    // FAST: zero-copy view
    auto block = A(Range{2, 5}, Range{5, 8});

3. Mind index ordering for cache efficiency
--------------------------------------------

Einsums tensors are row-major by default (last index varies fastest).
When the contraction pattern aligns with memory layout, PackedGemm can avoid
expensive packing:

.. code-block:: cpp

    // C_il = A_ijk * B_jkl
    // k is the innermost index of A → good cache locality for A
    // l is the innermost index of B → good cache locality for B

4. Use ComputeGraph for iterative code
---------------------------------------

Capture once, replay many times. The graph avoids re-analyzing the contraction
pattern on each call:

.. code-block:: cpp

    cg::Graph graph("iteration");
    { /* capture */ }

    // Apply optimization passes once
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);

    // Replay 1000 times -- no overhead from dispatch analysis
    for (int iter = 0; iter < 1000; iter++) {
        graph.execute();
    }

5. Check the profiler annotation for dispatch path
---------------------------------------------------

In the profile viewer, each ``einsum`` node shows an ``algorithm`` annotation, 
visible in the tree view or flame graph tooltip.  At runtime, any contraction
that falls back to the ``GENERIC`` nested-loop algorithm emits a one-time
warning to the log, for example:

.. code-block:: text

    [warning] einsum dispatch: GENERIC fallback for "C"("i") = "A"("i", "j") * "B"("j", "i")
              (ranks 1/2/2).  This contraction is not accelerated by BLAS.

If ``try_packed_gemm`` rejects a contraction, an INFO-level log explains why:

.. code-block:: text

    [info] PackedGemm: skipping -- no N-dims (all C indices come from A).
           Consider rewriting as GEMV or transposing.

Set ``--einsums:log-level 2`` (INFO) to see the PackedGemm reasons, or
``--einsums:log-level 3`` (WARN) to see only the final GENERIC fallback.

Understanding PackedGemm
=========================

PackedGemm is a BLIS-inspired backend that handles tensor contractions beyond
simple GEMM. It works by:

1. Classifying the contraction into M (output dims from A), N (output dims
   from B), K (link dims), and batch dims
2. Packing A and B into cache-friendly contiguous buffers
3. Calling BLAS GEMM on packed tiles

This is much faster than the generic nested-loop algorithm because:

- Data is laid out for optimal cache reuse through L1/L2/L3 blocking.
- BLAS GEMM is heavily optimized by the vendor with SIMD and loop unrolling.
- Packing cost is amortized over the GEMM computation.

PackedGemm handles:

- Single M/N/K: standard GEMM, often dispatched to direct BLAS instead.
- Multi-K: :math:`C_{il} = \sum_{jk} A_{ijk} B_{jkl}`, flattening the K dims.
- Multi-M: :math:`C_{ijl} = \sum_k A_{ijk} B_{kl}`, flattening the M dims.
- Multi-N: :math:`C_{ijl} = \sum_k A_{ik} B_{kjl}`, flattening the N dims.
- Batch: :math:`C_{bij} = \sum_k A_{bik} B_{bkj}`, a parallel batch GEMM.
- Any combination of the above.

For batch contractions, PackedGemm uses ``gemm_batch`` which runs all
batch slices in parallel via OpenMP. This is much faster than the serial
loop used previously.

6. Use TaskPool for data-parallel workloads
--------------------------------------------

For workloads with thousands of independent tasks (e.g., integral generation),
use the TaskPool instead of OpenMP:

.. code-block:: cpp

    #include <Einsums/TaskPool/TaskPool.hpp>

    auto &pool = einsums::task_pool::TaskPool::get_singleton();

    // parallel_for: chunked, work-stealing, load-balanced
    pool.parallel_for("integral_batches", 0, num_shell_pairs, [&](size_t pair) {
        compute_integrals(pair, &F_local);
    });

    // parallel_reduce: thread-local accumulators, no false sharing
    double energy = pool.parallel_reduce<double>("energy", 0, N,
        []() { return 0.0; },
        [&](size_t i, double &acc) { acc += compute_contribution(i); },
        [](double &g, double const &l) { g += l; }
    );

TaskPool advantages over raw OpenMP:

- Work-stealing handles load imbalance automatically
- Continuations (``.then()``) and dataflow avoid manual synchronization
- Profiler integration shows per-task and per-worker timing
- Future-proof for MPI distribution

Dispatch Reference: What Is and Isn't Accelerated
===================================================

The table below lists every dispatch tier, what it handles, and what falls
through to the next tier.  The chain is evaluated top-to-bottom; the first
match wins.

.. list-table:: Einsum Dispatch Chain
   :header-rows: 1
   :widths: 10 30 30 30

   * - Tier
     - Algorithm
     - Handles
     - Skips to next when
   * - 1
     - ``DOT`` (``sdot``/``ddot``)
     - :math:`c = \sum_i A_i B_i`: scalar result, A and B have identical
       index packs
     - Output is not scalar, or indices differ
   * - 2
     - ``DIRECT`` (element-wise)
     - :math:`C_{ij} = A_{ij} B_{ij}`: all three index packs identical,
       no conjugation
     - Index packs don't all match
   * - 3
     - ``GER`` (outer product)
     - :math:`C_{ij} = \alpha x_i y_j`: no link indices, indices in A and B
       are contiguous subsets of C's indices
     - Link indices exist, or A/B indices are not contiguous in C
   * - 4
     - ``GEMV`` (matrix-vector)
     - :math:`y_i = \sum_j A_{ij} x_j`: one rank-2 operand, one rank-1
       operand, one link index, result is rank-1
     - Ranks don't match the GEMV pattern
   * - 5
     - ``GEMM`` (direct BLAS)
     - :math:`C_{ij} = \sum_k A_{ik} B_{kj}`: all rank-2, single M/N/K,
       contiguous stride-1 layout
     - Ranks > 2, or non-contiguous layout, or multi-M/N/K
   * - 6
     - ``SORT_GEMM`` (permute + BLAS)
     - Same structural requirements as GEMM, namely a clean M/N/K decomposition,
       but the indices are scrambled.  Transposes tensors via HPTT, then dispatches to
       GEMM.
     - No clean M/N/K decomposition exists, or Hadamard indices present
   * - 7
     - ``PACKED_GEMM`` (BLIS-style)
     - Higher-rank contractions with a valid M/N/K decomposition: multi-M,
       multi-N, multi-K, batch dims, and any combination.  Handles arbitrary
       rank as long as every C index classifies as M-only from A, N-only
       from B, or batch from both.
     - m_count=0 or n_count=0, meaning no M or N dims, no link indices, or
       single-M/N with non-stride-1 C layout
   * - 8
     - ``GENERIC`` (nested loops)
     - Everything else.  Correct but slow: O(product of all dim sizes) with
       no BLAS acceleration.
     - Final fallback.

Contractions NOT Accelerated (GENERIC Fallback)
-----------------------------------------------

The following contraction patterns currently fall through to the generic
nested-loop algorithm.  They are correct but not BLAS-accelerated:

Hadamard contractions with reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An index appears in all three tensors AND there is a separate link index:

.. code-block:: text

    C(i) = A(i,j) * B(j,i)    # i is Hadamard, j is link
                                # No clean M/N split: i is in both A and B

This is NOT the same as the batch case ``C(b,i,j) = A(b,i,k) * B(b,k,j)``
which IS accelerated.  The difference: in the batch case, the remaining
indices after removing batch dims still have a valid M/N/K decomposition.

Transposed trace (dot with permuted indices)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    scalar = A(i,j) * B(j,i)    # Like dot(A, B^T) but B's indices are reversed

The ``DOT`` dispatch requires ``A_indices == B_indices`` (same order).  This could
be accelerated by transposing B first, but is not currently implemented.

No M-dims or no N-dims
^^^^^^^^^^^^^^^^^^^^^^

All output (C) indices come from only one operand:

.. code-block:: text

    C(i,j) = A(i,j,k) * B(k)    # All C indices from A, none from B → no N-dims

This is structurally a GEMV, or matrix-vector multiplication, but with a higher-rank
tensor.  PackedGemm requires at least one M-dim and one N-dim.

Duplicate indices within one tensor (true Hadamard / trace)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    C(i) = A(i,k,k) * B(k)    # k appears twice in A (trace)

These are sent directly to the generic algorithm because the BLAS dispatch
chain cannot handle duplicate indices within a single operand.

Mixed data types
^^^^^^^^^^^^^^^^

.. code-block:: text

    Tensor<float> A;
    Tensor<double> B;
    Tensor<double> C;
    einsum(Indices{i,j}, &C, Indices{i,k}, A, Indices{k,j}, B);  # float × double

Different value types across A, B, and C skip the BLAS specializations.

Non-BasicTensor types with no special dispatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``BlockTensor``, ``TiledTensor``, and other composite tensor types that
don't provide a specialized ``einsum_special_dispatch`` override will
recurse into their blocks/tiles and eventually call ``einsum`` on the
underlying ``Tensor``/``TensorView`` pieces, which are accelerated.

How to Tell What Path Your Code Takes
--------------------------------------

Three ways to identify the dispatch path:

1. Log warnings are the easiest. Run with the default log level, and any GENERIC
   fallback emits a one-time warning per unique contraction pattern.

2. Profiler annotations: in the profiler tree/flame view, each ``einsum``
   zone has an ``algorithm`` annotation showing ``DOT``, ``GEMM``,
   ``PACKED_GEMM``, ``SORT_GEMM``, or ``GENERIC``.

3. The algorithm-choice output parameter lets you check programmatically:

   .. code-block:: cpp

       #include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
       using einsums::tensor_algebra::detail::AlgorithmChoice;

       AlgorithmChoice algo;
       einsum(Indices{i,j}, &C, Indices{i,k}, A, Indices{k,j}, B, &algo);
       // algo == GEMM, PACKED_GEMM, GENERIC, etc.

What's Not Covered
==================

This tutorial covers CPU performance. For GPU acceleration, see the HIP/CUDA
documentation (when available).
