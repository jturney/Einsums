..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

=============
GEMM Batching
=============

The ``GEMMBatching`` optimization pass collapses groups of independent
GEMM-pattern einsum nodes into a single ``blas::gemm_batch`` call. The
primary win is **OpenMP-parallel execution across the batch
dimension**: many small independent matrix multiplies can now run in
parallel instead of one-by-one.

.. note::

   Einsums's current ``blas::gemm_batch`` is an OpenMP-parallel loop
   over per-matrix ``dgemm`` calls (``libs/Einsums/BLASVendor/src/gemm_batch.cpp``).
   It is **not** a vendor-native batched GEMM (MKL's
   ``cblas_dgemm_batch``, OpenBLAS's batched path). If Einsums ever
   gains a vendor-native override, this pass benefits automatically
   with no code change on the pass side.

   This means: the speedup comes from **batch-level parallelism**, not
   from vendor-level dispatch amortization. The win is largest when
   each individual GEMM is small enough that its own internal BLAS
   threads don't already saturate the machine — attention heads,
   Kronecker factors, per-sample evaluations, batched chemistry
   fragments. For large matrices where each GEMM already occupies all
   cores, batching can be neutral or slightly negative.

The pass runs automatically as part of ``PassManager::create_default()``
— if you already use the default pipeline, no code changes are needed.

.. contents::
   :local:

When batching kicks in
======================

The pass only considers einsums that match the "GEMM shape":

- Rank-2 × rank-2 → rank-2 contraction.
- Exactly one link (contracted) index.

At capture time, a ``GemmHint`` is attached to matching einsums'
descriptors; einsums that don't match (any rank-3+, multi-link, or
reduction patterns) get no hint and are ignored by the pass.

For a group of matching einsums to batch together, every member must
agree on:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Notes
   * - Dependency level
     - Two einsums at different topological levels have a data
       dependency between them; can't execute in parallel, can't batch.
   * - ``m``, ``n``, ``k``
     - All three GEMM dimensions must match exactly.
   * - ``trans_a``, ``trans_b``
     - Derived from the index pattern. ``ik;kj->ij`` is ``NN``;
       ``ki;kj->ij`` is ``TN``; etc.
   * - Element type
     - All members must share one of: ``float``, ``double``,
       ``std::complex<float>``, ``std::complex<double>``.
   * - ``alpha``, ``beta`` (bit-equal)
     - Compared at the bit level, not numerically. 1.0 and 0.9999...
       never batch together even though they look "close".
   * - ``lda``, ``ldb``, ``ldc``
     - ``blas::gemm_batch`` takes one leading-dim triple for the whole
       batch. Members with mismatched strides (e.g. a view vs an
       owning tensor) are rejected as a group.

Any mismatch in the key splits the group; a workload can produce
multiple batches if, say, half the einsums are ``float`` and half are
``double`` — the pass emits one ``BatchedGemm`` per compatible group.

What the graph looks like
=========================

Before the pass:

.. code-block:: cpp

   cg::Graph graph("many_gemms");
   {
       cg::CaptureGuard guard(graph);
       for (int i = 0; i < 32; ++i)
           cg::einsum("ik;kj->ij", &Cs[i], As[i], Bs[i]);
   }
   // graph.num_nodes() == 32

After ``GEMMBatching`` (which ``create_default()`` runs for you):

.. code-block:: text

   graph.num_nodes() == 1

The single node has ``kind == OpKind::BatchedGemm``, 2*32 = 64 inputs
(A_0, B_0, A_1, B_1, …) and 32 outputs, and stores a
``BatchedGemmDescriptor`` recording the shared ``m/n/k/lda/ldb/ldc``
plus the ``batch_count`` and scalar-type tag.

The executor packs the data pointers from each tensor slot on every
call (so ``graph.rebind()`` continues to work) and then dispatches
``blas::gemm_batch<T>`` once per ``graph.execute()``.

End-to-end example
==================

A runnable example lives at
``libs/Einsums/ComputeGraph/examples/GEMMBatchingDemo.cpp``. Its core is:

.. code-block:: cpp

   namespace cg = einsums::compute_graph;

   constexpr int BATCH = 32;
   constexpr int M = 16, K = 16, N = 16;

   std::vector<Tensor<double, 2>> As, Bs, Cs;
   for (int i = 0; i < BATCH; ++i) {
       As.push_back(create_random_tensor<double>(fmt::format("A{}", i), M, K));
       Bs.push_back(create_random_tensor<double>(fmt::format("B{}", i), K, N));
       Cs.push_back(create_zero_tensor<double>(fmt::format("C{}", i), M, N));
   }

   cg::Graph graph("batched_workload");
   {
       cg::CaptureGuard guard(graph);
       for (int i = 0; i < BATCH; ++i)
           cg::einsum("ik;kj->ij", &Cs[i], As[i], Bs[i]);
   }

   auto pm = cg::PassManager::create_default();
   graph.apply(pm);         // GEMMBatching collapses the 32 nodes into 1
   graph.execute();

On a typical development box (32 × 16×16×16 double GEMMs) the demo
reports roughly **2-3× speedup** from the batched dispatch. Bigger wins
show up as the matrices shrink (overhead dominates more) and as the
batch size grows.

When batching does NOT help (or actively hurts)
================================================

- **Large matrices** (say ≥ 256×256×256). BLAS dispatch overhead is a
  small fraction of the compute cost; batching won't help much.
- **Very small batches** (2-3 members). The setup cost of packing
  pointer arrays can outweigh the saved dispatch.
- **GPU paths.** The pass currently lives before GPU placement; GPU
  batching (``cublasDgemmBatched`` / ``cublasDgemmStridedBatched``) is
  a future addition. For now, GPU-placed einsums are not batched.
- **Distributed tensors.** ``DistributionPlanning`` runs after
  ``GEMMBatching``. Einsums that would need distributed dispatch are
  not given a ``GemmHint`` by the capture path when that case is
  relevant — and even when they are, ``DistributionPlanning`` inspects
  ``EinsumDescriptor`` which ``BatchedGemm`` replaces. Treat the
  interaction with distributed dispatch as "use one or the other,
  don't expect both" for now.

Opting out
==========

If you need to disable batching (e.g. for performance A/B testing), you
can build the pipeline manually and omit it:

.. code-block:: cpp

   cg::PassManager pm;
   pm.add<cg::passes::ConstantFolding>();
   pm.add<cg::passes::ScaleAbsorption>();
   pm.add<cg::passes::PermuteFusion>();
   pm.add<cg::passes::CSE>();
   pm.add<cg::passes::DeadNodeElimination>();
   // ... all other default passes except GEMMBatching
   graph.apply(pm);

Or apply ``GEMMBatching`` explicitly and inspect what it did:

.. code-block:: cpp

   auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
   std::cout << "created " << pass.num_batches() << " batches, "
             << "absorbed " << pass.total_batched() << " einsums\n";

Implementation notes
====================

- Leading dimensions are read at execute time via the extractors stored
  in the ``GemmHint``. Each extractor captures the original tensor's
  C++ type (``AType``, ``BType``, ``CType``) so it can call
  ``tensor.impl().get_lda()``. This keeps the pass tensor-type-agnostic
  while still getting correct leading dims under any Einsums tensor
  layout.
- Complex alpha/beta are promoted from the real-valued user input with
  zero imaginary part (matches the rest of the capture path).
- The pass only batches entries it can prove are safe. When any safety
  check (uniform strides, matching keys, no data dependencies) fails,
  the group is left unchanged and execution falls back to the usual
  per-einsum path. Correctness is preserved in every case.

TODO / future work
==================

- **MPS batched-matmul backend**: currently the MPS branch of
  ``gpu::blas::gemm_strided_batched`` falls through to the CPU
  pointer-array ``gemm_batch`` (the OpenMP-parallel loop). Apple's
  Metal Performance Shaders offers real GPU-accelerated batched
  matmul via ``MPSMatrixMultiplication`` (``batchStart`` /
  ``batchSize``) or ``MPSNDArrayMatrixMultiplication`` (macOS 13+).
  Wiring either into the MPS branch would give Apple Silicon users
  the same benefit CUDA/HIP users get from the vendor strided-batched
  primitive. Caveats: MPS's double-precision is weak (may still
  round-trip to CPU inside Metal), and ``MPSMatrix`` expects
  row-major storage, so the col-major-batch-last layout path needs
  attention.
- **Pointer-array GPU batching**: the GPU path handles strided
  batches today (what 3D capture emits). The pass-driven pointer-
  array batches (from ``GEMMBatching`` over N independent 2D
  einsums) stay on CPU. Extending to GPU needs either
  ``cublasDgemmBatched`` (the pointer-array variant) or per-slice
  D2H staging onto a contiguous device buffer first.
- **Partial-contraction fast paths**: multi-link cases like
  ``"bijk;bjkl->bil"`` can sometimes be flattened to GEMM when the
  multiple link indices are memory-adjacent. Not detected today.

See also
========

- :doc:`optimization_passes` — catalog of every pass in the default pipeline.
- :doc:`operations` — ``cg::einsum`` string syntax reference.
- ``libs/Einsums/ComputeGraph/tests/unit/GEMMBatching.cpp`` —
  correctness suite exercising all skip paths plus multi-type and
  multi-batch cases.
- ``libs/Einsums/ComputeGraph/tests/unit/StridedBatchedGemm.cpp`` —
  3D, 4D, and 5D batched-contraction capture + correctness coverage,
  plus the Target::GPU routing test.
