.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

===================
Optimization Passes
===================

The ComputeGraph provides a catalog of built-in optimization passes. The default
pipeline (``create_default()``) adds 17 passes that always run, plus up to five
GPU passes and five distributed passes that are included only when a GPU or MPI
backend (or its mock) is available. Some passes modify the graph, the
graph-transforming kind. Others only analyze and report, the analysis-only kind.

Applying Passes
===============

.. code-block:: cpp

   // Single pass (returns pair of modified flag + pass instance):
   auto [modified, cse] = graph.apply<cg::passes::CSE>();

   // Multiple passes via PassManager:
   cg::PassManager pm;
   pm.add<cg::passes::ScaleAbsorption>()
     .add<cg::passes::CSE>()
     .add<cg::passes::Reorder>();
   graph.apply(pm);

   // Default pipeline (all built-in passes in recommended order):
   auto pm = cg::PassManager::create_default();
   graph.apply(pm);

   // Apply to all stages of a pipeline:
   pipeline.apply(pm);

Writing Custom Passes
=====================

.. code-block:: cpp

   class MyPass : public cg::OptimizerPass {
   public:
       std::string name() const override { return "MyPass"; }
       bool run(cg::Graph &graph) override {
           auto &nodes = graph.nodes();
           // Inspect and modify nodes...
           // If you modify the order, call graph.mark_sorted()
           return true;  // Return true if modified
       }
   };

Graph-Transforming Passes
=========================

ScaleAbsorption
----------------

Absorbs ``Scale(α, C)`` into any subsequent operation that writes to ``C``
with a zero beta/c_prefactor:

- **Einsum**: ``c_prefactor=0`` becomes ``c_prefactor=α``
- **Gemm**:  ``beta=0`` becomes ``beta=α``
- **Permute**: ``beta=0`` becomes ``beta=α``

The scale node is removed from the graph and its effect folded into the
following operation's prefactor. Reports ``num_absorbed()`` count.

CSE: Common Subexpression Elimination
--------------------------------------

**Pattern**: Two nodes with identical ``OpKind``, ``inputs``, and ``OpData``.

**Result**: Second node removed; its outputs redirected to first node's outputs.

ConstantFolding
----------------

Identifies nodes whose inputs are all constant: graph-owned intermediate
tensors (``is_intermediate=true``) that are never written by any node. User-owned
tensors are not assumed constant because they may change between loop iterations
or successive ``execute()`` calls. This makes the pass safe for both one-shot
graphs and loop bodies.

Propagation follows the dependency chain: if node A is folded, nodes depending
only on A's outputs are also foldable.

.. note::

   This pass has side effects. It executes folded nodes during the pass itself.
   It is included in ``create_default()`` and is safe for Pipeline loop bodies.

Reports ``num_folded()`` count.

DeadNodeElimination
--------------------

Removes nodes whose outputs are all graph-owned intermediates (``is_intermediate=true``)
with no consumers. This is useful after CSE or other passes eliminate consumers,
because the original producer may then become dead.

Control flow, memory, and side-effect nodes are never eliminated.

Reports ``num_eliminated()`` count.

Reorder
--------

**Algorithm**: Memory-aware Kahn's algorithm with priority queue.

Among ready nodes, schedules the one that frees the most memory first.
Reduces peak memory by releasing large intermediates earlier.

LoopInvariantHoisting
----------------------

For each Loop node, identifies operations inside the body whose inputs are
never modified by any other operation in the body. These loop-invariant
operations are moved before the loop to execute only once.

Self-modifying operations (scale, axpy, element_transform) are never hoisted
since they read and write the same tensor.

Reports ``num_hoisted()`` count.

SymmetryPropagation
--------------------

Walks the graph and tags intermediate tensors whose symmetry can be
proven from their inputs, pushing the inferred ``SymmetryDescriptor`` to
the backing tensor so the rank-2 BLAS dispatch fires at
``graph.execute()``. Current rules: scale preserves symmetry; axpy/axpby
with same-descriptor operands preserves it; a rank-2 self-contraction
(``AᵀA`` or ``AAᵀ``) produces a symmetric result; a permute of a
symmetric tensor stays symmetric.

Only mutates graph-owned intermediates. Reports ``num_inferred()`` count.

See :doc:`symmetry` for the full symmetry-aware-tensor story.

Analysis-Only Passes
====================

These passes analyze the graph and report findings but do not modify it.

MemoryPlanning
---------------

Computes tensor liveness intervals and reports memory statistics:

.. code-block:: cpp

   auto [modified, mem] = graph.apply<cg::passes::MemoryPlanning>();
   mem.print_report(std::cout);

Reports ``total_memory()`` and ``peak_memory()``.

ChainParenthesization
----------------------

Detects chains of GEMM-pattern einsum nodes and computes the optimal
multiplication order using the classical O(n³) matrix chain DP algorithm.

Reports ``original_flops()`` and ``optimal_flops()``.

InplaceOptimization
--------------------

Detects intermediate tensors with exactly one producer and one consumer.
These are candidates for in-place operation, where the consumer overwrites
the intermediate's buffer instead of allocating a separate output.

Reports ``num_candidates()`` count.

GEMMBatching
-------------

Collapses groups of independent GEMM-pattern einsum nodes at the same
dependency level into a single ``OpKind::BatchedGemm`` node whose executor
calls ``blas::gemm_batch<T>``. One BLAS dispatch covers the whole batch
instead of N. This is a substantial win for workloads that issue many
small contractions: stacked attention heads, per-sample transforms,
Kronecker-factored updates, and batched chemistry kernels. See
:doc:`gemm_batching` for a full walk-through and timing numbers.

**Candidate:** a 2D×2D→2D einsum with exactly one link index. Capture
populates an internal ``GemmHint`` on the descriptor only for this shape;
other einsums are skipped.

**Group key:** ``(level, m, n, k, trans_a, trans_b, scalar_type, alpha_bits, beta_bits)``.
Everything in the key must match exactly. Alpha and beta are compared
bit-equal, so 1.0 and 0.9999... never accidentally batch together.

**Uniform-stride check:** ``blas::gemm_batch`` takes a single
``lda``/``ldb``/``ldc`` for the whole batch. The pass probes the first
member's leading dimensions and rejects the group if any other member
disagrees.

**Element types:** float, double, std::complex<float>, std::complex<double>.
Complex alpha and beta are assumed real, which matches the capture path.

Reports ``num_batches()`` and ``total_batched()`` counts.

PermuteFusion
--------------

Absorbs an axis-reordering Permute node into the subscript of the Einsum
that reads it, eliminating one tensor-shaped data copy per match. The
einsum's indices are mutated in place through the
``std::shared_ptr<EinsumIndices>`` captured by its executor, and the
permute-output slot is redirected to the pre-permute tensor.

Safety conditions (all must hold)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The permute's output has exactly one consumer.
- The permute is a pure axis reorder: ``alpha == 1``, ``beta == 0``,
  and ``c_indices`` is a duplicate-free permutation of ``a_indices``.
- The consumer einsum has a populated ``EinsumDescriptor::indices``
  shared state. String-captured einsums are the only form after the
  tuple-overload removal, so this always holds for them.

Reports ``num_candidates()``, the number of detected pairs, and
``num_rewrites()``, the number of pairs that passed the safety filter and
were actually fused.

IOPrefetch
-----------

Moves ``DiskRead`` nodes as early as legally possible in the schedule.

Most ``DiskRead`` nodes have no predecessors, since they load data from files
that exist independently of the graph. By moving them to the beginning, the pass
maximizes the window between the read's ``async_start`` and its first consumer,
enabling maximum I/O-compute overlap when used with the ``DataflowExecutor``.

``DiskWrite`` nodes are not moved. Writes should execute as late as possible
to avoid blocking compute on I/O completion.

.. code-block:: cpp

   auto [modified, prefetch] = graph.apply<cg::passes::IOPrefetch>();
   // prefetch.num_prefetched() == number of DiskRead nodes moved earlier

Reports ``num_prefetched()`` count. Included in ``create_default()`` after Reorder.

ElementWiseFusion
------------------

Fuses consecutive element-wise operations on the same tensor. Currently handles
consecutive ``Scale`` operations:

**Pattern**: ``Scale(2.0, A)`` followed by ``Scale(3.0, A)``

**Result**: Merged into ``Scale(6.0, A)``, executing both lambdas sequentially.

Reports ``num_fused()`` count.

DistributiveFactoring
----------------------

Detects groups of einsums accumulating into the same output tensor with a
shared operand and rewrites them using the distributive property:

**Pattern**: ``R += A*B1; R += A*B2``

**Result**: ``T = B1 + B2; R += A*T``, which saves one matrix multiply per
additional term.

Reports ``num_groups()`` and ``num_eliminated()`` counts.

Allocation and Distribution Passes
====================================

These passes handle deferred tensor allocation and distributed computing.

DistributionPlanning
---------------------

Decides per-tensor whether to replicate (copy to all ranks) or block-distribute
(partition along the largest dimension). On single rank, this is a no-op.

Reports ``num_distributed()`` and ``num_replicated()`` counts.

See :doc:`distributed` for details.

Materialization
----------------

For each deferred tensor (from ``declare_tensor()``), inserts ``Materialize``
and ``Initialize`` nodes just before the tensor's first use. Memory is
allocated during ``execute()``, not during the pass itself. This enables
lazy, just-in-time allocation.

Reports ``num_materialized()`` and ``num_initialized()`` counts.

See :doc:`workspace` for details.

FreeInsertion
--------------

Inserts ``Free`` nodes after each intermediate tensor's last consumer. When
executed, Free nodes call ``Tensor::release()`` to immediately free the backing
storage, reducing peak memory for graphs with many large intermediates.

Key behaviors
^^^^^^^^^^^^^

- **Only frees intermediates**: user-provided tensors (inputs and outputs) are never freed.
- **Size threshold**: only frees tensors above 1 MB by default, configurable via the constructor. Small tensors are kept alive to avoid alloc/free overhead in loops.
- **Re-execution safe**: after release, the tensor retains its dimensions. The Materialize node at the tensor's first use re-allocates on the next execution. This makes it safe for Pipeline stages and SCF loops.
- **Loop overhead**: for re-executed graphs, large intermediates are freed and re-allocated each iteration. The overhead, around 1 ms for 100 MB, is negligible against compute time, and the memory saved by not keeping dead intermediates alive is significant.

.. code-block:: cpp

   // Default: free intermediates > 1MB
   pm.add<cg::passes::FreeInsertion>();

   // Custom threshold: only free intermediates > 100MB
   pm.add<cg::passes::FreeInsertion>(/*min_bytes=*/100 * 1024 * 1024);

   // Disable freeing (keep everything alive, useful for tight loops)
   pm.add<cg::passes::FreeInsertion>(/*min_bytes=*/SIZE_MAX);

Reports ``num_freed()`` count.

Memory Budget (DataflowExecutor)
----------------------------------

The ``DataflowExecutor`` supports an optional memory budget that limits
simultaneously live tensor data:

.. code-block:: cpp

   cg::DataflowExecutor df;
   df.set_memory_budget(2ULL * 1024 * 1024 * 1024);  // 2 GB limit
   graph.execute(df);

When a budget is set, the executor gates Materialize node submissions: if
allocating a new tensor would exceed the budget, the submitter thread waits
until Free nodes complete and release enough memory. Only the submitter blocks;
worker threads continue executing ready tasks.

Without a budget (default, ``set_memory_budget(0)``), the executor schedules
all tasks upfront for maximum parallelism.

Communication Passes
=====================

These passes optimize distributed communication (analogous to GPU transfer passes).

CommunicationInsertion
-----------------------

Inserts collective communication nodes (Allreduce, Broadcast, Allgather) where
needed between distributed operations.

CommunicationElimination
-------------------------

Removes redundant communication, for example a back-to-back allreduce of the
same tensor without intervening modification.

Reports ``num_eliminated()`` count.

CommunicationScheduling
-------------------------

Overlaps communication with computation using the ``async_start`` / ``async_finish``
mechanism (same pattern as async I/O and GPU transfers).

ContractionPlanning
====================

Replaces the analysis-only ``ChainParenthesization`` with a multi-objective pass
that considers:

- **FLOPs**: shape-dependent GEMM efficiency from HardwareProfile
- **Memory traffic**: roofline model (bandwidth-limited vs compute-limited)
- **Transfer costs**: host↔device when tensors cross GPU boundaries
- **Communication costs**: allreduce for distributed contractions
- **Device memory budget**: spill penalty when GPU memory is tight

Works with arbitrary-rank tensors, not just rank-2 matrices. Uses the
HardwareProfile database for architecture-specific cost estimation.

Reports per-chain: ``original_time_us``, ``optimal_time_us``, ``speedup``,
``comm_cost_us``, ``has_distributed``.

See :doc:`hardware_profiles` for how to provide calibrated performance data.

Recommended Pass Order
======================

.. code-block:: cpp

   // Use the default PassManager (recommended):
   auto pm = cg::PassManager::create_default();
   graph.apply(pm);

The default pipeline runs in this order. The GPU steps are added only when a GPU
backend or its mock is present, and the MPI steps only when an MPI backend or its
mock is present:

.. code-block:: text

    1. ConstantFolding          : fold constant subexpressions
    2. ScaleAbsorption          : absorb scale into next operation
    3. PermuteFusion            : fold pure axis reorders into einsum indices
    4. CSE                      : common subexpression elimination
    5. DeadNodeElimination      : remove unused intermediates
    6. ElementWiseFusion        : fuse consecutive element-wise ops
    7. LoopInvariantHoisting    : move invariants out of loops
    8. GEMMBatching             : collapse groups into blas::gemm_batch
    9. Reorder                  : memory-aware topological sort
   10. IOPrefetch               : move DiskReads early for async overlap
   11. DistributionPlanning     : decide replicate vs distribute
   12. Materialization          : insert allocation nodes for deferred tensors
   13. SymmetryPropagation      : tag intermediates whose symmetry is provable
       GPUPlacement             : decide CPU vs GPU per node (GPU only)
       TransferInsertion        : insert H2D/D2H transfer nodes (GPU only)
       TransferElimination      : remove redundant transfers (GPU only)
       GPUDiagnostics           : report GPU placement statistics (GPU only)
       StreamAssignment         : assign GPU streams for async (GPU only)
       InputSlicing             : carve distributed tensor inputs (MPI only)
       SUMMAExpansion           : expand distributed GEMMs (MPI only)
       CommunicationInsertion   : insert allreduce/broadcast (MPI only)
       CommunicationElimination : remove redundant communication (MPI only)
       CommunicationScheduling  : overlap communication with compute (MPI only)
   14. FreeInsertion            : insert Free nodes at last-consumer
   15. MemoryPlanning           : analyze tensor liveness
   16. ContractionPlanning      : multi-objective contraction ordering
   17. InplaceOptimization      : detect in-place candidates

.. code-block:: cpp

   // To access analysis results, use apply<> for individual passes:
   auto [modified, mem] = graph.apply<cg::passes::MemoryPlanning>();
   mem.print_report(std::cout);
