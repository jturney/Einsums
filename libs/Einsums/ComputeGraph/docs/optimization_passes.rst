.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _computegraph_optimization_passes:

===================
Optimization Passes
===================

The ComputeGraph provides a catalog of built-in optimization passes. The default
pipeline (``PassManager::create_default()``) adds 27 passes that always run, plus
five GPU passes and five distributed passes that are included only when a GPU or
MPI backend, or its mock, is available. One further pass,
``DistributiveFactoring``, is workload-specific and must be added by hand.

Most passes transform the graph. A few only measure it, and one,
``MemoryPlanning``, does both depending on how it is constructed.

Applying Passes
===============

.. code-block:: cpp

   // The usual entry point: default pipeline, plus a report of what it did.
   graph.optimize();
   std::cout << graph.explain();

   // Explicit optimization level.
   graph.optimize(cg::OptLevel::O1);

   // Single pass (returns a pair of modified flag + pass instance):
   auto [modified, cse] = graph.apply<cg::passes::CSE>();

   // Multiple passes via PassManager:
   cg::PassManager pm;
   pm.add<cg::passes::ScaleAbsorption>()
     .add<cg::passes::CSE>()
     .add<cg::passes::Reorder>();
   graph.apply(pm);

   // Default pipeline, built explicitly:
   auto pm = cg::PassManager::create_default();
   graph.apply(pm);

   // Apply to all stages of a pipeline:
   pipeline.apply(pm);

Optimization Levels
-------------------

``PassManager::create_for(OptLevel)`` builds a pipeline sized to how much
restructuring you want:

``O0``
    No passes at all.
``O1``
    Node-count cleanup only: ``ConstantFolding``, ``ScaleAbsorption``,
    ``PermuteFusion``, ``CSE``, ``DeadNodeElimination``, ``ElementWiseFusion``,
    then ``Materialization``. Cheap, no restructuring, no memory planning.
``O2``
    The full default pipeline. This is what ``graph.optimize()`` uses.

``Materialization`` is in ``O1`` because it is correctness-enabling rather than
an optimization: a graph that uses ``declare_tensor()`` cannot execute without
it. It runs after ``DeadNodeElimination`` so dead deferred tensors are never
allocated.

Phases
------

Every pass declares a ``PassPhase``, and the phase answers one question: may a
saved graph keep this pass's output, or must the output be re-derived on the
machine that loads it?

``analysis``
    Writes annotations onto tensors and nodes and never rewrites the node set.
    ``SymmetryPropagation`` and ``SpacePropagation``.
``structural-algebraic``
    Machine-independent rewrites of the mathematics.
    This is the only phase whose output is persisted, so a pass here may consult
    a cost model only as a hint: its output has to stay correct, if not optimal,
    under a different one.
``structural-resource``
    Changes the node set for machine-dependent reasons such as tiling, GPU
    placement, distribution planning, input slicing and SUMMA expansion.
``tuning``
    Schedule, memory, batching and thread decisions over a node set that is
    already final.
``diagnostic``
    Read-only reporting.
    ``CrossSpaceValidation``, ``ScalingAnalysis`` and ``GPUDiagnostics``.

The split is three-way on the structural side rather than two-way because
"changes the node set" and "is valid on another machine" are independent
questions.
``TiledExpansion`` and ``GPUPlacement`` rewrite the node set for reasons that
belong to the hardware, so calling them structural would write a machine's tile
shapes into a file and calling them tuning would let them run before the algebra
is settled.

Four factories build the phase views, each preserving the default pipeline's
relative order:

.. code-block:: cpp

   auto analysis   = cg::PassManager::analysis_pass_manager();   // analysis + diagnostic
   auto structural = cg::PassManager::structural_pass_manager(); // structural-algebraic
   auto resource   = cg::PassManager::resource_pass_manager();   // structural-resource
   auto tuning     = cg::PassManager::tuning_pass_manager();     // tuning

   // What a load of a saved graph replays over the persisted algebraic form.
   graph.apply(resource);
   graph.apply(tuning);

The four are views of ``create_default()``, not a re-planned pipeline, and the
default sequence is unchanged by their existence.
It is hand-ordered rather than derived from the phases, because the order
carries constraints the phase rule does not express: ``TiledExpansion`` runs
first so that every algebraic pass below it sees dense nodes rather than one
opaque ``Custom`` node, which is a deliberate and documented deviation from
"algebraic before resource".

Two mechanisms keep the labels honest.
``Graph::structure_version()`` counts changes to the node set, distinct from
``analysis_version()``, which bumps at every mutation-declaration point
including a plain re-sort.
``PassManager::run()`` watches that counter around every pass and throws when an
analysis or diagnostic pass moves it, and it re-runs the analysis passes once at
the end of a pipeline whose structure changed after they last looked, so the
annotations always describe the node set that will execute.

Runtime Controls
----------------

.. code-block:: bash

   # Skip named passes without rebuilding, to A/B whether one helps.
   ./my_program --einsums:pass:disable "ContractionPlanning,GEMMBatching"

   # Log node count and wall-clock time around every pass.
   ./my_program --einsums:pass:verbose

   # Run every pass in analysis-only mode: each reports what it found,
   # then the graph is restored. No modification persists.
   ./my_program --einsums:pass:analyze

   # Dump the algebra each region rewrite raised, before and after it rewrote.
   # The first thing to reach for when an optimized graph produces a wrong number.
   ./my_program --einsums:graph:dump-regions --einsums:pass:verbosity 2

In code, ``PassManager::set_verbosity(level)`` does the same as
``--einsums:pass:verbose`` and propagates the level to every pass already added
and every pass added afterwards. Level 1 reports totals, level 2 narrates each
modification, level 3 adds per-candidate detail.

``PassManager::disable(name)`` and ``enable(name)`` are the programmatic half of
``--einsums:pass:disable``, and exist because a driver that bisects a wrong
result runs the same pipeline once per pass and would otherwise have to mutate
process-global configuration to do it.
Switches are read at ``run()``, so a pass can be switched off before it is added.
An explicit ``enable`` overrides the option, because the more specific statement
about this pipeline is the one that should win.
A switch name matching no pass in the pipeline is reported through ``explain()``
and logged as a warning rather than doing nothing in silence.

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

       // Which phase this belongs to. The default is `tuning`, the phase whose
       // output is never saved, so a pass that skips this is safe rather than
       // silently persisted. See Phases above.
       cg::PassPhase phase() const override { return cg::PassPhase::StructuralAlgebraic; }

       // Opt in when the rewrite is safe applied independently to a loop body
       // or a conditional branch. The manager then drives the recursion.
       bool recurse_into_subgraphs() const override { return true; }

       // Clear per-apply counters. The manager calls this once per apply(),
       // NOT once per run(): with recursion enabled, run() is called once per
       // subgraph, and resetting there would report only the last one.
       void reset_stats() override { _num_rewrites = 0; }
   };

Two things bite when writing a rewrite:

- **Rebuild the executor, do not just edit ``Node::inputs``.** Captured
  executors resolve operands through ``TensorSlot`` pointers baked in at capture
  time, so editing the declared I/O changes the schedule but not the
  computation. Einsum nodes are rebuilt through ``Graph::make_einsum_node``.
- **Counters are per-apply, not per-run.** See ``reset_stats()`` above.

Graph-Transforming Passes
=========================

TiledExpansion
--------------

Lowers tiled operations into per-tile **dense** nodes, and runs first in the
default pipeline for exactly that reason: a tiled contraction captures as one
opaque ``OpKind::Custom`` node that no other pass can read, so expanding it is
what puts the tiles in front of CSE, ContractionPlanning, GEMMBatching,
InplaceOptimization and MemoryPlanning.

See :doc:`tiled` for the full story: sparsity handling, the densification
trade, elementwise fusion, and the node budget.

Reports ``num_expanded()``, ``num_tile_nodes()``, ``num_declined()``,
``num_screened()``, ``num_densified()``, ``num_fused()`` and
``num_gathers_reused()``.

DeltaElimination
----------------

Contraction against a Kronecker delta is a rename, so the pass does the rename
and drops the delta::

    tmp[i,j] = A[i,k] delta[k,j]     ->  (nothing; consumers read A directly)
    C[i,l]  += tmp[i,j] D[j,l]       ->  C[i,l] += A[i,j] D[j,l]

Two nodes and one intermediate become one node. Machine-generated input,
spin-summed equations in particular, produces these in bulk, and each one is a
full-size GEMM against an identity matrix, so what is saved is a whole
contraction rather than a constant factor on one.

The rewrite is **bitwise-exact**, which is why it runs by default and needs no
opt-in: ``sum_k A[i,k] * I[k,j]`` has exactly one nonzero term, every other
product is exactly ``0.0``, and adding exact zeros into a running sum changes
nothing. The result is the same float, not a nearby one.

That claim has one deliberate hole. If any ``A[i,k]`` with ``k != j`` is infinite
or NaN, the captured form computes ``0.0 * inf``, which is NaN, and the sum is
poisoned; the rewritten form returns ``A[i,j]`` unharmed. The two genuinely
differ on non-finite input. This is accepted rather than guarded because the
difference is one-directional: the rewrite only ever *removes* a NaN the
arithmetic had no reason to produce, and can never introduce one.

Recognition is by tag; see ``ProvenancePropagation`` below. A tensor that happens
to hold an identity but carries no tag is left alone.

What happens to the contraction's output is decided by the escape rule rather
than by a heuristic. An output nothing outside the region can observe is
dissolved outright and its readers repointed at the surviving operand. One that
escapes, because the caller holds it or something outside reads it, still has to
be produced, so the contraction becomes a ``Permute`` carrying the reordering and
both prefactors.

Declined, each through the skip tally: a delta whose two letters are both free in
the output (a diagonal extraction), both contracted (a trace), a conjugated
surviving operand, and any rename whose result does not carry exactly the
output's letters.

Self-gating and cheaply so: the pass declines before forming a single region when
no tensor in the graph is declared an identity, which is every graph nobody has
annotated.

Reports ``num_eliminated()`` and ``num_dissolved()``.

ConstantFolding
---------------

Identifies nodes whose inputs are all constant: graph-owned intermediate
tensors (``is_intermediate=true``) that are never written by any node. User-owned
tensors are not assumed constant because they may change between loop iterations
or successive ``execute()`` calls. This makes the pass safe for both one-shot
graphs and loop bodies.

Propagation follows the dependency chain: if node A is folded, nodes depending
only on A's outputs are also foldable.

.. note::

   This pass has side effects. It executes folded nodes during the pass itself.
   It is included in the default pipeline and is safe for Pipeline loop bodies.

Reports ``num_folded()``.

ScaleAbsorption
---------------

Absorbs ``Scale(α, C)`` into any subsequent operation that writes to ``C``
with a zero beta/c_prefactor:

- **Einsum**: ``c_prefactor=0`` becomes ``c_prefactor=α``
- **Gemm**:  ``beta=0`` becomes ``beta=α``
- **Permute**: ``beta=0`` becomes ``beta=α``

The scale node is removed from the graph and its effect folded into the
following operation's prefactor. Reports ``num_absorbed()``.

PermuteFusion
-------------

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

Runs before CSE/DeadNodeElimination so duplicate permute-einsum patterns
collapse into the same fused node, and before Materialization and GPU placement
so those passes do not allocate or place tensors that are about to disappear.

Reports ``num_candidates()``, the number of detected pairs, and
``num_rewrites()``, the number that passed the safety filter.

LayoutAssignment
----------------

Chooses the storage order of every graph-owned intermediate so the contractions
that touch it read their operands flat.

A vendor GEMM reads ``A`` as a flat ``(M,K)`` matrix and ``B`` as a flat
``(K,N)`` one. An operand whose contracted letters sit at one **end** of its
index list has such a reading already, and the transpose flag settles which end.
An operand whose contracted letters are **interleaved** with its free ones has no
flat reading at all, so the kernel copies it into one first. That copy is a whole
tensor of traffic per replay, it never appears as a node, and it is the only cost
in a contraction that storage order decides.

An intermediate the graph owns has no storage order anyone promised, so the order
is free to choose:

.. code-block:: text

   W[i,j,x] = A[i,k]   B[k,x,j]     // C's free groups disagree with B's -> B is copied
   R[i,x,y] = W[i,j,x] D[j,y]       // W's contracted letter is between its free ones -> W is copied

Storing ``W`` as ``(i,x,j)`` removes both copies. Neither contraction can see the
other's cost, which is why this is not a peephole: ``PermuteFusion`` is the local
form of the same idea and declines the moment a tensor has two readers that
disagree.

Candidates
^^^^^^^^^^

A tensor is a decision variable when it is rank three or more, still deferred,
and the escape analysis reports nothing outside the graph can observe it. Rank
two is excluded because a matrix has two readings and BLAS takes either one
through ``transa``, so there is nothing to win. A materialized tensor is excluded
because re-laying out a live buffer is a data movement this pass does not
perform. Any use it cannot rewrite - a node with no index list, an operand that
repeats a letter, a contraction carrying a letter that is neither free,
contracted nor batched - pins the tensor, and says so in the skip tally.

Reports ``num_relaid_out()``, ``num_copies_removed()`` and
``estimated_saving_us()``.

CSE: Common Subexpression Elimination
-------------------------------------

**Pattern**: Two nodes with identical ``OpKind``, ``inputs``, and ``OpData``.

**Result**: Second node removed; its outputs redirected to first node's outputs.

DeadNodeElimination
-------------------

Removes nodes whose outputs are all graph-owned intermediates (``is_intermediate=true``)
with no consumers. This is useful after CSE or other passes eliminate consumers,
because the original producer may then become dead.

Control flow, memory, and side-effect nodes are never eliminated.

Reports ``num_eliminated()``.

SymmetrizedAccumulation
-----------------------

Folds the CCSD permutational-symmetrization idiom
:math:`r_2 \mathrel{+}= s\,(t + P(t))`. Chemistry residuals accumulate a tensor
and its index-swapped transpose into the same output, which appears in the graph
as four nodes per site:

.. code-block:: text

   einsum(spec, A, B)            -> tmp     // tmp = pf*(A x B)   (kept)
   axpby(s, tmp,  1.0, r2)                  // r2 += s*tmp
   permute("jiba <- ijab", tmp)  -> tmpP    // tmpP = P(tmp)      (an involution)
   axpby(s, tmpP, 1.0, r2)                  // r2 += s*P(tmp)

The rewrite makes the permute accumulate **directly** into the output,
:math:`r_2 = r_2 + s_2 P(t)` through the existing ``string_permute`` kernel with
``beta = 1``, and deletes the second axpby along with the ``tmpP`` buffer. No
new kernel and no new ``OpKind``.

Runs after DeadNodeElimination and before ElementWiseFusion, which would
otherwise compose the two axpby and hide the pattern. Recurses into loop bodies,
since the CCSD residual is captured as one.

**What it buys:** the :math:`O(o^2v^2)` ``tmpP`` buffer, so peak memory. Replay
time is roughly compute-neutral, because the transpose itself is kept.

**Limitations:** runtime tensors of a single dtype only; the matched permute must
be involutive with ``alpha == 1``; scratch reused across sites matches per
generation, and an interference guard rejects sites where another node observes
the half-symmetrized output.

Reports ``num_candidates()``, ``num_matched()`` and ``num_rewritten()``.

ElementWiseFusion
-----------------

Fuses consecutive element-wise operations on the same tensor. Currently handles
consecutive ``Scale`` operations:

**Pattern**: ``Scale(2.0, A)`` followed by ``Scale(3.0, A)``

**Result**: Merged into ``Scale(6.0, A)``, executing both lambdas sequentially.

Reports ``num_fused()``.

LinearCombinationContractionFolding
-----------------------------------

Folds transpose-paired contractions into one, the CCSD "2J-K" idiom. Sibling of
``DistributiveFactoring``: where that pass sums *different* operands sharing the
*same* index pattern, this one folds contractions that reuse the *same* operand
tensor read with *permuted* index patterns.

Because einsum is linear in each operand,

.. math::

   \sum_k \alpha_k\,\mathrm{einsum}(\mathrm{spec}_k, A, B)
     = \mathrm{einsum}(\mathrm{spec}_0, A,\; \sum_k \alpha_k P_k(B))

where :math:`P_k` permutes :math:`B` into operand-0's layout. The rewrite builds
:math:`L = \sum_k \alpha_k P_k(B)` once and runs a single contraction, trading
:math:`N` flop-bound contractions for one contraction plus :math:`N-1` cheap
memory-bound permute/axpy steps.

.. code-block:: text

   Before:  Fae[a,e] += 2 * t1[m,f] * ovvv[m,a,f,e]
            Fae[a,e] -= 1 * t1[m,f] * ovvv[m,a,e,f]

   After:   L[m,a,f,e] = 2*ovvv[m,a,f,e] - ovvv[m,a,e,f]   (permute + axpy)
            Fae[a,e]  += t1[m,f] * L[m,a,f,e]              (single einsum)

Runs before ``LoopInvariantHoisting`` deliberately: the :math:`L` build is its
own node whose only input is the paired operand, so when that operand is
loop-invariant, the common case of an integral block from one-time setup,
hoisting lifts the builder out of the loop and :math:`L` is built once instead
of every replay.

**Limitations:** two-input einsums only; every non-first member must purely
accumulate (``c_prefactor == 1``) so the reassociation is exact; conjugated
einsums are skipped; all operands must be runtime tensors of one dtype; on a real
dtype the prefactors must be real-valued. An interference guard rejects the group
if any node between the first and last member touches the output or the operands.

Reports ``num_groups()`` and ``num_eliminated()``.

LoopInvariantHoisting
---------------------

For each Loop node, identifies operations inside the body whose inputs are
never modified by any other operation in the body. These loop-invariant
operations are moved before the loop to execute only once.

Self-modifying operations (scale, axpy, element_transform) are never hoisted
since they read and write the same tensor.

Reports ``num_hoisted()``.

ScratchPrivatization
--------------------

Breaks the false dependencies that a reused scratch buffer creates. Captured
code that recycles one tensor through many unrelated write-read episodes, the
CCSD idiom ``for each term: tmp = contract(...); r2 += tmp``, serializes the
whole graph: every overwrite of ``tmp`` carries a WAR edge from the previous
episode's readers and a WAW edge from its writer, so operations that share no
data still run strictly one after another and a parallel executor finds no width.

The pass splits a tensor's accesses into *generations*, one per pure overwrite,
and renames the interior generations onto a bounded pool of clones, round-robin.
The **last** generation keeps the original tensor, which is what makes the
rewrite safe without knowing who else holds it: outside observers can only look
between executions, and between executions the original buffer holds exactly the
value it always did.

Clones are declared as graph-owned deferred intermediates, so Materialization
allocates them and FreeInsertion and MemoryPlanning manage them like any other
scratch. The pool is capped, since more clones than the machine has cores adds
memory without adding parallelism.

.. important::

   By default the pass rewrites only graphs that carry an installed executor.
   Under the built-in sequential replay the clones cost cache locality and buy
   nothing. Call ``Graph::set_executor`` **before** ``apply()``, or
   ``set_require_executor(false)`` to privatize unconditionally.

A tensor is left alone whenever renaming cannot be proven safe locally: any
access through a view, any read before the first pure overwrite, any touch by a
control-flow or lifecycle node or by a node kind the pass cannot rebuild, or
fewer than two generations.

Reports ``num_tensors_privatized()``, ``num_copies_created()`` and
``num_nodes_rebuilt()``.

ContractionPlanning
-------------------

Multi-objective contraction ordering, using the shared ``CostModel``. It
considers:

- **FLOPs**: shape-dependent GEMM efficiency
- **Memory traffic**: roofline model (bandwidth-limited vs compute-limited)
- **Transfer costs**: host-device when tensors cross GPU boundaries
- **Communication costs**: allreduce for distributed contractions
- **Device memory budget**: spill penalty when GPU memory is tight

Works with arbitrary-rank tensors, not just rank-2 matrices, and declares the
intermediates it introduces as **deferred**. That is why it belongs in the
planning phase rather than at the end: it must precede GEMMBatching and Reorder,
which schedule the final node set, and DistributionPlanning and Materialization,
which size and allocate its intermediates.

Reports ``chains_restructured()``, ``intermediates_created()``, and per chain
``original_time_us``, ``optimal_time_us``, ``speedup``, ``comm_cost_us`` and
``has_distributed``.

See :doc:`hardware_profiles` for how to provide calibrated performance data.

GEMMBatching
------------

Collapses groups of independent GEMM-pattern einsum nodes at the same
dependency level into a single ``OpKind::BatchedGemm`` node whose executor
calls ``blas::gemm_batch<T>``. One BLAS dispatch covers the whole batch
instead of N. This is a substantial win for workloads that issue many
small contractions: stacked attention heads, per-sample transforms,
Kronecker-factored updates, and batched chemistry kernels. See
:doc:`gemm_batching` for a full walk-through and timing numbers.

**Candidate:** a 2D x 2D -> 2D einsum with exactly one link index. Capture
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

Must stay **before** DistributionPlanning, which reads ``EinsumDescriptor`` on
every node: BatchedGemm nodes are not inspected by the distribution or GPU
passes, so an einsum that needs those optimizations should not be batched first.

Reports ``num_batches()``, ``total_batched()`` and ``num_gate_skipped()``, the
groups left parallel by the profitability gate.

Reorder
-------

**Algorithm**: Memory-aware Kahn's algorithm with priority queue.

Among ready nodes, schedules the one that frees the most memory first.
Reduces peak memory by releasing large intermediates earlier.

IOPrefetch
----------

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

Reports ``num_prefetched()``.

SymmetryPropagation
-------------------

Walks the graph and tags intermediate tensors whose symmetry can be
proven from their inputs, pushing the inferred ``SymmetryDescriptor`` to
the backing tensor so the rank-2 BLAS dispatch fires at
``graph.execute()``. Current rules: scale preserves symmetry; axpy/axpby
with same-descriptor operands preserves it; a rank-2 self-contraction
(:math:`A^{T}A` or :math:`AA^{T}`) produces a symmetric result; a permute of a
symmetric tensor stays symmetric.

Runs after Materialization, so the tensors exist, and before GPU placement, so
downstream passes and executions see the inferred symmetry. Only mutates
graph-owned intermediates.

Reports ``num_inferred()``. See :doc:`symmetry` for the full story.

SpacePropagation
----------------

Fills in the index spaces of graph-owned intermediates from the annotations their
producers' operands carry.
An index space is what an axis ranges over, occupied orbitals or virtuals or an
auxiliary basis, and it is declared per tensor slot with
``Graph::annotate_spaces``.
A method annotates its handful of persistent inputs, and this pass carries the
annotation through the intermediates, so the effort is proportional to a
program's inputs rather than to its node count.

Current rules: an einsum output slot takes the space of the letter that produced
it; a scale inherits its input's spaces, since scaling changes values and never
what an axis ranges over; a permute or transpose reorders the input's spaces by
its own index letters; an ``Axpby`` takes the spaces every annotated input agrees
on, slot by slot.

.. code-block:: cpp

   graph.annotate_spaces(A, {occ, virt});
   graph.annotate_spaces(B, {virt, aux});
   auto [modified, spaces] = graph.apply<cg::passes::SpacePropagation>();
   // spaces.num_inferred() == the number of intermediates annotated

One sweep in topological order is a fixpoint: a node is visited after every node
that writes its inputs, so a chain of contractions resolves end to end in a
single run and a second run infers nothing new.

Only graph-owned intermediates are annotated, and only when a tensor has exactly
one writer in this graph and is not referenced by a child sub-graph, the same
soundness rule ``SymmetryPropagation`` uses.
An annotation the user declared is never overwritten; one that capture or an
earlier sweep inferred may be refined, and each handle records which of the two
it holds so a validation pass can report a weaker verdict on a derived slot.
Nothing is ever pushed back onto an unannotated input: inheritance across
operands is off, because it is how one wrong annotation spreads silently.
Operands that disagree about a letter are declined and counted in
``skip_reasons()`` rather than raised, since diagnosing a cross-space conflict
belongs to a validation pass.

Runs beside ``SymmetryPropagation``, after Materialization and before the backend
passes. The two are independent analyses and neither reads the other's output.

Reports ``num_inferred()``.

ProvenancePropagation
---------------------

Carries a tensor's **provenance tag** across the operations that preserve its
identity.

A tag says what a tensor *is*, which is a different question from how big it is
and the one a pass that wants to *recognize* something has to ask. Index spaces
cannot answer it: they say an axis ranges over virtuals, not that this particular
matrix is a Kronecker delta or a Coulomb metric. Declare one with
``Graph::annotate_tag``, or from Python::

    cg.annotate(delta, tag="identity")
    cg.annotate(eri, spaces=("occ", "occ", "virt", "virt"),
                tag={"name": "eri", "basis": "cc-pvdz"})

The vocabulary is open and unvalidated, because the set of things worth
recognizing grows with the passes that recognize them. The cost is that a
misspelled name is a tag nothing matches, and the only symptom is a pass that
quietly finds no candidates; a pass is expected to report that through its skip
tally.

A tag is a **declaration** and is never inferred from a tensor's contents. A pass
could look at the data and notice a tensor happens to hold an identity today, and
that would be wrong in a way easy to miss: a structural-algebraic rewrite is what
a saved graph keeps, and a later bind may supply a different tensor under the
same manifest name.

What propagates, and what deliberately does not: only the axis reorderings,
``Permute``, ``Transpose`` and ``HPTTPermute``. A **view** does not, and the
reason is worth stating because the obvious rule gets it wrong. "Views, permutes
and copies preserve identity" is false for the tag that matters most here: a view
of a Kronecker delta is an identity only when the slice is a square block on the
diagonal, and ``delta[0:2, 0:4]`` is an ordinary rectangular matrix of ones and
zeros. Eliminating a contraction against that would produce a wrong number rather
than a slower one. So the rule is whole-tensor. That is conservative for tags
whose slices genuinely would inherit, and the trade is deliberate: a tag that is
missing costs a pass one candidate, while a tag that is wrong costs a wrong
answer and no later pass can tell the difference.

An inferred tag never overwrites a declared one. A disagreement is reported
through the skip tally rather than resolved by picking a winner.

Reports ``num_propagated()``.

CrossSpaceValidation
--------------------

Flags a contraction letter that binds a slot of one index space against a slot of
another.
The bug it exists to catch is a letter tying an ``occ`` slot of one operand to a
``virt`` slot of the next.
In a coupled-cluster transcription that is almost always a mistake, and it is one
the differential fuzzers cannot catch, because a hand-derived reference and its
implementation are self-consistently wrong together.

Capture already raises when a letter binds two different declared spaces within
one contraction, so this pass exists for everything capture cannot see:
annotations that arrive after capture through ``Graph::annotate_spaces``, which
capture never revisits; annotations ``SpacePropagation`` inferred, which that pass
declines to argue about by design; and verdicts that need the registry's declared
relations rather than a comparison of two ids.

Each letter is re-derived from the operands' current handle annotations, and every
slot binding it is compared against the first.
Two slots naming the same space report nothing.
Two spaces declared disjoint are an **error**, because the contraction as written
sums over an empty intersection and is identically zero.
One space contained in the other is a **note**, since contracting a ``pno`` slot
against a ``virt`` slot is a restriction of the parent space rather than a
mistake, and it is listed only because an unintended restriction looks exactly
like an intended one.
Anything the registry cannot relate is a **warning**, because "unknown" is a
first-class answer that has to be treated as carefully as "no".

A finding involving a slot whose annotation was *inferred* rather than declared is
reported one severity level lower and says so in its message.
An authoritative verdict resting on a derived premise is worse than a weak verdict
resting on a firm one.

.. code-block:: cpp

   graph.annotate_spaces(A, {occ, virt});
   graph.annotate_spaces(B, {occ, aux});   // 'a' meets virt on A and occ on B
   auto [modified, check] = graph.apply<cg::passes::CrossSpaceValidation>();
   for (auto const &finding : check.findings()) {
       std::cerr << finding.message << '\n';
   }

The pass never throws, never mutates, and never fails a pipeline.
An unannotated program yields nothing at all, which is the honest answer when the
registry has no premises to reason from.

Reports ``num_errors()``, ``num_warnings()``, ``num_notes()`` and ``findings()``,
with the full detail available through ``print_report(std::ostream &)`` or, for a
caller that is not holding a stream, ``report_string()``.

ScalingAnalysis
---------------

Reports how a program scales: every contraction's symbolic cost, every
intermediate's symbolic size, the rate-limiting term and a bound on the memory
footprint, all as polynomials in index-space scales rather than as numbers.
It is the cost model of the algebraic optimizer delivered as a user-facing feature
first, and it is the natural check that a program's space annotations are the ones
its author meant.

A node's cost comes from the spaces its operands carry when the pass runs, not
from the map frozen into the node at capture, so the report reflects a declaration
that arrived late and everything ``SpacePropagation`` inferred immediately before
it.
Letters that no annotation reaches become anonymous per-letter variables, so an
unannotated program still yields a complete report, just one written in ``?i``
rather than in ``o``.

.. code-block:: cpp

   auto [modified, scaling] = graph.apply<cg::passes::ScalingAnalysis>();
   scaling.total_flops().to_string(&registry);   // e.g. "2*o^2*v^4"
   scaling.rate_limiting();                      // the node(s) carrying that term
   scaling.print_report(std::cout);

The rate-limiting verdict ranks flops through the same total order the structural
passes use, so two nodes tie only when their polynomials are literally identical
and the answer is the same in every process.
``memory_bound()`` is the **sum** of the intermediate sizes, which is an upper
bound on the high-water mark and not the high-water mark itself; a liveness-aware
figure needs the interval analysis ``MemoryPlanning`` does for bytes, applied to
polynomials, and that is a later task.
A loop body is analysed once rather than once per iteration, since a trip count is
a runtime quantity, and every entry carries its graph's name so a body's
contribution stays attributable.

Only contraction nodes are costed.
Every other kind is counted in ``skip_reasons()`` rather than given an invented
formula.

Reports ``node_costs()``, ``intermediate_sizes()``, ``total_flops()``,
``total_traffic()``, ``memory_bound()``, ``rate_limiting()`` and
``num_unannotated_nodes()``, with the full table available through
``print_report(std::ostream &)`` or ``report_string()``.
A caller that cannot hold a ``SymbolicPoly`` reads the same numbers as text
through ``total_flops_str()``, ``node_flops()``, ``intermediate_sizes_str()`` and
``rate_limiting_labels()``.

StreamContractionFusion
-----------------------

Fuses contractions that stream the same large tensor into one pass over it: the
SCF "J and K from one TEI read" idiom.

It detects einsum nodes that each contract the same large tensor :math:`S`
against a small operand, where every output index and every small-operand index
is drawn from :math:`S`'s index pattern. Each element of :math:`S` is then read
exactly once, so such contractions are memory-bandwidth bound and their cost is
one stream of :math:`S`. N of them cost N streams executed separately, but only
**one** stream when fused: the replacement node walks :math:`S` once in storage
order and feeds every member's accumulator from the same element.

.. code-block:: text

   Before (two 800 MB streams of the TEI at n = 100, plus the scrambled
   K pattern running below stream speed):
       J[mu,nu] += 2 * TEI[mu,nu,lam,sig] * D[lam,sig]
       K[mu,nu] -= 1 * TEI[mu,lam,nu,sig] * D[lam,sig]

   After: one storage-order stream of TEI updating both J and K.

**Output handling.** Outputs are normally accumulated in thread-private buffers
and reduced at the end, which requires them to stay cache-resident: the cap is
derived from the ``CostModel``'s cache hierarchy as last-level-cache / threads
bytes per output. When a member's output exceeds the cap, the group switches to
owner-computes chunking rather than declining. Partitioning a physical
:math:`S` axis whose label lands in the output pins one output coordinate, so
threads owning disjoint blocks write disjoint output slices directly, with no
private copy, no reduction and no size cap. The kernel partitions the
highest-stride covering axis, because a low-stride partition turns each thread's
read into a strided comb, measured about 5x slower than contiguous slabs.

**Relationship to LCCF.** Both serve the 2J-K algebra. LCCF materializes a
linear combination :math:`L` and contracts once, measured 2.7x over unfused for
the Fock idiom, but it still makes roughly four passes over :math:`S`-sized data.
This pass replaces those with a single pass, and when both are registered it
runs first, consuming the pattern LCCF would otherwise fold.

**Limitations:** two-input, non-conjugated einsums with no repeated index in any
of C/A/B; all three operands runtime tensors of one dtype; real dtypes require
real prefactors; distributed operands decline, since they belong to the
communication passes; :math:`S` needs at least 4096 elements and must be at
least 8x larger than each member's small operand and output.

Runs after Materialization, so its size thresholds read real dims, and after
SymmetryPropagation, which inspects the einsums it may consume.

InplaceOptimization
-------------------

In-place storage merging for elementwise consumers. When an element-aligned
elementwise node (``DirectProduct``, ``DirectDivision``, or ``Axpby`` with
``beta == 0``) pure-overwrites a graph-owned intermediate while reading another
graph-owned intermediate for the **last** time, the output reuses the dying
input's storage: the graph metadata merges the two ids CSE-style, the output's
lifecycle nodes are dropped, and its executor slot is durably redirected. One
buffer allocation and its write traffic disappear per merge. The CC
amplitude-update pattern, ``R -> Tnew = R * invD``, is the canonical win.

Runs before FreeInsertion and MemoryPlanning so each merge removes a buffer and
shortens the intervals those liveness passes then work with.

**Soundness guards.** Only whitelisted consumers, whose output element ``i``
depends solely on element ``i`` of the inputs, may alias output with input;
contractions and permutes never qualify. Both tensors must be graph-owned
non-viewed intermediates with identical dims and byte size, the source with
exactly one producer and dying at the consumer, the destination with exactly one
writer and no ``Initialize``. Graphs with control flow at this level are skipped,
since bodies reference parent tensors invisibly to plain use-counts, and bodies
are processed on their own recursion level. GPU-placed graphs are skipped,
because device shadows swap buffers behind the slots.

Reports ``num_merged()``, and ``num_candidates()``, which is still the
single-producer/single-consumer census this pass exposed when it was
analysis-only.

FreeInsertion
-------------

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

Reports ``num_freed()``.

MemoryPlanning
--------------

Liveness analysis, memory statistics, and the planned host arena. It is the last
pass in the default pipeline, running after Materialization and FreeInsertion so
the intervals it computes are the final ones.

With ``apply_arena`` set, which is the default and the default-pipeline
configuration, the liveness intervals become placement: non-overlapping
graph-owned intermediates get first-fit-decreasing offsets in one shared,
64-byte-aligned arena per graph, sized to the peak. Two intermediates whose
lifetimes do not overlap then share the same bytes instead of each holding their
own allocation.

.. code-block:: cpp

   auto [modified, mem] = graph.apply<cg::passes::MemoryPlanning>();
   mem.print_report(std::cout);

   // Analysis only: compute and report the statistics, place nothing.
   pm.add<cg::passes::MemoryPlanning>(/*apply_arena=*/false);

Only intermediates that are materialized, not viewed or aliased, with
``materialize_into_fn`` and ``release_fn`` and a nonzero byte size, are
arena-placed. Device statistics are reporting-only; no device arena is applied.

Reports ``total_memory()``, ``peak_memory()``, ``num_planned()``,
``planned_arena_bytes()`` and ``planned_tensor_bytes()``.

Opt-In Passes
=============

DistributiveFactoring
---------------------

A workload-dependent rewrite that is **not** in the default pipeline. Add it by
hand when the pattern it matches is in your graph.

Detects groups of einsums accumulating into the same output tensor with a
shared operand and rewrites them using the distributive property:

**Pattern**: ``R += A*B1; R += A*B2``

**Result**: ``T = B1 + B2; R += A*T``, which saves one matrix multiply per
additional term.

.. code-block:: cpp

   cg::PassManager pm;
   pm.add(std::make_shared<cg::passes::DistributiveFactoring>());
   graph.apply(pm);

Reports ``num_groups()`` and ``num_eliminated()``.

Allocation and Distribution Passes
==================================

DistributionPlanning
--------------------

Decides per-tensor whether to replicate (copy to all ranks) or block-distribute
(partition along the largest dimension). On single rank, this is a no-op.

Reports ``num_distributed()`` and ``num_replicated()``. See :doc:`distributed`.

Materialization
---------------

For each deferred tensor (from ``declare_tensor()``), inserts ``Materialize``
and ``Initialize`` nodes just before the tensor's first use. Memory is
allocated during ``execute()``, not during the pass itself. This enables
lazy, just-in-time allocation.

Reports ``num_materialized()`` and ``num_initialized()``. See :doc:`workspace`.

Memory Budget (DataflowExecutor)
--------------------------------

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

GPU Passes
==========

Included only when a GPU backend or its mock is available.

``GPUPlacement``
    Decide CPU vs GPU per node, using the shared ``CostModel``. A no-op when
    ``--einsums:gpu:disable`` is set.
``TransferInsertion``
    Insert host-to-device and device-to-host transfer nodes.
``TransferElimination``
    Remove redundant transfers.
``GPUDiagnostics``
    Report GPU vs CPU node counts, transfer bytes, and peak device memory.
``StreamAssignment``
    Assign GPU streams so transfers can overlap compute.

Communication Passes
====================

Included only when an MPI backend or its mock is available. These optimize
distributed communication, analogously to the GPU transfer passes.

``InputSlicing``
    Carve distributed tensor inputs into the local pieces each rank owns.
``SUMMAExpansion``
    Expand a distributed GEMM into its SUMMA panel schedule.
``CommunicationInsertion``
    Insert collective nodes (Allreduce, Broadcast, Allgather) where needed
    between distributed operations.
``CommunicationElimination``
    Remove redundant communication, for example a back-to-back allreduce of the
    same tensor with no intervening modification. Reports ``num_eliminated()``.
``CommunicationScheduling``
    Overlap communication with computation through the
    ``async_start``/``async_finish`` mechanism, the same pattern used for async
    I/O and GPU transfers.

See :doc:`distributed` for the full distributed story.

The Default Pipeline
====================

.. code-block:: cpp

   graph.optimize();                 // the usual entry point
   // or, explicitly:
   auto pm = cg::PassManager::create_default();
   graph.apply(pm);

The default pipeline runs in this order. The GPU steps are added only when a GPU
backend or its mock is present, and the MPI steps only when an MPI backend or its
mock is present:

.. code-block:: text

    1. ProvenancePropagation     : carry tensor tags across the graph
    2. TiledExpansion            : lower tiled ops into per-tile dense nodes
    3. DeltaElimination          : substitute away contractions with a delta
    4. ConstantFolding           : fold constant subexpressions
    5. ScaleAbsorption           : absorb scale into the next operation
    6. PermuteFusion             : fold pure axis reorders into einsum indices
    7. CSE                       : common subexpression elimination
    8. DeadNodeElimination       : remove unused intermediates
    9. SymmetrizedAccumulation   : fold r += s*(t + P(t)) sites
   10. ElementWiseFusion         : fuse consecutive element-wise ops
   11. LinearCombinationContractionFolding : fold transpose-paired contractions
   12. DistributiveFactoring     : factor a shared operand out of a sum
   13. LoopInvariantHoisting     : move invariants out of loops
   14. ScratchPrivatization      : rename reused scratch onto clones
   15. LayoutAssignment          : store intermediates so contractions read flat
   16. ContractionPlanning       : multi-objective contraction ordering
   17. GEMMBatching              : collapse groups into blas::gemm_batch
   18. Reorder                   : memory-aware topological sort
   19. IOPrefetch                : move DiskReads early for async overlap
   20. DistributionPlanning      : decide replicate vs distribute
   21. Materialization           : insert allocation nodes for deferred tensors
   22. SymmetryPropagation       : tag intermediates whose symmetry is provable
   23. SpacePropagation          : infer index spaces on intermediates
   24. CrossSpaceValidation      : flag letters binding two different spaces
   25. ScalingAnalysis           : report cost polynomials and the limiting term
   26. StreamContractionFusion   : one pass over a streamed tensor, not N
       GPUPlacement             : decide CPU vs GPU per node       (GPU only)
       TransferInsertion        : insert H2D/D2H transfer nodes    (GPU only)
       TransferElimination      : remove redundant transfers       (GPU only)
       GPUDiagnostics           : report GPU placement statistics  (GPU only)
       StreamAssignment         : assign GPU streams for async     (GPU only)
       InputSlicing             : carve distributed tensor inputs  (MPI only)
       SUMMAExpansion           : expand distributed GEMMs         (MPI only)
       CommunicationInsertion   : insert allreduce/broadcast       (MPI only)
       CommunicationElimination : remove redundant communication   (MPI only)
       CommunicationScheduling  : overlap communication w/ compute (MPI only)
   27. InplaceOptimization       : merge outputs into dying inputs
   28. FreeInsertion             : insert Free nodes at last-consumer
   29. MemoryPlanning            : liveness analysis + the host arena

Reading the results
-------------------

``graph.explain()`` harvests the counters every pass exposes into one report:

.. code-block:: cpp

   graph.optimize();
   std::cout << graph.explain();

.. code-block:: text

     - PermuteFusion: folded 14 permute(s) into contractions
     - ScratchPrivatization: 3 scratch tensor(s) split onto 12 clone(s), 47 node(s) rebuilt
     - ContractionPlanning: restructured 2 of 6 GEMM chain(s), 2 intermediate(s)
         chain of 3: est. 812.4us -> 233.1us (3.49x)
     - GEMMBatching: 4 batch(es) absorbing 96 GEMM(s); 1 group(s) left parallel by the profitability gate
     - MemoryPlanning: peak 184.20 MB of 512.75 MB total
         arena: 184.20 MB hosting 31 intermediate(s) (512.75 MB of buffers)

For an individual pass's own getters, apply it directly:

.. code-block:: cpp

   auto [modified, mem] = graph.apply<cg::passes::MemoryPlanning>();
   mem.print_report(std::cout);
