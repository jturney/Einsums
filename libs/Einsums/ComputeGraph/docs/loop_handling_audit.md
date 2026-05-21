# Loop-handling audit of ComputeGraph optimization passes

_Audit date: 2026-05-19. Scope: every pass in `src/Passes/*.cpp` (29 total)._

## Methodology

Walked every `src/Passes/*.cpp`, classified by:

1. Whether the source mentions `OpKind::Loop`.
2. What it does with that knowledge.
3. What loop semantics the pass *should* enforce.

Also checked `PassManager::run` itself — it does **not** recurse into loop bodies
(or conditional branches), so individual passes are responsible for their own
recursion.

Conditional bodies (`then_branch`/`else_branch`) have the same problem shape as
loops; this document focuses on loops but the same fixes apply.

## Legend

**Current behavior** — how the pass treats a `Loop` node today.

| Tag | Meaning |
|---|---|
| `oblivious` | Walks `graph.nodes()`, never reads `OpKind`, treats Loop as just another node (its inputs/outputs are the loop's I/O). |
| `opaque-skip` | Explicitly `if Loop continue;` — won't rewrite a Loop node, but doesn't look inside. |
| `peeks` | Reads `LoopDescriptor::body` for one specific purpose (e.g. pinning tensors) but doesn't run on it. |
| `recurses` | Actually runs its logic on body nodes. |

**Target behavior** — what the pass *should* do.

| Tag | Meaning |
|---|---|
| `body-yes` | Semantics should fire on the body too. |
| `body-no` | Body is irrelevant to this pass. |
| `body-hoist` | Should look inside but emit its output to the *parent* graph (alloc/free hoisting). |

**Severity**:

- **bug** — silently produces a wrong result today.
- **miss** — produces a correct-but-suboptimal result; optimization left on the table inside loops.
- **fine** — current behavior matches target.

## Pass-by-pass matrix

### Memory lifecycle (4 passes)

| Pass | Current | Target | Severity | Notes |
|---|---|---|---|---|
| **Materialization** | oblivious | body-hoist | **bug** | Scans `graph.tensors_map()` for `AllocState::Deferred`. Workspace-declared tensors used inside loop bodies live in the workspace, not the parent graph — pass never sees them. `body.declare_tensor()` deferred tensors would live in the body's map and the pass wouldn't visit the body. Allocs/inits must hoist to parent (allocate once before the loop) — naïve recursion would reallocate every iteration. |
| **FreeInsertion** | ~~oblivious~~ **body-hoist (FIXED, Step 3)** | body-hoist | ~~bug~~ **done** | Now walks descendants and hoists a body intermediate's single Free to the parent *after* the outermost enclosing loop/conditional (live across all iterations, freed once when the loop is done). Workspace tensors stay alive (`is_intermediate=false`). Also fixed a latent idempotency bug: Free nodes are excluded from the last-use scan so repeated runs don't double-free. See `Pass_FreeInsertion.cpp`. |
| **MemoryPlanning** | oblivious | body-yes (with loop-carried awareness) | **miss** | Reports memory totals based on parent nodes only. Loop body's footprint is invisible. Peak-memory numbers underreport when the body has intermediates. |
| **InplaceOptimization** | oblivious | body-yes | **miss** | Last-write-then-overwrite candidacy is computed flatly. Inside-body candidates aren't considered. No correctness risk; just leaves performance on the table. |

### Graph rewrites (12 passes)

| Pass | Current | Target | Severity | Notes |
|---|---|---|---|---|
| **CSE** | oblivious | body-yes | **miss** | Won't dedupe identical work inside a body. Safe (just suboptimal). |
| **DeadNodeElimination** | opaque-skip | body-yes-with-context | **bug** | Preserves the Loop node itself (correct). But body's nodes are never examined, and even if recursed, "dead inside body" must consult parent post-loop uses to decide truly-dead. |
| **ConstantFolding** | opaque-skip | body-yes-with-care | **miss/bug** | Correctly refuses to fold a Loop node. Folding *inside* a body needs to know which inputs change across iterations — only loop-invariant inputs are fold-safe. Today it never tries; safe-but-suboptimal. Naïve recursion without iteration-variance check would be a bug. |
| **Reorder** | oblivious | body-yes (per-body) | **miss/bug** | Topological reorder must not cross loop boundaries. Today it doesn't reorder body nodes at all (correct by accident). Future recursion must scope each reorder to within one body. |
| **ScaleAbsorption** | oblivious | body-yes | **miss** | Body's scale-into-next-op opportunities missed. Safe. |
| **ElementWiseFusion** | oblivious | body-yes | **miss** | Body's element-wise chains not fused. Safe. |
| **PermuteFusion** | oblivious | body-yes | **miss** | Body's permute-into-gemm/einsum not folded. Safe. |
| **GEMMBatching** | oblivious | body-no (single-iter) + maybe body-yes | **miss** | Across iterations is a different concept (no fixed batch dim). Inside one body, batching siblings is fine and untouched today. |
| **ChainParenthesization** | oblivious | body-yes | **miss** | Chain detection runs only on parent. A GEMM chain that lives entirely inside the body (typical) is invisible. |
| **DistributiveFactoring** | oblivious | body-yes | **miss** | Body factoring missed. |
| **SymmetryPropagation** | oblivious | body-yes-with-care | **miss/bug** | Symmetry annotations propagated through parent only. Inside a loop, propagation must consider that an "input" might be overwritten by a previous iteration. |
| **LoopInvariantHoisting** | recurses (loop-aware by design) | body-yes (target) | **fixed** | Designed loop-aware (reads `LoopDescriptor::body`, hoists invariants to parent) but only checked input-invariance, not single-writer outputs — could hoist a producer whose output was overwritten in the loop (miscompile). Now guarded; see Progress log. |

### GPU / transfer (5 passes)

| Pass | Current | Target | Severity | Notes |
|---|---|---|---|---|
| **GPUPlacement** | oblivious | body-yes | **miss/bug** | Placement decisions are flop-cost-based per node. Body nodes are invisible — a hot GEMM inside an SCF loop won't be placed on GPU at all. If it recurses, must place body nodes consistently with parent's decisions about loop I/O tensors. |
| **GPUDiagnostics** | oblivious | body-yes | **miss** | Reported H2D/D2H counts miss body transfers. |
| **StreamAssignment** | oblivious | body-yes | **miss** | Body nodes don't get stream IDs; runtime assigns default. |
| **TransferInsertion** | oblivious | body-yes-with-hoist | **bug** | Inserts H2D/D2H only at parent. A GPU op inside a body needs its inputs H2D'd — either every iteration (current: missing transfer, would crash) or hoisted before the loop (correct for loop-invariant inputs). |
| **TransferElimination** | **peeks** | body-yes | partial | Peeks into body to pin tensors so it doesn't D2H something the body reads. The *one* place outside LIH where loop-aware design appears today. Doesn't run on the body — body-internal redundant transfers wouldn't be cleaned. |

### Distributed (6 passes)

| Pass | Current | Target | Severity | Notes |
|---|---|---|---|---|
| **DistributionPlanning** | oblivious | body-yes (with consistency) | **bug** | Plans target_a/target_b/shared/link classification per einsum. Body einsums aren't classified — they fall back to replicated, breaking chained distributed contractions inside SCF/CC loops. Recursion must ensure body's distribution agrees with parent's. |
| **InputSlicing** | oblivious | body-yes | **bug** | Slicing temps for distributed inputs aren't created for body operations. |
| **SUMMAExpansion** | oblivious | body-yes | **bug** | A GEMM inside a body never gets SUMMA'd. |
| **CommunicationInsertion** | opaque-skip | body-yes-with-hoist | **bug** | Skips Loop nodes deliberately. Body allreduce/broadcast needs are invisible. |
| **CommunicationElimination** | oblivious | body-yes-with-care | **miss** | Doesn't see body allreduces, can't dedupe them. "already_reduced" tracking needs to respect that a body iteration may invalidate a previous reduce. |
| **CommunicationScheduling** | oblivious | body-yes | **miss** | iallreduce/wait splitting only happens at parent. Async overlap opportunities inside body missed. |

### I/O & analysis (2 passes)

| Pass | Current | Target | Severity | Notes |
|---|---|---|---|---|
| **IOPrefetch** | oblivious | body-yes-with-hoist | **bug** | Moves DiskRead to graph start. A DiskRead inside a body should be hoisted to before the loop if the file path is loop-invariant. Today it can't see body reads at all. |
| **ContractionPlanning** | oblivious | body-yes | **miss** | Chain detection same story as ChainParenthesization. Analysis-only so no correctness risk; blind to body chains. |

## Summary

**29 passes total:**

- **1 correct** (`LoopInvariantHoisting`)
- **1 partial / correct-for-its-purpose** (`TransferElimination` peeks but doesn't run)
- **27 don't handle loops** — of those:
  - **11 are bugs today**: silently wrong or unimplemented for the body — `Materialization`, `FreeInsertion`, `DeadNodeElimination`, `TransferInsertion`, `DistributionPlanning`, `InputSlicing`, `SUMMAExpansion`, `CommunicationInsertion`, `IOPrefetch`, and `Reorder`/`ConstantFolding` if anyone tries to recurse naïvely.
  - **16 are misses**: safe-but-suboptimal — `CSE`, `ScaleAbsorption`, `ElementWiseFusion`, `PermuteFusion`, `GEMMBatching`, `ChainParenthesization`, `DistributiveFactoring`, `SymmetryPropagation`, `MemoryPlanning`, `InplaceOptimization`, `GPUDiagnostics`, `StreamAssignment`, `GPUPlacement` (partial), `CommunicationElimination`, `CommunicationScheduling`, `ContractionPlanning`.

## Cross-cutting infrastructure recommendations

1. **Add `Graph::for_each_subgraph(visitor)`** that walks `LoopDescriptor::body`,
   `ConditionalDescriptor::then_branch`, and `else_branch`. Today every
   loop-aware pass open-codes the `std::get_if<LoopDescriptor>` dance.
2. **Decide whether `PassManager::run` recurses, or each pass does.** Either
   works; pick one. Today neither does, which is the worst combination — every
   pass writes its own opt-in.
3. **Introduce a per-pass `recursion_policy()` declaration** (`none` /
   `bodies-too` / `hoist-from-bodies`). Makes the contract explicit so reviewers
   can spot a new pass that picked the wrong one.
4. **Test infrastructure**: every `Pass_*.cpp` test file should grow a
   "…inside a loop body" test case. Today none do (except
   `LoopInvariantHoisting.cpp`).
   - **End-to-end correctness guard**: the `scf_simulation.py` and
     `mp2_simulation.py` examples now *assert* their reference energies
     (H2, HeH+) rather than only checking the exit code. This is what
     caught the CSE and Reorder divergences — a pass that silently corrupts
     the captured graph makes the SCF diverge, which a pass-level unit test
     with small/synthetic graphs can miss. Keep these assertions; they're a
     cheap whole-pipeline regression net.
   - **Promoted to unit tests**: `tests/unit/test_scf_python.py` and
     `tests/unit/test_mp2_python.py` import the example run-functions and
     assert the reference energies, so the whole-pipeline guard runs as part
     of the normal unit suite (not just the examples).
   - **Hardening test**: `tests/unit/LoopOptimizationCorrectness.cpp` runs
     the *full* default PassManager on cg-loop bodies with heavy in-place
     reuse (incl. the CSE/Reorder bait patterns), plus nested and sequential
     loops, comparing the executed result to an eager reference. Verified
     non-vacuous: flipping CSE back to recurse makes it fail. New
     loop-recursing passes should add a case here.
5. **Conditional bodies** — same story applies to `then_branch`/`else_branch`.
   Anything that fixes loops should fix conditionals on the way through.

## Suggested order of attack

1. **Infrastructure**: `Graph::for_each_subgraph` + per-pass `recurse_into_subgraphs()`
   hook + `PassManager::run` wiring. No behavior change (all passes default
   `recurse_into_subgraphs() = false`).
2. **Memory-lifecycle bugs**: Materialization → FreeInsertion → MemoryPlanning →
   InplaceOptimization. Unblocks the workspace example pattern cleanly.
3. **GPU/transfer bugs**: TransferInsertion → TransferElimination → GPUPlacement
   → StreamAssignment → GPUDiagnostics.
4. **Distributed bugs**: DistributionPlanning → InputSlicing → SUMMAExpansion →
   CommunicationInsertion → others.
5. **"Miss" sweep**: turn on the analysis/rewrite passes for body recursion
   once the bugs are gone.

## Status — what's done and what remains

**Done and verified — 23 of 29 passes resolved, all locally testable on this
machine:**

- Infrastructure: `Graph::for_each_subgraph`, `recurse_into_subgraphs()` hook,
  `PassManager::run` recursion, `Graph::collect_subtree_referenced_ptrs`.
- Hoist-aware (own walk, opt-out of auto-recursion): **Materialization**
  (+ init-kind propagation through capture), **FreeInsertion**, **GPUPlacement**
  (shared device budget).
- Recurse (opt-in): **ScaleAbsorption, ElementWiseFusion, PermuteFusion,
  DistributiveFactoring, GEMMBatching, StreamAssignment, TransferInsertion,
  TransferElimination, DeadNodeElimination** (subtree-live guard),
  **ConstantFolding** (materialization guard), **SymmetryPropagation**
  (single-writer + no-child-write guard), **ContractionPlanning**.
- Aggregate across the tree (opt-out, aggregate in `run()`): **MemoryPlanning,
  GPUDiagnostics, ChainParenthesization, InplaceOptimization**.
- Loop-aware by design but **had a soundness gap, now fixed**:
  **LoopInvariantHoisting** (added a single-writer-output guard — see below).
- Resolved as opt-out for soundness (see Reclassified section, behavior
  unchanged from flat, documented): **CSE, Reorder**.

That's 21 actively loop-aware (incl. IOPrefetch, below) + 2 deliberately-opt-out
= 23 resolved.

**IOPrefetch (done)** — hoist-aware: walks loop bodies (recursively) and
lifts a loop-invariant `DiskRead` out to before the outermost enclosing loop,
so the file is read once instead of every iteration. Eligibility: the read's
destination has exactly one *value* writer across the whole loop subtree (the
read itself — counted by tensor pointer, so a nested loop merely *reading* the
data doesn't block it) and is already materialized (eager) — IOPrefetch runs
before MaterializationPass, so a deferred destination keeps its per-iteration
read to avoid loading into unallocated storage. Also fixed a latent flat bug:
the within-graph prefetch now respects the read's *output*-tensor
anti-dependencies (a DiskRead has no inputs, so the old code always slid it to
position 0, even past a writer of its destination). Hoisted reads keep a valid
destination id (re-registered by pointer in the target graph) so deeply nested
reads lift level-by-level. Tests in `Executor.cpp` (hoist, don't-hoist-when-
overwritten, nested) run IOPrefetch *in isolation* — LIH (which runs earlier)
also hoists DiskReads, so the isolation keeps the IOPrefetch test focused on
the pass under test.

**LoopInvariantHoisting (fixed) — was *not* actually loop-correct.** The audit
originally marked LIH "correct by design," but it only validated a node's
*inputs* (`all_inputs_invariant`), never that its *output* was single-writer in
the loop. Two consequences, both real miscompiles:
  - An einsum `C = A·B` (invariant inputs) followed by an in-place
    `scale(0.9, C)` got the einsum hoisted, dropping the per-iteration reset so
    the scale compounded (`C, 0.9C, 0.81C, …` instead of `0.9·AB` each
    iteration). An *existing* test asserted this hoist happened — it was
    encoding the bug, since it only checked structure, not values.
  - A `DiskRead` (no inputs → trivially "invariant") got hoisted even when
    another body node overwrote its destination.
  Fix: only hoist a node whose every output has exactly one *value*-writer in
  the loop subtree (counted by tensor pointer, lifecycle nodes excluded — same
  helper as IOPrefetch). The existing test was corrected to a sound positive
  case; added negative + value-correctness tests (`Pass_LoopInvariantHoisting.cpp`
  and the Python mirror), and verified a hoisted node with a *deferred* output
  is still materialized correctly (Materialization, which runs after LIH,
  inserts the Materialize before the hoisted node — so LIH needs no
  materialized guard). The `self_modifying` guard is kept (belt-and-suspenders;
  the input check already catches in-place ops since their output is also an
  input).

**Remaining — 6 passes:**

1–6. **The distributed group (6 passes): DistributionPlanning, InputSlicing,
   SUMMAExpansion, CommunicationInsertion, CommunicationElimination,
   CommunicationScheduling.**
   - Why deferred: these are no-ops at a single MPI rank, so their loop behavior
     can't be exercised on this machine. Writing them blind is exactly how the
     CSE and Reorder soundness gaps slipped in (the SCF energy assertion caught
     those only because SCF is non-distributed). They should be done on a
     multi-rank MPI box, each with a `mpirun -np N` correctness check analogous
     to the SCF/MP2 energy guards — verify a distributed einsum chain inside a
     loop body gives the same result as the replicated (1-rank) computation.
   - Likely shape of the work (by analogy to what's done):
     - **DistributionPlanning** — must classify body einsum indices consistently
       with the parent's distribution decisions; recursion plus a consistency
       check that a tensor's distribution doesn't differ between parent and body.
     - **InputSlicing / SUMMAExpansion** — per-graph transforms like
       TransferInsertion; probably just `recurse_into_subgraphs() = true` once a
       multi-rank test confirms the body slicing/expansion is correct each
       iteration.
     - **CommunicationInsertion** — hoist-aware: an allreduce for a
       loop-invariant reduction could lift out of the loop (like
       TransferInsertion's H2D hoist); correctness-first version just inserts
       inside the body.
     - **CommunicationElimination** — needs care about loop-carried invalidation
       (an allreduce result from iteration *i* isn't valid in *i+1* if the tensor
       was rewritten), parallel to the SymmetryPropagation single-writer concern.
     - **CommunicationScheduling** — async overlap; per-graph, likely safe to
       recurse after a multi-rank check.

## Progress log

- **Infrastructure (done)** — `Graph::for_each_subgraph`, `OptimizerPass::recurse_into_subgraphs()`
  (default false), `PassManager::run` post-order recursion. Tests in
  `RecursionPlumbing.cpp`.
- **Materialization (done)** — walks descendants, hoists Materialize/Initialize
  to the parent before the outermost enclosing loop. Plus Step 2.5: init kind
  (`declare_zero_tensor` etc.) now rides through capture on the tensor itself
  (`PendingInit`), so hoisted tensors are zeroed too. Tests in `Workspace.cpp`.
- **FreeInsertion (done)** — hoists a body intermediate's single Free to the
  parent after the outermost enclosing loop; workspace tensors stay alive.
  Fixed a latent idempotency double-free. Tests in `Pass_FreeInsertion.cpp`.
- **Safe local-rewrite passes (done)** — opted into recursion via
  `recurse_into_subgraphs() = true`: **ScaleAbsorption, ElementWiseFusion,
  PermuteFusion, DistributiveFactoring, GEMMBatching, StreamAssignment**.
  Each is a per-graph local rewrite (or per-node stream assignment) that's
  correct on a flat sub-graph. Policy contract + behavioral DNE-in-loop test
  in `RecursionPlumbing.cpp`.

  **Two passes were initially flipped on but reverted after the SCF example
  caught a divergence** (the examples now assert energies — see below):
  - **CSE** — unsound on mutable-tensor reuse. It merges two nodes that
    compute the same value into *different* output tensors and redirects the
    duplicate's readers to the original's output; wrong when that output is
    later written independently. The SCF body's `axpby(1,H,0,F)` and
    `axpby(1,H,0,sum_HF)` (F and sum_HF then diverge) made the run diverge.
    Needs a write-once precondition on the merged output before it can
    recurse. Reverted to opt-out.
  - **Reorder** — its memory-aware topological sort doesn't fully preserve
    the WAR/WAW anti-dependencies created by a loop body's in-place reuse;
    recursing reordered the SCF body's snapshot+recompute sequence and broke
    multi-iteration convergence (HeH+). Reverted to opt-out.

  Lesson: "local rewrite" isn't automatically loop-safe — passes that assume
  functional/SSA-ish dataflow can be unsound on the heavy in-place tensor
  reuse a loop body exhibits. The remaining six were re-verified as a *full
  combination* against both SCF systems, not one-pass-at-a-time.

### Reclassified / deferred (with reasons)

- **DeadNodeElimination (done)** — the groundwork is `Graph::collect_subtree_referenced_ptrs`,
  which gathers the underlying tensor pointers referenced anywhere in a
  graph's descendant sub-graphs (ids differ across graphs; the pointer is the
  stable identity). DNE now treats a tensor whose pointer is in that set as
  live, so a producer feeding only a nested loop is never eliminated. With
  that guard in place, DNE opts into recursion. Tests: dead-intermediate-in-body
  and the keeps-producer-feeding-nested-loop case in `Pass_DeadNodeElimination.cpp`.
- **ConstantFolding (done)** — loop-*carried* correctness was already fine (a
  loop-carried tensor is written somewhere in the body, so it's in
  `written_tensors` and never treated as constant). The blocker was that CF
  *executes* foldable nodes at pass time and bakes the result, running
  **before** Materialization — so recursing could execute a body node against
  an unallocated workspace shell. Fixed with a **materialization guard**: CF
  only folds a node when all its input/output tensors are
  `AllocState::Materialized` at pass time; deferred body tensors are skipped.
  (Moving the pass later wouldn't help — Materialize nodes only allocate at
  graph-execute time, not pass time.) With the guard, CF opts into recursion.
  Note: CF's constant criterion (`is_intermediate && never-written`) is rarely
  met in practice — graph intermediates are written by their Alloc node — so
  CF is close to a no-op today; the fix is about making recursion *safe*, not
  about new folding. Test: `Pass_ConstantFolding.cpp` (safe no-op on a
  deferred-tensor loop body + correct end-to-end result).
- **SymmetryPropagation (done)** — the loop concern was a *stale tag*: it
  infers a symmetry guarantee on a tensor from its producing op, but a later
  overwrite (another body node, or a nested-loop write a Loop node doesn't
  expose) could invalidate it. The tag is advisory today (no
  correctness-critical consumer in TensorAlgebra/LinearAlgebra), but to make
  recursion unconditionally sound the pass now only tags a tensor that has
  **exactly one value-writer in this graph** (lifecycle Alloc/Free/
  Materialize/Initialize nodes don't count) **and isn't referenced by a child
  sub-graph** (reusing `collect_subtree_referenced_ptrs`). This also fixes the
  pre-existing flat multi-writer staleness. With both guards it opts into
  recursion. The pass only reads op structure (no execution, no node
  changes), so there's no materialization concern. Tests: infers on a
  body self-contraction; refuses a multi-writer body tensor; refuses a tensor
  written by a nested loop (`Pass_SymmetryPropagation.cpp`).
- **CommunicationElimination** — same "needs care about loop-carried
  overwrites" shape, and it's a distributed pass (only meaningfully testable
  under multi-rank MPI). Deferred.
- **Analysis-aggregation passes (done: MemoryPlanning, ChainParenthesization,
  GPUDiagnostics, InplaceOptimization)** — each now aggregates over the whole
  graph tree inside `run()` (a per-graph helper + a `for_each_subgraph` walk),
  rather than being re-run per sub-graph (which would clobber the counters
  with the last subgraph's numbers). `recurse_into_subgraphs()` stays false.
  Sums for counts/totals; max for peaks (a precise cross-loop peak would need
  loop-carried liveness — max is a defensible lower bound for reporting).
  Tests: loop / nested-loop cases in `Pass_MemoryPlanning.cpp`,
  `Pass_ChainParenthesization.cpp`, `Pass_InplaceOptimization.cpp`, and a
  loop case in `GPUPasses.cpp`.
- **ContractionPlanning (done)** — the earlier "unmaterialized intermediates"
  worry was **wrong on inspection**: the intermediates it creates via
  `create_tensor_dynamic` are *eager* (→ `create_zero_tensor`, allocated at
  pass time; the Alloc node is a no-op marker), so they don't depend on the
  Materialization pass that runs earlier. Chain restructuring to the optimal
  parenthesization is associativity-equivalent and per-graph, so it's safe to
  recurse. Verified the real case: the SCF body's X·F·X chain gets
  restructured and both H2 and HeH+ energies are unchanged. Tests: a
  100×1·1×100·100×1 chain restructured inside a loop body matches the eager
  reference (`LoopOptimizationCorrectness.cpp`). Opts into recursion.
- **GPU group (done: GPUPlacement, TransferInsertion, TransferElimination)** —
  verified locally (this machine has MPS; placement is active).
  - **GPUPlacement**: walks the whole graph tree collecting candidates, then
    places within a *single shared* device-memory budget — parent and body
    draw from the same budget, so a loop body's hot GEMM gets placed without
    over-subscribing device memory. `recurse_into_subgraphs()` stays false
    (own walk, for the shared budget).
  - **TransferInsertion / TransferElimination**: `recurse_into_subgraphs() =
    true`. Each is a clean per-graph transform — run on a body, TransferInsertion
    makes the body self-contained (H2D before each GPU op, D2H after);
    TransferElimination cleans body-internal redundancy (its `is_loop_tensor`
    pin already protects tensors a nested loop needs).
  - **Known limitation / future optimization**: TransferInsertion re-transfers
    loop-invariant inputs every iteration. Hoisting those H2D before the loop
    (and the matching D2H after) is the remaining refinement — correctness is
    fine today, it's a performance opportunity.
  - Tests: loop placement, shared-budget-across-loop, and body-transfer
    insertion in `GPUPasses.cpp`.
- **Distributed hoisting** (DistributionPlanning, InputSlicing, SUMMAExpansion,
  CommunicationInsertion, CommunicationScheduling) — only meaningfully testable
  under real multi-rank MPI; at 1 rank these are no-ops. Not yet done.

## Differential fuzz harness (2026-05-20)

`tests/unit/test_fuzz_differential_python.py` generates random programs over a
pool of square matrices and length-N vectors (scale / axpy / axpby / gemm /
einsum / symm_gemm / gemv / ger, plus loop and conditional control flow) and
runs each three ways — a numpy oracle, the raw graph (no passes), and the
optimized graph (`default_pass_manager`) — asserting all three agree.
RAW≠oracle means an executor bug; OPTIMIZED≠oracle means a pass miscompiled.
Failures print the seed and the offending program; a greedy reducer (in the
session notes) shrinks a failure to a minimal program.

It found and drove fixes for **fourteen** real soundness bugs that the
hand-written loop tests had missed — in the read-modify-write / control-flow /
view-alias interactions that the per-pass loop work above had only partially
covered:

1. **Reorder dropped WAR edges** — its dependency graph encoded RAW and WAW but
   not write-after-read, so the memory-aware sort could float a write of a
   tensor ahead of an earlier read of the old value. In-place reuse at the top
   level broke. Fixed by tracking readers-since-last-write and adding
   reader→writer edges (`Passes/Reorder.cpp`).
2. **CSE merged mutable buffers** — value-numbering treated tensor outputs as
   immutable SSA values, merging two ops that write *different* destinations and
   redirecting readers; a later in-place write then corrupted the survivors.
   Also, axpby/axpy/scale coefficients aren't in `op_data`, so CSE couldn't tell
   two of them apart. Fixed with three guards: only pure-overwrite producers are
   eligible (`cse_eligible`: einsum/permute/gemm with zero destination
   prefactor), shared inputs must be stable between the two nodes, and both
   output buffers must be single-writer (`Passes/CSE.cpp`).
3. **LIH ignored writes inside nested subgraphs** — a loop body's
   `body_writes` set only collected *direct* node outputs, so a tensor mutated
   only inside a conditional branch (or inner loop) looked invariant and its
   consumer was hoisted out. Fixed by also consulting the by-pointer
   subtree-writer count for input invariance (`Passes/LoopInvariantHoisting.cpp`).
4. **Accumulating ops didn't expose their read destination** — an einsum/gemm
   with a nonzero destination prefactor (and gemv with `beta != 0`, and ger
   always) reads its own output, but the node listed only the source operands as
   inputs and LIH's self-modify check only recognized scale/axpy/axpby/element-
   transform. Two complementary fixes in `Operations.hpp`: **`cg::gemm`/`gemv`
   list their destination as an input when `beta != 0`, and `cg::ger` always
   does** (it always accumulates) — so the read is visible to every pass and the
   scheduler. LIH also gained `reads_its_output` (covers nonzero-prefactor
   einsum/permute/batched-gemm) for the descriptor-carrying path. Pure
   overwrites (`beta == 0`) keep the source-only input form and stay hoistable.
5. **Control-flow nodes had no scheduling edges** — Loop and Conditional nodes
   are created with empty input/output lists (bodies are captured afterward), so
   `topological_sort` and Reorder could float them past producers/consumers of
   the tensors their bodies touch. The executor itself misordered (a RAW
   failure). Fixed with `Graph::effective_io(node)`, which augments a
   control-flow node's I/O with the buffers its subtree reads/writes (mapped
   back to this graph's TensorIds); `topological_sort` and Reorder now schedule
   using effective I/O. The node's own lists stay empty so structural passes
   (e.g. DeadNodeElimination) are unaffected.

The mixed-shape / transpose / bigger-seed extension surfaced three more:

6. **LIH minted a fresh TensorId for a hoisted output** — `remap_or_register`
   registered a *new* parent id for a hoisted node's output even when the parent
   already had an id for that buffer, hiding the WAW/aliasing relation with
   parent nodes writing the same tensor (the scheduler keys on TensorId). LIH
   alone kept the order, but Reorder in the full pipeline then swapped them and
   the wrong write won. Fixed by reusing the existing parent id for a buffer
   already known there (`Passes/LoopInvariantHoisting.cpp`).
7. **LIH hoisted a producer read by an *earlier* body node** — when a body node
   reads tensor T *before* the producer of T runs (e.g. `ger` reads `v`, then
   `gemv` overwrites `v`), T is loop-carried *through* the producer; hoisting the
   producer changes what the earlier reader sees. Added a guard: don't hoist if
   any earlier body node reads the candidate's output.
8. **`effective_io` dropped subgraph-only buffers** — a buffer used *only* inside
   sub-graphs (e.g. a tensor a loop body and a conditional branch both touch, but
   no direct parent node references) has no parent TensorId, so it was dropped
   from effective I/O and two control-flow nodes touching it got no edge between
   them. Order then fell to tie-break and the wrong one won (a RAW executor
   failure). Fixed by registering such buffers in the parent on first sight so
   they get a stable shared id (`effective_io` is now non-const).

The batched-einsum (rank-3) + **view/slice** extension surfaced six more — a
whole cluster around *alias resolution* (a write through a view of T must be
seen as a write to T), plus a latent TensorId bug the views happened to trip:

9.  **TensorId 0 collided with the "no alias" sentinel** — ids were allocated
    from 0, but `TensorHandle::aliases` defaults to 0 and the codebase tests
    `aliases == 0` for "not a view". So a view of the *first-registered* tensor
    (id 0) had `aliases == 0` and silently failed to resolve to its parent — the
    executor read stale data (a RAW failure). Fixed by starting `_next_tensor_id`
    at 1, reserving 0 as the sentinel (`Graph.hpp`).
10. **Reorder didn't resolve view aliases** — it keyed its dependency graph on
    raw TensorIds, so a write through a view of T looked independent of a read of
    T and got reordered past it. Fixed by resolving through `Graph::resolve_alias`.
11. **DeadNodeElimination didn't resolve view aliases** — a view-write's output
    is the (unread) view tid, so DNE deleted it as dead even though it updates a
    live parent. Fixed by resolving outputs/inputs to the owner before the
    intermediate/consumed/subtree checks.
12. **LIH didn't resolve view aliases** — `count_subtree_writers_by_ptr` (and the
    invariance / single-writer checks) recorded the view object's pointer, not
    the owner's, so a consumer of T was hoisted past a view-write that mutates T
    each iteration. Fixed by resolving to the owner pointer.
13. **LIH's earlier-reader guard missed control-flow readers** — bug #7's guard
    scanned `earlier.inputs`, but an earlier *nested loop/conditional* reads its
    output inside its subtree (empty raw inputs). A loop-carried tensor written
    by a node after an inner loop that reads it got wrongly hoisted. Fixed by
    using `effective_io` (subtree-aware reads) in the guard.
14. **`effective_io` didn't resolve view aliases** — a view-write inside a loop
    body was attributed to the view pointer, so the loop wasn't seen as writing
    the owner and a later reader of the owner was reordered before the loop.
    Fixed by resolving aliases in the subtree collection.

The common lesson: **every analysis that reasons about *which buffer* a node
touches must resolve view aliases** (`Graph::resolve_alias`). `topological_sort`
already did; Reorder, DNE, LIH, and `effective_io` did not until views fuzzed
them. (And a sentinel must never collide with a valid id.)

A later escalation (element_transform + view-as-gemm-operand ops; bigger/deeper
programs; and four new execution modes — re-execution/replay, random pass-pipeline
permutations, double-optimize idempotency, and random-pipeline+replay) found no
*new* ComputeGraph soundness bugs (strong evidence the alias/scheduling machinery
is now robust), but the longer run exposed a 15th, unrelated bug:

15. **Profiler shutdown SIGSEGV** — `einsums::profile::Server::tick(this=0x0)` on
    the background consumer thread. The `Profiler` Meyers singleton destroys
    `_server` before the consumer thread (members destruct in reverse order),
    while the consumer's periodic tick callback still runs `_server->tick()`. The
    `Py_AtExit→finalize()→Profiler::shutdown()` path exists but is guarded by
    `is_running()` and races with C++ static destruction. Fixed by adding an
    explicit `~Profiler()` that stops the consumer thread (and server) before the
    members destruct (`Profile/.../Profile.hpp`). Lesson: a long-running Python
    test process is itself a cheap shutdown-race fuzzer.

A complex128 suite (~1050 cases, flat/control-flow/deep/random-pipeline) found no
further bugs — confirming complex tensor data flows correctly through every op and
pass. Every cg op uses *no* conjugation (plain transpose, `geru` not `gerc`, einsum
`conj=false`, symm `B^T A B`), so the identical numpy oracle is correct for complex;
scalars stay real (einsum prefactor bindings are real-only).

A **deferred-tensor mode** (half the matrices declared via `Workspace.declare_zero_tensor`,
run both with explicit `materialize_all()` and via the default manager's
MaterializationPass) found a 16th bug: MaterializationPass materialized a deferred
buffer **twice** when it was used both inside a loop body and in the parent (one
buffer, two graph-local TensorIds), and the second `init_zero` wiped a loop's
accumulation before a later read. Fixed by deduping materialization requests by
underlying `tensor_ptr` — materialize once, at the earliest use.

The deferred mode was then extended to defer all three pools (matrices, vectors,
rank-3) and to a **replay** variant (execute the optimized graph twice — deferred
tensors re-zero each execute via their Initialize node, eager tensors carry over);
no further bugs surfaced, confirming the materialization / Free / Initialize
machinery is replay-safe for deferred tensors.

The core modes (flat/control-flow/deep) are then parametrized over all four
dtypes (`@pytest.mark.parametrize("dtype", ALL_DTYPES)`): float32, float64,
complex64, complex128 — the first coverage of single precision through the
passes. Two care points keep it false-positive-free: the oracle computes in the
trial precision (`interp_np` casts per op, else float32 promotes to float64), and
differential tolerances are looser than `tolerance_for`'s same-computation values
(1e-3 for single, 1e-5 for double) with a tighter magnitude cap for single
precision. No dtype bugs found.

A `cg.Pipeline` multi-stage mode was also added (a program is a sequence of stage
sub-programs sharing one tensor pool — a later stage reads what an earlier wrote;
oracle = interp over the concatenation; run raw and with the default manager
applied per stage, over all dtypes). No pipeline bugs found.

After the fixes: ~4900 fuzz cases (all four dtypes; eager and deferred allocation;
single graphs and multi-stage pipelines; single-execute and replay; degenerate
overflow/NaN programs auto-skipped) + the full 83-test ComputeGraph suite
(C++ and Python) pass.
The harness is registered as
`Tests.Unit.Modules.ComputeGraph.FuzzDifferentialPython`. Conditionals became
fuzzable once `add_conditional` was bound to Python (return type changed from
`std::pair<Graph&,Graph&>` to `std::tuple<Graph&,Graph&>`). `cg::read`/DiskRead
fuzzing is still deferred.

## Further stress-testing ideas (2026-05-21)

After ~4900 differential cases across all the modes/dtypes above, more *random
numerical* programs hit diminishing returns (several rounds found nothing). The
differential oracle only checks whether the numbers match — it is blind to memory
safety, concurrency, and to passes/executors the fuzzer never actually triggers.
The high-value next steps change the *dimension* under test, ranked:

1. **Sanitizers (ASan/UBSan) over the corpus.** Differential testing verifies
   values, not memory. The passes/executor do heavy pointer work (moving nodes,
   view aliasing, by-`tensor_ptr` dedup, deferred materialize/free, `adopt`
   deleters); a use-after-free / OOB can produce correct numbers by luck and stay
   invisible to the oracle (the profiler shutdown SIGSEGV was that class, found
   only by chance). ASan finds them deterministically. **Build gotcha:**
   `EINSUMS_WITH_SANITIZERS` is a *STRING passed straight to* `-fsanitize=` (valid:
   `address;leak;memory;thread;undefined`) — use `-DEINSUMS_WITH_SANITIZERS=address`,
   **not** `=ON` (which would emit `-fsanitize=ON`). The flags ride
   `einsums_public_flags`, so libEinsums + C++ tests are instrumented (ASan in the
   test `main` → clean); driving the Python harness under ASan needs the clang
   ASan runtime `DYLD_INSERT_LIBRARIES`'d into the conda python.
2. **Cross-executor differential.** The fuzzer only uses sequential `execute()`.
   `SequentialExecutor` / `OpenMPExecutor` / `DataflowExecutor` are all bound and
   `execute(executor)` works; run each program through all three and demand they
   agree — tests the parallel/dataflow dependency scheduling (separate code) for
   races / missing edges.
3. **Pass-firing coverage instrumentation.** We don't know which passes actually
   fire (`modified=True`) on fuzz inputs; a never-firing pass is untested even
   though it "passes." Measure per-pass fire-rate and bias generation to trigger
   the cold ones (GEMMBatching needs identical sibling einsums; Distributive-
   Factoring / ChainParenthesization need specific chain structure).
4. **Bit-exact determinism check.** Execute the optimized graph twice and require
   *bit-identical* output (stronger than oracle-close) — catches nondeterministic
   scheduling (e.g. unordered-pointer iteration in `effective_io`).
5. **Data-dependent conditionals with safe margins.** Conditionals are coin-flips
   today; the predicate-reads-a-tensor path is only in the SCF/MP2 examples. Use
   `mean(T) > threshold` with a guaranteed margin (computed from the oracle) so
   oracle and graph never disagree on the branch.
6. **Negative / error-path fuzzing.** Malformed einsum specs and mismatched shapes
   should produce graceful errors, not crashes — an untested robustness surface.

## AddressSanitizer recipe (validated 2026-05-21)

ASan was run over both the C++ unit suite and the Python differential fuzzer. It
found two real memory bugs the differential oracle is blind to — both fixed:
- **#17** profiler ring-buffer heap-use-after-free (transient DataflowExecutor
  worker threads vs. the Consumer's raw pointer; fixed via shared ownership).
- **#18** the tensor destruction "canary" read freed memory (UB *and* unreliable);
  replaced with a `weak_ptr` liveness token.

After those, the **full 150-test C++ unit suite** and the **full ~4900-case Python
fuzzer** are both ASan-clean.

### Build (separate dir; no MLIR — it's removed)

`EINSUMS_WITH_SANITIZERS` is a STRING passed straight to `-fsanitize=`, so use
`=address`, **not** `=ON`. The Python bindings need BOTH `EINSUMS_BUILD_PYTHON=ON`
*and* `EINSUMS_PYBIND_AUTOGEN=ON` (the latter builds the codegen tool and creates
the `PyEinsums` target). pybind11's FindPython may grab the *base* conda python —
pin the env interpreter explicitly, or the `_core.so` is built for the wrong ABI.

```
CONDA=/Users/jturney/miniconda3/envs/einsums-dev
cmake -S . -B build-asan -GNinja \
  -DCMAKE_PREFIX_PATH=$CONDA \
  -DCMAKE_C_COMPILER=$CONDA/bin/clang -DCMAKE_CXX_COMPILER=$CONDA/bin/clang++ \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DEINSUMS_WITH_SANITIZERS=address \
  -DEINSUMS_BUILD_PYTHON=ON -DEINSUMS_PYBIND_AUTOGEN=ON \
  -DPython_EXECUTABLE=$CONDA/bin/python3.14 \
  -DEINSUMS_WITH_TESTS=ON -DEINSUMS_WITH_TESTS_UNIT=ON \
  -DEINSUMS_WITH_TESTS_BENCHMARKS=OFF -DEINSUMS_WITH_TESTS_EXAMPLES=OFF \
  -DEINSUMS_WITH_TESTS_HEADERS=OFF -DEINSUMS_WITH_TESTS_REGRESSIONS=OFF \
  -DEINSUMS_WITH_TESTS_EXTERNAL_BUILD=OFF
cmake --build build-asan                       # C++ tests
cmake --build build-asan --target PyEinsums    # instrumented _core.so
```

### Run — C++ suite (ASan linked into the test `main`, the clean path)

```
ASAN_OPTIONS=detect_leaks=0:abort_on_error=0 \
  ctest --test-dir build-asan -L UNIT_ONLY
```

### Run — Python fuzzer (preload the runtime; conda python is not SIP-blocked)

The ASan-instrumented `_core.so` is loaded by a non-ASan python, so the ASan
runtime must be force-loaded first (else "AddressSanitizer ... loaded too late").
`detect_leaks=0` (LSan unsupported on arm64 macOS) and `detect_container_overflow=0`
(std::vector redzone annotations mismatch across the un-instrumented libpython /
numpy / BLAS boundary) are required; no suppressions file was needed beyond those.

```
ASANRT=$CONDA/lib/clang/22/lib/darwin/libclang_rt.asan_osx_dynamic.dylib
DYLD_INSERT_LIBRARIES=$ASANRT \
ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0:abort_on_error=1 \
PYTHONPATH=build-asan/lib \
  $CONDA/bin/python3.14 -m pytest \
  libs/Einsums/ComputeGraph/tests/unit/test_fuzz_differential_python.py
```
