//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Lower tiled operations into per-tile DENSE nodes.
 *
 * A tiled contraction captures as one opaque ``OpKind::Custom`` node executed
 * tile-by-tile by ``detail::tiled_runtime_einsum``. Nothing can optimize it: the
 * einsum-rewriting passes all filter on ``OpKind::Einsum``, and a whole-tiled node
 * could not satisfy them anyway, since GEMMBatching wants one ``lda`` and
 * MemoryPlanning's arena wants one buffer, neither of which a tile container has.
 *
 * This pass replaces that node with one ordinary dense einsum per contributing
 * tile combination, built through @ref Graph::make_einsum_node. Because the result
 * is ordinary dense nodes, every existing pass then applies unchanged: the tile
 * GEMMs are same-shape so GEMMBatching can batch them, CSE deduplicates repeated
 * tile work, MemoryPlanning can pack per-tile buffers (each tile IS a dense
 * ``RuntimeTensor``, so it has ``materialize_into``), and Reorder schedules them.
 *
 * Tiled ``scale`` and ``axpy`` lower the same way, into one dense ``OpKind::Scale``
 * or ``OpKind::Axpy`` per stored tile. They are trivially per-tile; the point is
 * that afterwards the tile buffers are visible to CSE and InplaceOptimization.
 *
 * @par Semantics it must reproduce exactly
 * Ground truth is ``detail::tiled_runtime_einsum``:
 * - **Alignment.** Every shared index letter must carry an identical tile
 *   partition on each operand. The runtime throws; this pass DECLINES instead,
 *   leaving the opaque node, because the fallback is correct and merely
 *   unoptimized.
 * - **Sparsity.** A tile pair contributes only when both operand tiles are
 *   present, so absent tiles are structural zeros and emit nothing. Output tile
 *   ``C(c)`` exists iff some contributing combination reaches it.
 * - **The output prefactor applies exactly once.** The runtime pre-scales
 *   pre-existing output tiles by ``c_pf`` and then accumulates every contribution
 *   with ``beta = 1``. Expanded, the FIRST node writing a pre-existing tile
 *   carries ``c_pf`` and later ones carry 1; the first node writing a tile this
 *   pass CREATED carries 0, a pure overwrite, so correctness does not depend on
 *   the new tile being zero-initialized.
 * - **Pre-existing output tiles that receive NO contribution are still scaled.**
 *   The runtime scales them up front regardless. Emitting nodes only for
 *   contributing pairs would leave them untouched, which is a silent numerical
 *   difference, so this pass emits a scale (or a zero when ``c_pf == 0``) for
 *   them.
 * - **A tiled axpy visits the tiles stored in X**, creating the matching Y tile
 *   when absent (it starts zeroed, so the accumulation is still right). A tile
 *   absent from X contributes nothing and a Y tile with no X counterpart is left
 *   alone. A tile-grid mismatch makes the runtime throw, so this pass declines
 *   and lets the surviving opaque node throw.
 *
 * @par Tile sparsity
 * Structural sparsity is free: an absent operand tile is a rigorous zero, so no
 * node is emitted for it and an output tile exists only if some contribution
 * reaches it. What that misses is a tile that is STORED but zero, which still
 * gets a full GEMM. @p zero_tile_tolerance screens those, and screening is done
 * here rather than as a pass over the expanded graph on purpose: a screened tile
 * is simply treated as absent, so the prefactor and output-existence rules above
 * apply unchanged instead of being re-derived against already-emitted nodes.
 *
 * Sparsity then propagates on its own. An output tile whose every contribution
 * screened out is never created, so it is absent to the next contraction, which
 * screens further work without being told anything.
 *
 * Two origins, per the chemistry roadmap, and the tolerance covers both: symmetry
 * blocking is EXACT (a block is zero unless the irrep product contains the
 * totally symmetric representation) and wants tolerance 0, while integral
 * screening and local-correlation domains are approximate and want a real
 * threshold.
 *
 * Only operands that NOTHING in the graph writes are screened. A produced
 * tensor's tiles hold whatever was in them before execution, so inspecting them
 * would screen on values the graph has not computed yet. Non-materialized tiles
 * are likewise left alone, since planning must not allocate.
 *
 * @par Emission order is what makes the tile GEMMs batchable
 * Contracted indices are enumerated SLOWEST and output indices fastest, so every
 * tile GEMM at the same accumulation step is emitted as one contiguous run.
 * GEMMBatching only batches a group whose span contains no outside node touching
 * the same buffers; letting the contracted index vary fastest drops each output
 * tile's later accumulations in between the first writes, which disqualifies every
 * group and leaves the expansion paying node-count cost for nothing. Per output
 * tile the contracted steps stay ascending, so each tile accumulates in the order
 * the runtime uses and the result is bit-identical -- only independent tiles move
 * relative to one another.
 *
 * @par Predicted tile sets
 * A tiled tensor produced inside the graph has none of its tiles yet at pass
 * time, so its sparsity cannot be read off the object -- and reading the empty
 * object would make a consumer expand into nothing. Planning therefore walks the
 * nodes in execution order carrying a predicted tile set per tensor, seeded from
 * the stored tiles and extended by each producer it plans. That is what lets a
 * contraction feed another, or feed a scale, and still expand.
 *
 * The prediction is also what decides whether the first write to an output tile
 * carries ``c_pf`` or overwrites, so it must be the set as of THIS point in the
 * program, not the final one.
 *
 * It is computed before the stranding fixpoint has settled, which is sound: a
 * producer that ends up rejected stays in the graph still naming its output, so
 * that tensor is stranded and every consumer whose prediction depended on it is
 * rejected too.
 *
 * @par Whole-tensor references must not be stranded
 * Expanding a node replaces its whole-tensor reads and writes with per-tile ones,
 * so the whole-tensor TensorId loses every reference that node owned. A node left
 * behind that still names that id would have its dependency edge silently
 * dropped, and with no writer a reader can be scheduled before the tiles are
 * filled. A candidate therefore expands only when EVERY node touching its tiled
 * tensors also expands; since rejecting one candidate can strand another, this
 * iterates to a fixpoint. Deciding this before any tile is created matters,
 * because minting a per-tile id creates the tile, and a spurious tile changes how
 * the runtime applies ``c_pf`` on the path that ends up not expanding.
 *
 * @par Densifying small tiles
 * Expanding per tile only pays when a tile contraction is worth a dispatch. It
 * often is not. A symmetry-blocked CCSD residual on a small model expands to
 * nearly 4000 einsum nodes carrying 1.3 MFLOP between them - a mean of 330 flops
 * per node, which is less work than the call costs. Blocking saves real
 * arithmetic there (roughly 10x, by never touching symmetry-forbidden blocks) and
 * then spends the whole saving dispatching it.
 *
 * So below @p min_tile_flops of arithmetic per tile contraction, the pass lowers
 * the contraction differently: gather each tiled operand into a dense buffer, run
 * ONE dense einsum, and scatter the result back into exactly the output tiles the
 * per-tile path would have written. Absent tiles gather as zeros and forbidden
 * output tiles are still never created, so the answer is the same up to floating
 * point summation order. Measured on the particle-particle ladder of that model,
 * this replaces 256 nodes taking 0.595 ms with 3 nodes taking 0.214 ms - faster
 * despite doing strictly more arithmetic, because the dense form reaches
 * ~61 GFLOP/s and 256 tiny contractions reach ~2.
 *
 * The trade is arithmetic for throughput, so it is gated twice: on per-tile work
 * (@p min_tile_flops), and on how much extra arithmetic densifying would cost
 * (@p max_densify_inflation). A genuinely sparse tiled tensor fails the second
 * gate and keeps its per-tile lowering, which is the right answer - there the
 * blocking is skipping work worth skipping.
 *
 * @warning **Off by default (@p min_tile_flops is 0), because those two gates are
 * not sufficient to decide it.** Measured on the same CCSD residual at two model
 * sizes, replay with the profiler disabled:
 *
 * | inflation cap | 26 spin-orbitals | 50 spin-orbitals |
 * |---------------|------------------|------------------|
 * | 16            | 11.0 ms (from 19.0) | 103.1 ms (from 71.5) |
 * | 4             | never fires      | never fires      |
 *
 * Those contractions inflate by between 4x and 16x, so the setting that wins 1.8x
 * on the small model is the same one that loses 1.4x on the larger one, and no
 * threshold on the flop RATIO separates them. What actually differs is achievable
 * throughput: the small model's tiles run at ~2 GFLOP/s where dense runs at ~60,
 * and the larger model's tiles are already fast enough that buying throughput with
 * extra arithmetic is a bad trade. Deciding this needs an estimate of TIME on each
 * side - tile-shape-dependent rate against dense rate, plus the gather/scatter
 * traffic - which is what @ref CostModel::estimate_gemm_time_us exists to provide.
 * Wiring that in is the work that would make this safe to default on. Until then
 * it is opt-in, and worth opting into only where tiles are measurably tiny.
 *
 * @par Limits
 * - Runs FIRST in @ref PassManager::populate_default: it is a lowering step, so
 *   every pass below should see the per-tile form rather than the opaque node.
 * - **Densification ignores tile screening.** @p zero_tile_tolerance prunes
 *   near-zero operand tiles from the per-tile enumeration; the densified path
 *   gathers them anyway. It contributes their (near-zero) value rather than
 *   dropping it, so the densified answer is the more accurate of the two.
 * - **Node budget.** Expansion produces up to (tiles of A) x (tiles of B) nodes,
 *   and per-node graph bookkeeping is on the order of microseconds, so a large
 *   grid can cost more in overhead than the contraction saves. The pass declines
 *   above @p max_nodes rather than guessing. A gate cannot be subtly wrong the way
 *   a cost heuristic can.
 * - **Sparsity is decided at pass time.** For an operand written earlier in the
 *   same graph that means a prediction (see above), which holds only while every
 *   writer is one this pass understands.
 * - Creating the predicted output tiles is a side effect of ``apply()`` on user
 *   data, earlier than the execute-time infer-and-create it replaces.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("tiled");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", &C_tiled, A_tiled, B_tiled);  // one Custom node
 * }
 * graph.optimize();   // now one dense Einsum per contributing tile pair,
 *                     // and GEMMBatching has collapsed them into gemm_batch
 * @endcode
 * Add it explicitly only to run it apart from the default pipeline:
 * @code
 * cg::PassManager pm;
 * pm.add(std::make_shared<cg::passes::TiledExpansion>());
 * graph.apply(pm);
 * @endcode
 */
class EINSUMS_EXPORT TiledExpansion : public OptimizerPass {
  public:
    /// @param max_nodes Decline to expand when the projected node count exceeds this.
    /// @param zero_tile_tolerance Screen a contraction's operand tiles out when
    ///        their Frobenius norm is ``<=`` this. Negative, the default, disables
    ///        screening: no tile is inspected and the emitted node set is
    ///        unchanged. Zero prunes only exactly-zero tiles. A positive value is
    ///        a numerical approximation and an accuracy knob.
    /// @param min_tile_flops Below this much arithmetic per emitted tile
    ///        contraction, densify instead of expanding per tile (see the class
    ///        documentation). Zero disables densification entirely, restoring the
    ///        always-expand behaviour.
    /// @param max_densify_inflation Refuse to densify when doing so would multiply
    ///        the arithmetic by more than this. Guards the sparse case, where the
    ///        tiled form is skipping most of the work and is right to.
    explicit TiledExpansion(size_t max_nodes = 4096, double zero_tile_tolerance = -1.0, double min_tile_flops = 0.0,
                            double max_densify_inflation = 16.0);

    [[nodiscard]] std::string name() const override { return "TiledExpansion"; }
    bool                      run(Graph &graph) override;
    void                      reset_stats() override;

    /// Safe per sub-graph: a tiled einsum is expanded within the single graph it
    /// is handed, and the nodes it emits stay in that graph.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Tiled nodes replaced by per-tile nodes.
    [[nodiscard]] size_t num_expanded() const { return _num_expanded; }
    /// Dense per-tile nodes emitted, contractions and elementwise ops together.
    [[nodiscard]] size_t num_tile_nodes() const { return _num_tile_nodes; }
    /// Candidates left alone: misaligned partitions, produced operands, over
    /// budget, or sharing a tiled tensor with a node that cannot expand.
    [[nodiscard]] size_t num_declined() const { return _num_declined; }
    /// Tile contractions not emitted because an operand tile screened as zero.
    [[nodiscard]] size_t num_screened() const { return _num_screened; }
    /// Contractions lowered to gather + one dense einsum + scatter rather than to
    /// per-tile nodes, because their tiles were too small to be worth dispatching.
    [[nodiscard]] size_t num_densified() const { return _num_densified; }

  private:
    size_t _max_nodes;
    double _zero_tolerance;
    double _min_tile_flops;
    double _max_inflation;
    size_t _num_expanded{0};
    size_t _num_tile_nodes{0};
    size_t _num_declined{0};
    size_t _num_screened{0};
    size_t _num_densified{0};
};

} // namespace einsums::compute_graph::passes
