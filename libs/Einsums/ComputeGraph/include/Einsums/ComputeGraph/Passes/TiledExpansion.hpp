//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/// When to lower a tiled contraction by densifying rather than per tile.
enum class Densify : std::uint8_t {
    Never,  ///< Always emit one node per contributing tile combination.
    Auto,   ///< Let the @ref CostModel decide, per contraction. The default.
    Always, ///< Densify whenever the lowering is structurally possible. For tests.
};

/// When to collapse a tiled elementwise op into one node over all its tiles
/// instead of one node per tile.
enum class FuseTiles : std::uint8_t {
    Never,  ///< Always emit one dense node per tile.
    Auto,   ///< Fuse when the tiles cost more to dispatch than to touch. The default.
    Always, ///< Fuse every group of two or more tiles. For tests.
};

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
 * or ``OpKind::Axpby`` per stored tile. They are trivially per-tile; the point is
 * that afterwards the tile buffers are visible to CSE and InplaceOptimization.
 * When the tiles are too small for that to pay, they instead collapse into a
 * single ``OpKind::TileElementwise`` node covering every tile; see @ref FuseTiles.
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
 * So when a tile contraction is not worth its dispatch, the pass lowers
 * the contraction differently: gather each tiled operand into a dense buffer, run
 * ONE dense einsum, and scatter the result back into exactly the output tiles the
 * per-tile path would have written. Absent tiles gather as zeros and forbidden
 * output tiles are still never created, so the answer is the same up to floating
 * point summation order. Measured on the particle-particle ladder of that model,
 * this replaces 256 nodes taking 0.595 ms with 3 nodes taking 0.214 ms - faster
 * despite doing strictly more arithmetic, because the dense form reaches
 * ~61 GFLOP/s and 256 tiny contractions reach ~2.
 *
 * The trade is arithmetic for throughput, so which lowering wins is a question
 * about TIME, not about either quantity alone. @ref Densify::Auto asks the shared
 * @ref CostModel: sum @ref CostModel::estimate_total_gemm_time_us over the tile
 * contractions that would be emitted (that estimate already carries the per-call
 * launch and allocation overhead, which is the whole point at this size), against
 * one dense contraction of the same indices plus the gather and scatter traffic at
 * memory bandwidth. Densify only when the second is smaller.
 *
 * @par Fusing small elementwise tiles
 * The elementwise ops have the same node-count problem and none of the tension.
 * A densified CCSD residual is 82% tiled ``scale`` and ``axpy`` nodes -- 1152 of
 * 1404 -- each scaling or accumulating a handful of kilobytes, which is less work
 * than dispatching it. Unlike a contraction there is nothing to trade: running the
 * same tiles from inside one node does exactly the same arithmetic and touches
 * exactly the same memory, so fusing is never slower.
 *
 * What per-tile nodes buy is optimizer exposure -- CSE can drop a repeated tile
 * scale, InplaceOptimization can see the buffers -- and no cost model can price
 * that. So the gate asks the question it CAN answer: is a tile's own memory
 * traffic worth the dispatch it costs? @ref FuseTiles::Auto compares the tiles'
 * total traffic at memory bandwidth against @ref CostModel::node_overhead_us per
 * tile, and fuses when the dispatches dominate. Declining to expand at all is not
 * an alternative: the leftover opaque node still names the whole tiled tensor, so
 * it would strand every contraction sharing that tensor (see below) and cost far
 * more than it saves.
 *
 * The same gate covers the scales this pass emits for output tiles a contraction
 * or a divide never reaches, since those are elementwise and just as small.
 *
 * What it is worth: on that residual the graph falls from 1404 nodes to 209 and
 * replay from 10.1 to 9.9 ms. The node count is the larger prize. Timing the
 * nodes individually puts all 1152 elementwise ops at 0.17 ms of a 9.8 ms replay,
 * because the executor's own per-node cost is about 0.2 us -- well under the
 * ``kernel_launch_overhead_us + alloc_overhead_us`` the model charges. Fusing is
 * still right, since it buys those nodes back for nothing, but the replay time
 * this recovers is a few percent and not the fifth that the node count suggests.
 *
 * @par Gathering an operand once
 * A densified contraction copies each tiled operand into a dense buffer, and the
 * same operand is usually contracted several times running: the CCSD residual
 * gathers the singles amplitudes 33 times and the doubles eleven. Those copies
 * are identical while nothing has written the tiles between them, so the pass
 * keys each gather on the exact list of tile ids it covers and reuses the buffer
 * when that list is already in hand.
 *
 * Two things keep it honest. A tensor that has GAINED a tile since produces a
 * different list, so it misses and is gathered afresh rather than silently
 * contracting a copy that is missing a block. And an entry is dropped as soon as
 * any emitted node writes a tile it covers -- which catches every writer, because
 * a candidate whose tiled operands are touched by a node that does not expand has
 * already been rejected by the stranding fixpoint. Reuse is safe across
 * scheduling too: the dependency scan tracks write-after-read, so a later writer
 * of those tiles cannot be hoisted above the gather that reads them.
 *
 * It is worth more than the densification that creates it. On the residual the
 * gathers fall from 92 nodes to 23 and from 2.20 ms to 0.71 of a 10 ms replay.
 *
 * A flop-based gate was tried first and does not work. Measured on one CCSD
 * residual at two model sizes, replay with the profiler disabled: an inflation
 * cap of 16 takes the 26 spin-orbital model from 19.0 to 11.0 ms but the
 * 50 spin-orbital model from 71.5 to 103.1 ms, and a cap of 4 never fires at
 * either size.
 *
 * Those contractions inflate the arithmetic by between 4x and 16x either way, so
 * no threshold on the ratio separates the case that wants densifying from the one
 * that does not. What differs is achievable throughput: the small model's tiles
 * run at ~2 GFLOP/s where dense reaches ~60, while the larger model's tiles are
 * already fast enough that buying throughput with extra arithmetic is a bad trade.
 * Only a time comparison sees that, because only it knows the shape-dependent
 * rate.
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
    /// @param densify Whether small-tile contractions are lowered by densifying.
    ///        @ref Densify::Auto, the default, compares estimated time both ways
    ///        per contraction and picks the cheaper.
    /// @param fuse Whether small-tile elementwise ops collapse into one node.
    ///        @ref FuseTiles::Auto, the default, fuses a group whose tiles cost
    ///        more to dispatch than the memory traffic they do.
    explicit TiledExpansion(size_t max_nodes = 4096, double zero_tile_tolerance = -1.0, Densify densify = Densify::Auto,
                            FuseTiles fuse = FuseTiles::Auto);

    /// As above, with an explicit cost model rather than a detected one. The
    /// default pipeline uses this so every cost-model pass shares one profile.
    TiledExpansion(size_t max_nodes, double zero_tile_tolerance, Densify densify, FuseTiles fuse, CostModel cost_model);

    [[nodiscard]] std::string name() const override { return "TiledExpansion"; }

    /**
     * @copydoc OptimizerPass::phase
     *
     * Structural-resource: expanding a tiled operand into per-tile dense nodes
     * is a node-set change made for machine reasons (tile shape, sparsity,
     * whether densifying beats staying sparse on this cache hierarchy), so its
     * output is re-derived on load rather than saved.
     *
     * @par Recorded deviation from the phase ordering rule
     * The phase rule wants every algebraic pass to run before every resource
     * pass, and the default pipeline runs this one FIRST, ahead of all of them.
     * The reason is that a tiled op captures as one opaque ``OpKind::Custom``
     * node: until it is lowered, CSE, ContractionPlanning, GEMMBatching,
     * InplaceOptimization and MemoryPlanning cannot read it at all, so ordering
     * this pass by phase would not delay a decision, it would delete one. The
     * deviation is deliberate and documented rather than fixed by reordering,
     * and it is why ``PassManager::populate_default`` is a hand-ordered
     * sequence that the phase-filtered managers VIEW rather than a sequence
     * derived from the phases.
     */
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralResource; }
    bool                    run(Graph &graph) override;
    void                    reset_stats() override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;

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
    /// Groups of elementwise tile ops collapsed into a single node, for the same
    /// reason.
    [[nodiscard]] size_t num_fused() const { return _num_fused; }
    /// Gathers not emitted because a dense copy of exactly those tiles was already
    /// available and nothing had written them since.
    [[nodiscard]] size_t num_gathers_reused() const { return _num_gathers_reused; }

  private:
    size_t    _max_nodes;
    double    _zero_tolerance;
    Densify   _densify;
    FuseTiles _fuse;
    CostModel _cost_model;
    size_t    _num_expanded{0};
    size_t    _num_tile_nodes{0};
    size_t    _num_declined{0};
    size_t    _num_screened{0};
    size_t    _num_densified{0};
    size_t    _num_fused{0};
    size_t    _num_gathers_reused{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
