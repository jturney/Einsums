//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file AxisTiling.hpp
 * @brief Slice the free axes of a captured program so its intermediates are produced and
 *        consumed a slice at a time.
 *
 * @par What it is for
 * A program written over all of its indices is the easiest one to read and the one an oracle
 * can be compared against, and it is also the one that declares the largest tensors. The
 * density-fitted MP2 correlation energy written over @c i, @c a, @c j and @c b declares five
 * tensors of @c o^2 @c v^2 elements, which is exactly the object density fitting exists to
 * avoid. The hand-written form of the same energy runs a loop over occupied pairs and never
 * holds more than @c v^2 at once. Nothing about the ALGEBRA differs between the two: which
 * indices are loop variables and which are tensor axes is the whole of it.
 *
 * This pass is that change of schedule, taken from the sizes rather than from a rule about
 * chemistry. It answers three questions and each is read off @ref option::GraphTilingMemoryCap.
 *
 * @par Which statements
 * A loop is built around one intermediate at a time, the largest over the cap first. Its region
 * is that tensor's writers and readers, and then the writers and readers of every other
 * intermediate over the cap that the slice labelling reaches, since the loop has to stream those
 * too. A tensor the labelling reaches that fits the cap is not followed: it stays whole and is
 * written and read through slices of its own buffer, so the statements of other terms that touch
 * it stay outside the loop. That is what lets a coupled-cluster residual tile its particle ladder
 * while the terms that sum over an occupied index of the same amplitudes run as captured. The
 * region need not be contiguous: its statements gather where the loop stands, which is allowed
 * when no statement in between depends on one of them while another depends on it. Every
 * intermediate over the cap that is left gets a region of its own.
 *
 * @par Which axes
 * A candidate is a free axis of the program that no contraction sums over, because slicing a
 * summed index would cut a contraction in half and cost a partial-sum buffer. A reduction into
 * a SCALAR is not such a contraction: it becomes an accumulation across iterations, which is
 * the one thing a loop body already knows how to do, so the axes it sums stay candidates.
 *
 * Feasibility is structural before it is numeric. A candidate set is propagated through the
 * region as a labelling, one label per sliced axis, and a program that cannot carry the
 * labelling consistently is not tileable on that set. On the four-index MP2 energy that is what
 * rejects the two VIRTUAL axes without any appeal to their extents: the exchange permutation
 * exchanges @c a with @c b, so a slice of the permuted tensor would need a slice of its source
 * that the body is not at, while the same permutation leaves @c i and @c j where they are.
 *
 * Among the sets that survive, the one that STREAMS the fewest bytes wins, subject to the cap.
 * Both halves matter. Traffic alone always prefers slicing less, since not slicing at all
 * re-reads nothing; the cap alone would prefer slicing the smallest axes, which on MP2 is the
 * occupied pair read once per virtual pair rather than the other way round. Minimum traffic
 * under a memory cap is the memory-constrained loop fusion the tensor contraction engine
 * literature states (Lam, Cociorva, Baumgartner and Sadayappan; Hartono and others), and this
 * is that formulation over a captured graph.
 *
 * @par How deep
 * One slice at a time is the smallest footprint and the worst kernel efficiency. A chunk of
 * slices is one grouped node, so the depth is the largest chunk whose intermediates still fit
 * the cap. The chunk must divide the slice count, so that every chunk is full and a grouped
 * node's member count is fixed at capture; the depth therefore walks the divisors of the slice
 * count downwards rather than taking the cap's quotient verbatim.
 *
 * @par Blocks instead of points
 * A chunk of points runs the body once per slice, so an operand that does not vary along a
 * sliced axis is read again at every slice of it. The ladder of a density-fitted coupled-cluster
 * residual reads the same integral block at every occupied pair, and density-fitted MP2 reads
 * the factor of one occupied index once per value of the other. Where a contraction of the
 * region has such an operand, the axis it does not carry is cut into RANGES instead: the body
 * keeps the letter at the block's extent, so one contraction covers the whole block and the
 * shared operand is read once per block. An axis no such operand rewards stays a point, and a
 * region with no reusable operand at all keeps the grouped chunk above.
 *
 * The block extents are chosen by the same objective as the axes, fewest bytes streamed under
 * the cap, evaluated per axis. On density-fitted MP2 that is a near-square block of occupied
 * pairs, because each factor is reused across the other index's block; where one operand is
 * reused along every sliced axis alike, only the block's area matters and the fewest axes are
 * cut. A block need not divide its axis: the remainder runs through a second body declared at
 * the remainder's extent, so a program over two ranged axes emits up to four loops, one per
 * combination of full and remainder blocks.
 *
 * @par Half the tiles of two exchangeable axes
 * Where two axes are cut into blocks of one extent and the region's contribution to its one output
 * is unchanged when they are exchanged, together with an exchange of other axes of the output,
 * only the tiles on and above the diagonal run. Each tile above it computes its contribution into a
 * scratch tile and adds it to its own block and, exchanged, to the mirrored one. The particle
 * ladder of a closed-shell coupled-cluster residual is that shape: it is unchanged under exchanging
 * (i,a) with (j,b) because its amplitude is, so it costs about half the arithmetic it did. The
 * exchange is proved, not assumed (@c contribution_symmetry in SymmetryRules.hpp), from symmetries the graph infers or
 * the caller declared on its tensors, and every declaration the proof rests on is checked before the
 * loops on every replay. Two exchangeable axes take the triangle over every grid, and its block is
 * the one the cost model prices cheapest, since halving the tiles changes the kernels' shapes as
 * well as their count.
 *
 * @par Which intermediates
 * Every intermediate carrying a sliced axis is re-declared at slice extents inside the body. An
 * input carrying one is read as a VIEW of the caller's buffer and never as a copy, so a rebind
 * at a new geometry keeps working. An input carrying none is read whole. An output carrying
 * none is the reduction, and becomes a loop-carried accumulation into the caller's tensor.
 *
 * @par The tier
 * The per-slice algebra is the captured algebra, operation for operation. What moves is the
 * ORDER of the reduction that becomes the accumulation, so the pass is re-associating rather
 * than bitwise-exact and is validated against that bound.
 *
 * @par Where the cap comes from
 * A positive cap bounds each intermediate on its own. The default derives a budget instead, the
 * @c --einsums:max-memory option, and holds the whole tree to it: every tensor some node of the
 * tree reads or writes, inputs included and each counted once, as if all were live at the same
 * time. That is conservative, since a tensor freed before another is written still counts, and it
 * makes the question the pass asks the one a user asks: does the program fit. Fitting is not the
 * whole question, though, since an intermediate that takes a large part of the machine costs the
 * kernels that write and read it, so the budget also gives each intermediate a share,
 * @ref option::GraphTilingBudgetShare of it. When the program fits and no intermediate is over its
 * share, the pass declines before it reads a node. Otherwise every intermediate a sliceable
 * statement writes that is over the lesser of its share and what the other tensors leave is a
 * seed, largest first, and its loop's buffers are held to that lesser bound together: every
 * streamed intermediate's block, and the buffers of each remainder loop beside the full block's,
 * since every loop declares its own and all of them live for the whole replay.
 *
 * @par What it declines
 * A program whose largest intermediate already fits the cap, which is what makes the pass a
 * no-op on a form that never needed it, and what makes the order against
 * @ref passes::LaplaceTransform the right way round: a transform that has already dissolved the
 * four-index intermediates leaves this pass nothing to do, and it says so through the tally
 * rather than tiling what is already small.
 *
 * @see option::GraphTilingMemoryCap
 * @versionadded{2.0.0}
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Rewrite a program as a loop over slices of its free axes, under a memory cap.
 *
 * @code
 *   cg::PassManager manager;
 *   auto tiling = std::make_shared<cg::passes::AxisTiling>();
 *   tiling->set_memory_cap(4096);
 *   manager.add(tiling);
 *   graph.apply(manager);
 * @endcode
 *
 * A member of @ref PassManager::create_default, after the algebraic rewrites. By default the cap
 * is what the @c --einsums:max-memory option leaves after every other tensor of the graph, so in the default
 * pipeline the pass acts only on a program that would not fit and otherwise declines without
 * touching the graph. A caller with a smaller footprint to live inside sets a cap of its own.
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT AxisTiling : public OptimizerPass {
  public:
    /// @name Reasons
    /// The codes this pass records its decisions under (@ref Decision::code).
    /// @{
    static constexpr Reason ZeroCap{"axis-tiling.zero-cap", "the memory cap is zero, which is how a caller asks for the captured schedule"};
    static constexpr Reason NoBudget{"axis-tiling.no-budget",
                                     "the cap is derived from --einsums:max-memory, which is zero, so there is no budget to tile against"};
    static constexpr Reason ResultHeld{"axis-tiling.result-held",
                                       "a tensor over the cap is declared as a result, which the caller may read after execute(), so it is "
                                       "held whole; declare it as an intermediate if nothing reads it afterwards"};
    static constexpr Reason FitsBudget{"axis-tiling.fits-budget", "the program's tensors fit the memory budget together and no "
                                                                  "intermediate holds more than its share, so there is nothing to stream"};
    static constexpr Reason FitsCap{"axis-tiling.fits-cap", "the largest intermediate already fits the cap, so there is nothing to stream"};
    static constexpr Reason NoSliceableWriter{"axis-tiling.no-sliceable-writer",
                                              "every intermediate over the cap is written by a statement this pass cannot slice"};
    static constexpr Reason UnknownFeature{"axis-tiling.unknown-feature", "the node carries a feature this pass does not understand"};
    static constexpr Reason OperandWithoutAxes{"axis-tiling.operand-without-axes",
                                               "every axis of one operand is sliced, so the body would name an operand with no axes left"};
    static constexpr Reason StatementBetween{"axis-tiling.statement-between",
                                             "a statement outside the region sits between two of its statements, depending on one while "
                                             "the other depends on it, so they cannot be gathered into one loop"};
    static constexpr Reason BestEffort{
        "axis-tiling.best-effort",
        "no slicing fits the memory budget, so the seed is cut into slices that fit its share of it, or its smallest slices"};
    static constexpr Reason NoAxisSet{"axis-tiling.no-axis-set", "no candidate axis set brings the seed under the cap"};
    static constexpr Reason ElementType{"axis-tiling.element-type", "the region's element type is not one the emitted body can name"};
    static constexpr Reason Sliced{"axis-tiling.sliced", "slice a region into a loop over blocks of its axes"};
    static constexpr Reason Triangle{"axis-tiling.triangle", "run only the tiles on and above the diagonal of two exchangeable axes, "
                                                             "and write each off-diagonal tile's mirror from it"};
    static constexpr Reason TriangleShape{
        "axis-tiling.triangle-shape", "the two sliced axes differ in extent or block, or the region reduces, sums into a tensor or writes "
                                      "more than one caller's tensor, so every tile runs"};
    static constexpr Reason NotExchangeable{"axis-tiling.not-exchangeable",
                                            "the region's contribution is not provably unchanged when its two sliced axes are exchanged, "
                                            "so every tile runs"};
    static constexpr Reason TriangleBacked{"axis-tiling.triangle-backed",
                                           "the region's output is backed by storage, and a mirrored tile lands outside the box the tile "
                                           "streams, so every tile runs"};
    static constexpr Reason BlockPinned{"axis-tiling.block-pinned", "the pinned block extent does not fit the memory the region may take"};
    /// @}

    /// @brief Default-construct, reading the cap from @ref option::GraphTilingMemoryCap.
    APIARY_EXPOSE AxisTiling() = default;

    /// @copydoc OptimizerPass::name
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string name() const override { return "AxisTiling"; }

    /// @copydoc OptimizerPass::phase
    ///
    /// A change of the node set taken for machine reasons, which is what the resource phase is:
    /// the algebra is untouched, so nothing here is saved and everything here is re-derived from
    /// the algebraic form on a load.
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralResource; }
    /// @copydoc OptimizerPass::understood_features
    /// Not views, conjugation or redirected slots: a sliced body re-emits a dotc as a dot and bakes the handle's own object.
    [[nodiscard]] std::optional<NodeFeatures> understood_features() const override {
        return NodeFeatures{} | NodeFeature::PermutationOperators | NodeFeature::ComplexPrefactor | NodeFeature::MixedPrecision |
               NodeFeature::Grouped | NodeFeature::ControlFlow | NodeFeature::Tiled | NodeFeature::RawScalar;
    }

    /// @copydoc OptimizerPass::recurse_into_subgraphs
    ///
    /// A loop body is where an iterative method keeps the work worth tiling: a coupled-cluster
    /// residual lives in the body of its DIIS loop. A region never crosses out of the graph it was
    /// found in, so the loop the rewrite emits nests inside the body, and a loop this pass emitted
    /// is not visited again in the same apply.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// @copydoc OptimizerPass::tier
    ///
    /// The per-slice algebra is bitwise the captured algebra. The reduction that becomes an
    /// accumulation across iterations is not, because a sum split into per-slice partial sums
    /// adds them in a different order.
    [[nodiscard]] PassTier tier() const override { return PassTier::ReAssociating; }

    /// @copydoc OptimizerPass::reset_stats
    void reset_stats() override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;

    /**
     * @brief Override the cap for this pipeline, in bytes.
     * @param[in] bytes Largest a single live intermediate may be. Zero disables the pass, and a
     *            negative value derives the cap from the @c --einsums:max-memory option and the graph's other
     *            tensors.
     *
     * Beats @ref option::GraphTilingMemoryCap, which is the two-level shape every other
     * per-pass knob in this module has.
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE void set_memory_cap(std::int64_t bytes);

    /// @brief The cap in force: this pipeline's override, or the option's value.
    /// @return The cap in bytes; negative when it is derived per graph from the @c --einsums:max-memory option.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("memory_cap") [[nodiscard]] std::int64_t memory_cap() const;

    /**
     * @brief Pin the block extent of a triangle's two axes.
     * @param[in] extent The block, two or more; zero lets the cost model choose, which is the default.
     *
     * A triangle's block is priced from the machine's measured kernel rates, so the block it
     * chooses moves between machines. A test that checks the emission at a given block pins it
     * here. A pinned block that does not fit the memory the region may take declines the triangle.
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE void set_block_extent(std::int64_t extent);

    /// @brief How many regions the last apply tiled.
    ///
    /// The other getters describe the first of them, the region around the largest intermediate;
    /// @ref explain names every one.
    /// @return The count, zero when the pass declined.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("num_tiled") [[nodiscard]] std::size_t num_tiled() const noexcept { return _num_tiled; }

    /// @brief The sliced axes of the last decision, named ``tensor[position]``.
    /// @return One entry per sliced axis, outermost first.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("axis_names") [[nodiscard]] std::vector<std::string> axis_names() const { return _axis_names; }

    /// @brief The index letter each sliced axis carries, where a contraction names one.
    /// @return One entry per sliced axis, empty where no node names the axis with a letter.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("axis_letters") [[nodiscard]] std::vector<std::string> axis_letters() const { return _axis_letters; }

    /// @brief The extent of each sliced axis.
    /// @return One entry per sliced axis, in the same order as @ref axis_names.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("axis_extents") [[nodiscard]] std::vector<std::int64_t> axis_extents() const { return _axis_extents; }

    /// @brief How many slices the chosen axes cut the program into.
    /// @return The product of the sliced extents, zero when the pass declined.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("slice_count") [[nodiscard]] std::size_t slice_count() const noexcept { return _slice_count; }

    /// @brief How many slices one iteration of the emitted loop handles.
    /// @return The chunk depth, the product of the block extents for a range emission, zero when
    ///         the pass declined.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("depth") [[nodiscard]] std::size_t depth() const noexcept { return _depth; }

    /// @brief How many iterations the emitted loops run.
    /// @return The iterations summed over every emitted loop, which is @ref slice_count divided
    ///         by @ref depth when the chunks divide the slices, zero when the pass declined.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("iterations") [[nodiscard]] std::size_t iterations() const noexcept { return _iterations; }

    /// @brief The largest single intermediate any examined graph declared before the rewrite, in bytes.
    /// @return The size over the graph and every loop body the pass visited, zero when none was examined.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("largest_before") [[nodiscard]] std::size_t largest_before() const noexcept { return _largest_before; }

    /// @brief The largest single intermediate the body declares after the rewrite, in bytes.
    /// @return The size at the chosen depth, zero when the pass declined.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("largest_after") [[nodiscard]] std::size_t largest_after() const noexcept { return _largest_after; }

    /// @brief The intermediates the rewrite streams, by name.
    /// @return One entry per re-declared tensor, in region order.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("streamed") [[nodiscard]] std::vector<std::string> streamed() const { return _streamed; }

    /// @brief The tensors the rewrite left whole, by name.
    /// @return One entry per tensor read or written without a sliced axis.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("whole") [[nodiscard]] std::vector<std::string> whole() const { return _whole; }

    /// @brief How the last decision emits a chunk.
    /// @return ``"range"`` when some sliced axis is cut into blocks the body keeps as an axis,
    ///         ``"triangle"`` when two such axes are exchangeable and only the tiles on and above
    ///         their diagonal run, ``"grouped"`` when every sliced axis is a point and a chunk is one
    ///         grouped node per family, and an empty string when the pass declined.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("emission") [[nodiscard]] std::string emission() const { return _emission; }

    /// @brief The block extent of each sliced axis.
    /// @return One entry per sliced axis, in the same order as @ref axis_names: the extent of a
    ///         full block of a ranged axis, and one for an axis taken a point at a time. Empty when
    ///         the pass declined or the emission is grouped.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("block_extents") [[nodiscard]] std::vector<std::int64_t> block_extents() const { return _block_extents; }

    /// @brief How many loops the rewrite emitted.
    /// @return One for a grouped emission or a range emission whose blocks divide their axes, up to
    ///         four for two ranged axes with remainders, zero when the pass declined.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("loops") [[nodiscard]] std::size_t loops() const noexcept { return _loops; }

    /// @brief The sign a triangle's mirrored tiles carry.
    /// @return +1 when the region's contribution is symmetric under exchanging its two sliced axes,
    ///         -1 when antisymmetric, and 0 when the last decision is not a triangle.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("triangle_sign") [[nodiscard]] int triangle_sign() const noexcept { return _triangle_sign; }

    /// @brief The tensor the reduction accumulates into across iterations, if there is one.
    /// @return Its name, or an empty string when the region has no such reduction.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("accumulator") [[nodiscard]] std::string accumulator() const { return _accumulator; }

    /// @brief The tensors a contraction of any tiled region sums one of its sliced axes into.
    /// @return Their names. Each is scaled by its contraction's destination prefactor once, before
    ///         the loop, and added to at every iteration; empty when no contraction sums a sliced axis.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("summed_into") [[nodiscard]] std::vector<std::string> summed_into() const { return _summed_into; }

  protected:
    /// @copydoc OptimizerPass::run
    bool transform(Graph &graph) override;

  private:
    /// @brief Tile every region of one graph of the tree, without descending into its sub-graphs.
    /// @param[in,out] graph The graph.
    /// @return True when a region was replaced by a loop.
    bool tile_graph(Graph &graph);

    /// @brief The memory one region may take.
    struct Allowance {
        /// The cap on each intermediate alone; an intermediate that fits it is not streamed.
        std::size_t cap{0};
        /// The memory budget the loop's buffers are held to together with every tensor outside
        /// the loop, or zero when @ref cap alone bounds them.
        std::size_t budget{0};
        /// The most the loop's buffers may hold together whatever the budget leaves, when
        /// @ref budget is set.
        std::size_t share{0};
        /// What the tree's tensors hold before the region is tiled, when @ref budget is set.
        std::size_t total{0};
    };

    /// @brief Tile the region around one seed, if any candidate axis set can.
    /// @param[in,out] graph     The graph, rewritten when a region is tiled.
    /// @param[in]     seed      An intermediate over the cap that a sliceable statement writes.
    /// @param[in]     allowance The memory the region may take.
    /// @return True when the region was replaced by a loop.
    bool tile_region(Graph &graph, TensorId seed, Allowance const &allowance);

    /// @brief The bytes of every tensor the tree around @p graph names, each counted once.
    ///
    /// The tree is the one whose root this pass has walked out to through @ref _enclosing, so a
    /// loop body counts its parent's tensors too. A view owns no storage and is not counted, and
    /// neither is a tensor no node of its graph reads or writes, which is never allocated.
    /// @param[in] graph A graph of the tree.
    /// @return The total in bytes.
    [[nodiscard]] std::size_t footprint(Graph const &graph) const;

    std::int64_t _memory_cap{0};
    bool         _cap_explicit{false};
    std::int64_t _block_pin{0};

    std::size_t               _num_tiled{0};
    std::size_t               _slice_count{0};
    std::size_t               _depth{0};
    std::size_t               _iterations{0};
    std::size_t               _largest_before{0};
    std::size_t               _largest_after{0};
    std::vector<std::string>  _axis_names;
    std::vector<std::string>  _axis_letters;
    std::vector<std::int64_t> _axis_extents;
    std::vector<std::string>  _streamed;
    std::vector<std::string>  _whole;
    /// One line per declared result over its cap that was held whole; reported at every
    /// verbosity, because the fix is in the caller's declaration and nothing else says so.
    std::vector<std::string>  _held_results;
    std::string               _accumulator;
    std::vector<std::string>  _summed_into;
    std::string               _emission;
    std::vector<std::int64_t> _block_extents;
    std::size_t               _loops{0};
    int                       _triangle_sign{0};
    std::vector<std::string>  _regions;

    /// For every graph this apply has visited, the graph that holds it and the position of the
    /// node there that holds it, so a run on a loop body can resolve what its parent declared.
    std::unordered_map<Graph const *, std::pair<Graph *, std::size_t>> _enclosing;
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
