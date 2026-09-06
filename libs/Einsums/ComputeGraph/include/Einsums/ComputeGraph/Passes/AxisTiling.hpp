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
 * Not a member of @ref PassManager::create_default. The cap's default is far above anything a
 * captured program declares, so a default-pipeline membership would only ever pay for the
 * analysis; a caller who has a footprint to live inside says so and adds the pass.
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT AxisTiling : public OptimizerPass {
  public:
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

    /// @copydoc OptimizerPass::tier
    ///
    /// The per-slice algebra is bitwise the captured algebra. The reduction that becomes an
    /// accumulation across iterations is not, because a sum split into per-slice partial sums
    /// adds them in a different order.
    [[nodiscard]] PassTier tier() const override { return PassTier::ReAssociating; }

    /// @copydoc OptimizerPass::run
    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::reset_stats
    void reset_stats() override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;

    /**
     * @brief Override the cap for this pipeline, in bytes.
     * @param[in] bytes Largest a single live intermediate may be. Zero disables the pass.
     *
     * Beats @ref option::GraphTilingMemoryCap, which is the two-level shape every other
     * per-pass knob in this module has.
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE void set_memory_cap(std::int64_t bytes);

    /// @brief The cap in force: this pipeline's override, or the option's value.
    /// @return The cap in bytes.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("memory_cap") [[nodiscard]] std::int64_t memory_cap() const;

    /// @brief How many regions the last apply tiled.
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
    /// @return The chunk depth, zero when the pass declined.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("depth") [[nodiscard]] std::size_t depth() const noexcept { return _depth; }

    /// @brief How many iterations the emitted loop runs.
    /// @return @ref slice_count divided by @ref depth, zero when the pass declined.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("iterations") [[nodiscard]] std::size_t iterations() const noexcept { return _iterations; }

    /// @brief The largest single intermediate the region declared before the rewrite, in bytes.
    /// @return The size, zero when no region was examined.
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

    /// @brief The tensor the reduction accumulates into across iterations, if there is one.
    /// @return Its name, or an empty string when the region has no such reduction.
    /// @versionadded{2.0.0}
    APIARY_EXPOSE APIARY_GETTER("accumulator") [[nodiscard]] std::string accumulator() const { return _accumulator; }

  private:
    std::int64_t _memory_cap{0};
    bool         _cap_explicit{false};

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
    std::string               _accumulator;
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
