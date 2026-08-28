//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file RegionRewrite.hpp
 * @brief The pass shape every region rewrite has, and the one that rewrites nothing.
 *
 * @par What the base does, so a client does not
 * @ref RegionRewrite forms regions deterministically, raises each one, hands the
 * algebra to @ref RegionRewrite::rewrite, and lowers whatever comes back. A
 * client writes only the middle step, which is the point: the region rule, the
 * escape analysis, the splice and the diagnostics are the parts that were hard
 * to get right, and every client getting them from one place is what makes the
 * next one cheap and the next bug findable.
 *
 * @par The dump
 * Every region rewrite can dump the raised expression before and after, because
 * an optimizer that rewrites mathematics will eventually produce a wrong number
 * and the diagnosis has to start somewhere readable. A diffable before-and-after
 * of the ALGEBRA is a far better
 * bug report than two node-list dumps, and this is where it is produced: set
 * ``einsums:graph:dump-regions``, or read @ref RegionRewrite::last_dumps from a
 * test, which is the same content without having to scrape stderr.
 *
 * @par Refusal is a first-class outcome
 * A region that cannot be raised, and a rewritten expression that cannot be
 * lowered, both leave the graph exactly as it was and record a @c note_skip
 * reason. That is not defensive coding; it is what makes a pipeline that looks
 * inert answerable, since the skip tally names the gate rather than leaving
 * "no optimizations applied" to mean two different things.
 *
 * @see TensorExpr.hpp for the IR, the region rule and the raise/lower pair
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/TensorExpr.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/// @brief One region's algebra, before and after a rewrite.
struct RegionDump {
    std::size_t region_index{0}; ///< Which region of the graph, in program order.
    std::size_t node_count{0};   ///< How many nodes it held.
    std::string before;          ///< The raised algebra.
    std::string after;           ///< The algebra the client returned, equal to @ref before when it changed nothing.
    bool        changed{false};  ///< Whether the client said it rewrote anything.
};

/**
 * @brief Base class for a pass that rewrites regions through @ref TensorExpr.
 *
 * Derive and override @ref rewrite. Everything else - forming regions, raising,
 * lowering, skip reasons, the before/after dump, the statistics @c explain reads
 * - comes from here.
 *
 * Exposed to Python despite being abstract, as @ref OptimizerPass is and for the
 * same reason: the binding for a concrete pass needs its DIRECT base to be a
 * bound type, or the generated class has no inheritance chain and
 * ``PassManager.add`` refuses it as an unrelated type. Nothing constructs one.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT RegionRewrite : public OptimizerPass {
  public:
    /**
     * @brief Raise every region, offer it to @ref rewrite, lower what changed.
     * @param[in,out] graph The graph to rewrite.
     * @return True when any region was rewritten.
     */
    bool run(Graph &graph) override;

    /// @brief Region rewrites are machine-independent algebra, so their output is saved.
    /// @return @ref PassPhase::StructuralAlgebraic.
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralAlgebraic; }

    /// @brief Zero the per-apply counters.
    void reset_stats() override;

    /// @brief What this pass did, for @ref PassManager::explain.
    /// @return One line per statistic worth reporting; empty when nothing happened.
    [[nodiscard]] std::vector<std::string> explain() const override;

    /**
     * @brief The before/after algebra of every region this run raised.
     *
     * Populated whenever ``einsums:graph:dump-regions`` is set or
     * @ref set_dump is on. A test asserts on this rather than scraping stderr,
     * which is also why it holds the rendering rather than printing it: a dump
     * a test cannot read is a dump nobody checks.
     *
     * @return The dumps, in region order.
     */
    [[nodiscard]] std::vector<RegionDump> const &last_dumps() const { return _dumps; }

    /// @brief Collect the before/after dumps regardless of the option.
    /// @param[in] on Whether to collect.
    void set_dump(bool on) { _dump = on; }

    /// @brief How many regions were formed on the last run.
    /// @return The count.
    [[nodiscard]] std::size_t regions_formed() const { return _regions_formed; }

    /// @brief How many regions were rewritten on the last run.
    /// @return The count.
    [[nodiscard]] std::size_t regions_rewritten() const { return _regions_rewritten; }

    /**
     * @brief Every dump of the last run, rendered as one block of text.
     *
     * The same content as @ref last_dumps, in the form a language without a
     * bound @ref RegionDump can assert on. Empty when nothing was dumped.
     *
     * @return The rendering.
     */
    [[nodiscard]] std::string dump_text() const;

  protected:
    /**
     * @brief Rewrite one region's algebra in place.
     *
     * @param[in]     graph  The graph, for anything the algebra does not carry.
     * @param[in]     region The region being offered.
     * @param[in,out] expr   The algebra. Rewrite in place.
     * @return True when @p expr was changed and should be lowered; false to leave
     *         the region exactly as it is, which costs nothing.
     *
     * A client that returns true must leave @p expr lowerable. Returning true
     * having produced something @ref lower_region refuses is reported as a skip
     * and leaves the region untouched, so a broken client costs a rewrite rather
     * than a wrong number - but it is a bug in the client, not a supported mode.
     */
    virtual bool rewrite(Graph const &graph, Region const &region, TensorExpr &expr) = 0;

    /**
     * @brief The smallest region this pass is interested in.
     *
     * One node by default. A pass that folds pairs should return two, so a
     * single-node region is never formed, raised and lowered for nothing.
     *
     * @return The minimum node count.
     */
    [[nodiscard]] virtual std::size_t min_region_nodes() const { return 1; }

  private:
    bool                    _dump{false};
    std::vector<RegionDump> _dumps;
    std::size_t             _regions_formed{0};
    std::size_t             _regions_rewritten{0};
    std::size_t             _regions_declined{0};
};

/**
 * @brief Raise every region and lower it unchanged.
 *
 * The framework's own test, in pass form, and the gate everything else rests on:
 * raising an expression and lowering it must produce a graph that computes
 * bit-for-bit what the original did. Nothing about that is guaranteed by construction. Lowering
 * rebuilds each node from the algebra rather than reusing what it raised, so
 * anything the IR fails to carry - a conjugation flag, a destination prefactor,
 * an index list a pass had rewritten through the live block - comes back as a
 * different number rather than as a missing field nobody notices.
 *
 * It is a pass rather than a test helper so the differential fuzz corpus can
 * drive it: the corpus is where the shapes this would otherwise never see live.
 * Not in @ref PassManager::create_default, because rewriting nothing is not an
 * optimization; it is registered where the fuzzers can reach it.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT RegionIdentity : public RegionRewrite {
  public:
    /// @brief Default-construct. Explicit so the binding codegen has a constructor to annotate.
    APIARY_EXPOSE RegionIdentity() = default;

    /// @brief The pass name.
    /// @return ``"RegionIdentity"``.
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string name() const override { return "RegionIdentity"; }

    /// @name The Python spelling of the base's introspection
    ///
    /// Forwarders rather than annotations on @ref RegionRewrite itself. The base
    /// is abstract and is not a type Python can hold, so the codegen has nothing
    /// to attach an inherited member to; every other pass in this module spells
    /// its counters on the concrete class for the same reason.
    /// @{

    /// @brief How many regions the last run formed.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("regions_formed") [[nodiscard]] std::size_t python_regions_formed() const { return regions_formed(); }

    /// @brief How many regions the last run rewrote.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("regions_rewritten") [[nodiscard]] std::size_t python_regions_rewritten() const {
        return regions_rewritten();
    }

    /// @brief Collect the before/after dumps regardless of the option.
    /// @param[in] on Whether to collect.
    APIARY_EXPOSE void set_dump(bool on) { RegionRewrite::set_dump(on); }

    /// @brief The last run's dumps as one block of text.
    /// @return The rendering, empty when nothing was dumped.
    APIARY_EXPOSE APIARY_GETTER("dump_text") [[nodiscard]] std::string python_dump_text() const { return dump_text(); }
    /// @}

  protected:
    /**
     * @brief Change nothing, and say so.
     * @param[in] graph  Unused.
     * @param[in] region Unused.
     * @param[in] expr   Unused.
     * @return Always true: the region IS lowered, which is the whole point. A
     *         pass that returned false would skip the lowering and test nothing.
     */
    bool rewrite(Graph const &graph, Region const &region, TensorExpr &expr) override;
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
