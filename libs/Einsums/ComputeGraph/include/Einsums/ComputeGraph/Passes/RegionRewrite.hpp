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
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/// @brief One region's algebra, before and after a rewrite.
struct RegionDump {
    std::size_t region_index{0}; ///< Which region of the graph, in program order.
    std::size_t node_count{0};   ///< How many nodes it held.
    std::string before;          ///< The raised algebra. Empty unless dumping is on.
    std::string after;           ///< The algebra the client returned. Empty unless dumping is on.
    bool        changed{false};  ///< Whether the client said it rewrote anything.

    /// What the region cost symbolically before and after, as polynomials in space scales.
    ///
    /// Recorded for every ACCEPTED rewrite whether or not dumping is on, because it is the one
    /// thing a report can say about a rewrite that is short enough to print unconditionally and
    /// still answers the question worth asking: did this make the arithmetic smaller? The
    /// algebra itself is far more informative and far too long, which is what dumping is for.
    std::string cost_before;
    std::string cost_after;
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

    /**
     * @brief What this pass did, for @ref PassManager::explain.
     *
     * FINAL, and that is the point rather than an accident. The report has two halves: what the
     * framework did (regions formed, declined and why, and what each accepted rewrite cost
     * before and after) and what the client did. A client that overrode this would silently
     * drop the first half, which is exactly what happened the first time one did - the pass
     * reported its own counters and the structural section vanished from the report that exists
     * to diagnose it. Clients add their lines through @ref describe instead.
     *
     * @return The framework's lines followed by the client's; empty when nothing happened.
     */
    [[nodiscard]] std::vector<std::string> explain() const final;

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
    APIARY_EXPOSE void set_dump(bool on) { _dump = on; }

    /**
     * @brief Check every accepted rewrite's reported cost against the NODES, off by default.
     *
     * The cost line is what a report offers as evidence a rewrite paid, and it is derived from
     * the algebra alone: a term the rewrite built carries whatever cost the client gave it, and
     * nothing compared that to the nodes the lowering then emitted. The after side read zero on
     * every rewrite `MultiTermFactorization` had ever made, for exactly that reason, and went
     * unread until a case asked whether a loop space had really left.
     *
     * With this on, the before side is checked against the flops of the region's own nodes and
     * the after side against the flops of the nodes the lowering emitted, both through
     * @ref symbolic_cost_for. Two derivations of one number, which is what makes either of them
     * evidence. Off by default because it walks the node set once per rewrite and a pipeline in
     * the default manager should not pay for a self-check.
     *
     * @param[in] on Whether to check.
     */
    APIARY_EXPOSE void set_verify_costs(bool on) { _verify_costs = on; }

    /// @brief Reported costs the node set disagreed with, from the last run.
    ///
    /// Empty unless @ref set_verify_costs is on, and empty when the two derivations agree.
    /// @return One line per disagreement, naming the region and both numbers.
    APIARY_EXPOSE APIARY_GETTER("cost_mismatches") [[nodiscard]] std::vector<std::string> cost_mismatches() const {
        return _cost_mismatches;
    }

    /// @brief How many regions were formed on the last run.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("regions_formed") [[nodiscard]] std::size_t regions_formed() const { return _regions_formed; }

    /// @brief How many regions were rewritten on the last run.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("regions_rewritten") [[nodiscard]] std::size_t regions_rewritten() const { return _regions_rewritten; }

    /**
     * @brief Every dump of the last run, rendered as one block of text.
     *
     * The same content as @ref last_dumps, in the form a language without a
     * bound @ref RegionDump can assert on. Empty when nothing was dumped.
     *
     * @return The rendering.
     */
    APIARY_EXPOSE APIARY_GETTER("dump_text") [[nodiscard]] std::string dump_text() const;

  protected:
    /**
     * @brief Rewrite one region's algebra in place.
     *
     * @param[in,out] graph  The graph, for anything the algebra does not carry.
     * @param[in]     region The region being offered.
     * @param[in,out] expr   The algebra. Rewrite in place.
     * @return True when @p expr was changed and should be lowered; false to leave
     *         the region exactly as it is, which costs nothing.
     *
     * A client that returns true must leave @p expr lowerable. Returning true
     * having produced something @ref lower_region refuses is reported as a skip
     * and leaves the region untouched, so a broken client costs a rewrite rather
     * than a wrong number - but it is a bug in the client, not a supported mode.
     *
     * @par What a client may do to the graph, and what it may not
     * The graph is MUTABLE because a rewrite that introduces tensors the user
     * never wrote has to create them somewhere, and a leaf carries a
     * @ref TensorId rather than a description. Registering and declaring tensors
     * is therefore fair game.
     *
     * NODES are not. A region is a range of positions in ``graph.nodes()`` and
     * those positions are live for the whole call: inserting or erasing one
     * moves the region out from under the splice that is about to replace it.
     * A client needing a node outside the region (a setup node holding a
     * fitting, say) records the intent here and emits it after @ref run
     * returns, when no region is open.
     */
    virtual bool rewrite(Graph &graph, Region const &region, TensorExpr &expr) = 0;

    /**
     * @brief A cheap check, before any region is formed, that this pass has work here.
     *
     * Forming regions and raising them costs a walk of the node set and an expression per
     * region, and a client that can rule a graph out from something far cheaper should. A
     * pass in @ref PassManager::create_default runs on every graph anyone optimizes, most of
     * which will have nothing for it; without this the framework's own cost would be paid by
     * every one of them for a guaranteed no-op.
     *
     * Returning false records a @c note_skip and stops, so a decline is still visible in the
     * report rather than silent.
     *
     * @param[in] graph The graph about to be examined.
     * @return True to proceed with region formation. True by default.
     */
    [[nodiscard]] virtual bool applicable(Graph const &graph) const;

    /**
     * @brief What this client wants to say, appended to the framework's report.
     *
     * @return One line per statistic worth reporting; empty by default, which is right for a
     *         client whose whole story is already in the region counts.
     */
    [[nodiscard]] virtual std::vector<std::string> describe() const { return {}; }

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
    /// Record a region turned away, counting the reason for the structural report as well as
    /// for the skip tally. One call rather than two statements at each site, because the two
    /// used to drift: the count said three declines and the tally named one.
    void decline(std::string_view reason, std::string_view detail = {});

  protected:
  private:
    bool                     _dump{false};
    bool                     _verify_costs{false};
    std::vector<std::string> _cost_mismatches;
    std::vector<RegionDump>  _dumps;
    std::size_t              _regions_formed{0};
    std::size_t              _regions_rewritten{0};
    std::size_t              _regions_declined{0};

    /// Why regions were turned away, counted. A subset of @ref skip_reasons: those are every
    /// decline the pass made including per-candidate ones, and these are the region-level ones
    /// the structural report is about.
    std::vector<std::pair<std::string, std::size_t>> _decline_reasons;
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

    /// @copydoc OptimizerPass::tier
    /// Raises every region and lowers it unchanged, so the lowered nodes run the same kernels over the same values
    /// in the same order. The framework's identity test asserts exactly that.
    [[nodiscard]] PassTier tier() const override { return PassTier::BitwiseExact; }

  protected:
    /**
     * @brief Change nothing, and say so.
     * @param[in] graph  Unused.
     * @param[in] region Unused.
     * @param[in] expr   Unused.
     * @return Always true: the region IS lowered, which is the whole point. A
     *         pass that returned false would skip the lowering and test nothing.
     */
    bool rewrite(Graph &graph, Region const &region, TensorExpr &expr) override;
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
