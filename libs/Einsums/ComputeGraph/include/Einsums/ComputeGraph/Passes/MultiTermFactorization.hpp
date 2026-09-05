//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file MultiTermFactorization.hpp
 * @brief Choose how to parenthesize every contraction in a region, and what the terms may share.
 *
 * @par The two decisions, which are one decision
 * A product of three or more tensors has many contraction orders and they differ by whole factors
 * of the scaling. Several such products, appearing in the same region, often want the SAME partial
 * product, and whether they can have it depends on the order each one was given. So choosing the
 * orders and choosing the shared intermediates cannot be done separately: fixing the orders first
 * leaves sharing whatever the orders happened to allow, and fixing the sharing first assumes an
 * order for the very products being shared.
 *
 * @code
 * R1[i,j] = A[i,k] B[k,l] C[l,j]
 * R2[i,j] = A[i,k] B[k,l] D[l,j]
 * @endcode
 *
 * Contract each left to right and both build ``(A B)[i,l]``, once each. Contract the second right
 * to left and it builds ``(B D)[k,j]`` instead, and there is nothing to share. Neither order is
 * wrong and neither is locally better; the difference only shows up when the two terms are looked
 * at together, which is the whole reason this pass exists and the reason `CSE` cannot find it:
 * `CSE` matches nodes that are already identical, and here the nodes to be shared are ones nobody
 * has written yet.
 *
 * @par What it is not
 * `DistributiveFactoring` factors a shared operand out of a SUM of contractions into one output:
 * ``A B1 + A B2`` becomes ``A (B1 + B2)``. That is a different rewrite (it changes how many
 * contractions there are, not how each one is bracketed) and the two compose rather than compete.
 * `ContractionPlanning` re-parenthesizes chains, but one chain at a time and only where every
 * operand reads as a flat matrix, so it cannot trade a locally worse order in one term for a
 * shared intermediate in another.
 *
 * @par How the search is bounded
 * Per term, the optimal binary tree comes from the standard subset dynamic program: the best way
 * to contract a set of factors, built up from the best way to contract each of its subsets. That
 * is @c 3^N in the factor count, so the factor count is capped and a term above the cap is
 * declined rather than approximated. The cap is TEN, which is what the opposite-spin correlation
 * energy needs: flattened through its direct product and its dot, and with the Laplace transform's
 * exponentials already in place, it is a nine-factor product over seven letters, and @c 3^9 is
 * nineteen thousand subsets.
 *
 * Across terms, the candidates are PAIRS of factors that occur in more than one term. Restricting
 * to pairs is what keeps the candidate set quadratic instead of exponential, and it costs less
 * than it looks: a shared subtree of three factors is built out of a shared pair, so committing
 * pairs one at a time and re-solving discovers the larger subtree over successive rounds. What it
 * genuinely cannot find is a shared triple whose every pair is unprofitable on its own.
 *
 * Both loops check @ref SearchBudget. A pass that runs out keeps the best assignment it had
 * reached, applies it, and reports that it was cut off, so exhausting the budget costs
 * optimization rather than correctness or determinism.
 *
 * @par What a rewrite is measured against
 * The CAPTURED bracketing, priced through the same model the search ranks its own trees with: the
 * product as the author wrote it, plus every definition the flattening dissolves, since those go
 * away if the rewrite is taken. Comparing against the searched cost instead asks whether a search
 * improves on itself, which it never does, and leaves the pass unable to fire on a re-bracketing
 * that shares nothing. That is the case a decoupled energy expression is: one statement, one
 * product, and a tree that is several scale orders cheaper than the one the equations were written
 * in.
 *
 * @par Why it is off by default
 * `einsums:graph:structural-search`. Every other structural-algebraic pass is a recognizer whose
 * runtime is a function of the node count; this one's is a function of how many candidates the
 * graph offers, which nobody can predict from outside. The saved IR exists precisely so that a
 * search runs once and a replay does not, so the interactive default is off and the
 * capture-and-save workflow turns it on.
 *
 * @par The result cache
 * The search is the expensive thing this pass does and it is a pure function of the region's
 * structure, so its answer is kept and replayed when a structurally identical region comes back.
 * The key is @ref Graph::content_hash together with the region's node span and the factor cap;
 * everything a plan depends on is inside those. The entry holds a PLAN and not a graph: the shared
 * pairs that were committed, in commit order, and the contraction tree chosen for each term. Both
 * are written as positions and bitmasks rather than as `TensorId`s, which is what lets a plan
 * found on one graph apply to another that hashes the same, and it is the same discipline the
 * saved IR uses for exactly the same reason.
 *
 * The case it pays for is a `Pipeline` whose stages present the same program: the second stage
 * flattens, replays, and emits, and never runs the subset program at all. It does NOT pay for
 * re-running a script, which is a new process with an empty cache; that case is what saving the
 * optimized IR is for. The cache lives on the pass instance and is cleared with
 * @ref clear_cache, rather than being a process-global table, for the reason the per-pipeline
 * budget override exists: a shared mutable table makes two pipelines unrunnable at once.
 *
 * A search that ran out of budget is never stored, so a plan that comes back out is one an
 * unbounded search would have found and the budget does not belong in the key.
 *
 * @par Why the tier is re-associating
 * Re-bracketing a product changes the order the sums are accumulated in, which is the definition
 * of this tier. The pass declares a norm-relative bound rather than bit equality, and is validated
 * against it.
 *
 * @see RegionRewrite.hpp for the framework this is a client of
 * @see SymbolicCost.hpp for the ranking, and for why the comparison has to be a total order
 * @see DistributiveFactoring.hpp for the sum-factoring rewrite this composes with
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Passes/RegionRewrite.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief What one region's search decided, written so a structurally identical region can replay it.
 *
 * Positions and bitmasks, never a @c TensorId and never a node. Two graphs that hash the same
 * flatten to the same terms in the same order, so a factor's position identifies it across both,
 * where its tensor id does not: the IR's own dense numbering makes the same argument.
 */
struct FactorizationPlan {
    /// @brief One term's contraction tree, as the subset program left it.
    struct Tree {
        bool                       ok{false}; ///< Whether a tree was found for this term.
        std::vector<std::uint32_t> split;     ///< split[mask] is the left half chosen for that mask.
        std::vector<std::uint8_t>  resolved;  ///< Whether split[mask] means anything.
    };

    /// Whether the search found anything worth emitting. False is worth caching too: it is the
    /// answer that took the whole search to reach.
    bool rewrites{false};

    /// Committed shared pairs in commit order. Each entry lists its occurrences as
    /// ``(term, left factor, right factor)``, in the factor numbering left by the previous commits.
    std::vector<std::vector<std::array<std::size_t, 3>>> commits;

    /// One entry per term, in the order the region flattens them.
    std::vector<Tree> trees;
};

/**
 * @brief Jointly choose contraction orders and shared intermediates over one region.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("mtf");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("i,l <- i,k ; k,l", 0.0, &T1, 1.0, A, B);     // whatever order the author wrote
 *     cg::einsum("i,j <- i,l ; l,j", 0.0, &R1, 1.0, T1, C);
 *     cg::einsum("k,j <- k,l ; l,j", 0.0, &T2, 1.0, B, D);     // the other bracketing
 *     cg::einsum("i,j <- i,k ; k,j", 0.0, &R2, 1.0, A, T2);
 * }
 * cg::PassManager pm;
 * auto pass = std::make_shared<cg::passes::MultiTermFactorization>();
 * pass->set_search_enabled(true);      // or set einsums:graph:structural-search
 * pm.add(pass);
 * graph.apply(pm);
 * // Both terms now read one shared (A B), and T2 is gone.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums.graph as cg
 * mtf = cg.MultiTermFactorization()
 * mtf.set_search_enabled(True)
 * pm = cg.PassManager(); pm.add(mtf); pm.add(cg.Materialization()); pm.run(g)
 * # mtf.num_shared -> 1   (getters are properties, not methods)
 * @endcode
 *
 * @par Limitations
 * - A term's factors are flattened out of the captured chain only through intermediates the region
 *   can dissolve: written once, overwritten rather than accumulated, carrying a product prefactor
 *   of one, and read only by statements that all resolve to ONE consumer once the folding is done.
 *   Anything else stays a factor in its own right, which is correct but hides the products inside
 *   it from the search.
 * - Three node kinds present the product the flattener reads: a contraction, a
 *   @c OpKind::DirectProduct (the same product with no summed letter) and a @c OpKind::Dot (the
 *   same product summed over every letter). An amplitude formed by a contraction, scaled by a
 *   direct product and reduced by a dot is therefore one product rather than a stored leaf every
 *   candidate has to rebuild.
 * - Shared occurrences must be in DIFFERENT terms. Two occurrences of one pair inside one term
 *   would have to be applied together under a disjointness check, since applying one moves the
 *   other's factor positions and the two may name a factor in common.
 * - Factors are capped at @ref max_factors, because the per-term program is exponential in that
 *   number and a cap is the honest way to bound it.
 * - A factor that repeats an index letter is a diagonal access, whose cost this pass does not
 *   model; such a term is declined.
 * - Shared candidates are pairs. See the file note for what that does and does not reach.
 * - A candidate that would leave a term with a single factor is declined: the result would be a
 *   copy rather than a contraction, which is `CSE`'s business and not a shape this emits.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT MultiTermFactorization : public RegionRewrite {
  public:
    /// @brief Default-construct. Explicit so the binding codegen has a constructor to annotate.
    APIARY_EXPOSE MultiTermFactorization() = default;

    /// @brief The pass name.
    /// @return ``"MultiTermFactorization"``.
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string name() const override { return "MultiTermFactorization"; }

    /// @copydoc OptimizerPass::tier
    /// Re-bracketing a product changes the order its sums accumulate in. That is what this tier
    /// is, and it is why this pass declares a norm-relative bound rather than bit equality.
    [[nodiscard]] PassTier tier() const override { return PassTier::ReAssociating; }

    /**
     * @brief Run the search on this pipeline whatever ``einsums:graph:structural-search`` says.
     *
     * The programmatic half of the option, and it exists for the reason ``PassManager::enable``
     * does: a driver that had only the option would have to mutate process-global configuration
     * to exercise one pipeline, which also makes it unrunnable concurrently. An explicit setting
     * wins over the option in both directions.
     *
     * @param[in] on Whether to search.
     */
    APIARY_EXPOSE void set_search_enabled(bool on) {
        _search_enabled  = on;
        _search_explicit = true;
    }

    /// @brief Whether the search will run, taking the explicit setting over the option.
    /// @return True when this pass will look for anything.
    APIARY_EXPOSE APIARY_GETTER("search_enabled") [[nodiscard]] bool search_enabled() const;

    /// @brief The largest number of factors a term may have before it is declined.
    /// @return The cap. The per-term program is @c 3^N in this number.
    APIARY_EXPOSE APIARY_GETTER("max_factors") [[nodiscard]] std::size_t max_factors() const { return _max_factors; }

    /// @brief Set the factor cap.
    /// @param[in] cap The new cap, clamped to at least two.
    APIARY_EXPOSE void set_max_factors(std::size_t cap) { _max_factors = cap < 2 ? 2 : cap; }

    /// @brief How many multi-factor terms the search re-bracketed.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_rebracketed") [[nodiscard]] std::size_t num_rebracketed() const { return _num_rebracketed; }

    /// @brief How many intermediates it introduced for more than one term to share.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_shared") [[nodiscard]] std::size_t num_shared() const { return _num_shared; }

    /// @brief How many captured intermediates it dissolved into their consumers.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_inlined") [[nodiscard]] std::size_t num_inlined() const { return _num_inlined; }

    /**
     * @brief Keep and replay plans on this pipeline whatever ``einsums:graph:factorization-cache``
     *        says.
     *
     * The programmatic half of the option, for the reason @ref set_search_enabled is: a driver
     * that had only the option would have to mutate process-global configuration to exercise one
     * pipeline. An explicit setting wins over the option in both directions, and switching the
     * cache off does not discard what it already holds; @ref clear_cache does that.
     *
     * @param[in] on Whether to cache.
     */
    APIARY_EXPOSE void set_cache_enabled(bool on) {
        _cache_enabled  = on;
        _cache_explicit = true;
    }

    /// @brief Whether plans will be kept, taking the explicit setting over the option.
    /// @return True when this pass will store and replay plans.
    APIARY_EXPOSE APIARY_GETTER("cache_enabled") [[nodiscard]] bool cache_enabled() const;

    /// @brief Forget every plan this pass has kept.
    ///
    /// The explicit clear the cache is specified to have. Nothing invalidates an entry on its own,
    /// because nothing can: an entry is keyed on the content hash of the graph that produced it,
    /// so a changed graph misses rather than reading a stale plan.
    APIARY_EXPOSE void clear_cache() { _cache.clear(); }

    /// @brief How many regions replayed a plan instead of searching, this apply.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_cache_hits") [[nodiscard]] std::size_t num_cache_hits() const { return _num_cache_hits; }

    /// @brief How many regions searched because no plan was held for them, this apply.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_cache_misses") [[nodiscard]] std::size_t num_cache_misses() const { return _num_cache_misses; }

    /// @brief How many plans are held.
    /// @return The count, which survives @ref reset_stats and is zeroed only by @ref clear_cache.
    APIARY_EXPOSE APIARY_GETTER("cache_size") [[nodiscard]] std::size_t cache_size() const { return _cache.size(); }

    /**
     * @brief Whether the last run stopped because its wall-clock allowance ran out.
     *
     * A report that could not distinguish this from "the graph was already optimal" would be
     * useless in exactly the case a budget exists for, so the pass says which one happened.
     *
     * @return True when the search was cut off.
     */
    APIARY_EXPOSE APIARY_GETTER("was_cut_off") [[nodiscard]] bool was_cut_off() const { return _cut_off; }

    /// @brief Zero the per-apply counters.
    void reset_stats() override;

  protected:
    /// @brief This pass's own lines, appended to the framework's region report.
    /// @return One line per statistic worth reporting.
    [[nodiscard]] std::vector<std::string> describe() const override;

    /**
     * @brief Search one region's algebra and rewrite it in place.
     * @param[in,out] graph  The graph, for extents and for declaring the tensors it introduces.
     * @param[in]     region The region being offered.
     * @param[in,out] expr   The algebra, rewritten in place.
     * @return True when anything changed.
     */
    bool rewrite(Graph &graph, Region const &region, TensorExpr &expr) override;

    /// @brief Decline before forming regions when the search is switched off.
    /// @param[in] graph The graph about to be examined.
    /// @return True when the search is enabled and the graph holds at least two contractions.
    [[nodiscard]] bool applicable(Graph const &graph) const override;

    /// @brief Nothing to search in a one-node region.
    /// @return Two.
    [[nodiscard]] std::size_t min_region_nodes() const override { return 2; }

  private:
    /// @brief A cached plan's identity: which graph, which region of it, under which cap.
    ///
    /// The region is named by its node span rather than by an ordinal, because a span is what the
    /// content hash already covers and an ordinal would silently rename every region after one
    /// that failed to raise.
    using PlanKey = std::tuple<std::uint64_t, std::size_t, std::size_t, std::size_t>;

    bool        _search_enabled{false};
    bool        _search_explicit{false};
    bool        _cache_enabled{true};
    bool        _cache_explicit{false};
    std::size_t _max_factors{10};
    std::size_t _num_rebracketed{0};
    std::size_t _num_shared{0};
    std::size_t _num_inlined{0};
    std::size_t _num_cache_hits{0};
    std::size_t _num_cache_misses{0};
    bool        _cut_off{false};

    std::map<PlanKey, FactorizationPlan> _cache;

    /// The hash of the graph this apply is running over, taken in @ref applicable before any
    /// region has been rewritten. Mutable because that hook is the only one called exactly once
    /// per graph, and taking the hash later would digest a graph an earlier region already moved.
    mutable std::uint64_t _graph_key{0};
    mutable bool          _graph_key_valid{false};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
