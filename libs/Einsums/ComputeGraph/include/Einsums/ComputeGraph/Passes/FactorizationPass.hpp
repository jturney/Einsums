//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/ComputeGraph/Passes/RegionRewrite.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Replace a tagged tensor by a provider's factors, and re-associate around them.
 *
 * The one pass every factorization method goes through. It finds a contraction with a tagged
 * operand, asks each @ref FactorizationProvider claiming that tag for a chain of factors,
 * brackets the substituted product with the subset search, costs the bracketing against what
 * was there, and takes the cheapest offer that is actually cheaper.
 *
 * @par What it does to one contraction
 * Given @c C[c] @c = @c f @c * @c T[t] @c * @c O[o] with @c T tagged, and a provider saying
 * @c T[t] @c = @c A[a] @c * @c B[b] summed over the letters @c a and @c b share that @c t does
 * not, the rewrite is:
 *
 * @code
 * X[x] = B[b] * O[o]
 * C[c] = f * A[a] * X[x]
 * @endcode
 *
 * with @c x the letters of @c b and @c o that @c a or @c c still needs. Substituting alone
 * would give a three-operand contraction, which has no node form and would be more arithmetic
 * besides; the regrouping is the entire point, and DF's saving is exactly this.
 *
 * A provider offering more than two factors gets the same treatment over a longer product. The
 * leaves handed to the search are the provider's factors and the other operand, every interior
 * node of the chosen tree becomes a declared intermediate and a binary contraction, and the
 * auxiliary letters are ordinary letters to the ranking.
 *
 * @par What it declines, and why each is a real case
 * - A tagged tensor some node WRITES. Its factors would have to be refitted whenever it
 *   changed, and the setup body this pass emits runs once per bound problem. Only an operand
 *   nothing in the graph produces is safely factorizable this way.
 * - A TWO-FACTOR split whose factors cannot be told apart by which one pairs with the other
 *   operand. The decomposition needs one factor carrying the letters shared with @c O and one
 *   not; if both do or neither does, there is no regrouping to make. Not asked of a longer
 *   chain, which has as many ways to meet the other operand as it has links and whose answer
 *   is the bracketing the search returns.
 * - A chain with more factors than the search is allowed. The subset program is exponential in
 *   the leaf count, and a cap is the honest way to bound it.
 * - A decomposed form that is not symbolically cheaper. Substituting an approximation and
 *   getting slower is the worst of both, and the comparison is the total order of
 *   @ref SymbolicCost so the answer does not depend on iteration order.
 * - A cost the accuracy budget will not pay for, refused through @ref OptimizerPass::approximate
 *   with the budget's own reason.
 * - A tagged tensor no two-operand contraction reads at all. There is nothing to re-associate
 *   around, so the tag is reported unclaimed at the end of the run rather than passed over in
 *   silence: a tensor read only elementwise, or only through a dot, is the shape a caller
 *   tagging an integral in an energy expression most often writes.
 *
 * @par Never in a default manager
 * A provider may be exact, but the pass is the lossy tier's entry point and Part 5.1 says that
 * tier is opt-in. It runs when a caller adds it, and only for tags a provider is registered for,
 * so a process with no providers pays one lookup.
 *
 * @see Factorization.hpp for what a provider offers
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT FactorizationPass : public RegionRewrite {
  public:
    /// @brief Use the process-wide provider registry.
    APIARY_EXPOSE FactorizationPass() = default;

    /**
     * @brief Use a caller-supplied registry rather than the process-wide one.
     * @param[in] registry The registry to query. Must outlive the pass.
     */
    /// @note Keeps the registry alive for as long as the pass, because the pass holds a
    ///       pointer to it and a Python caller has no other reason to keep a reference.
    APIARY_EXPOSE APIARY_KEEP_ALIVE(1, 2) explicit FactorizationPass(FactorizationRegistry &registry) : _registry(&registry) {}

    /// @brief The pass name.
    /// @return ``"FactorizationPass"``.
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] std::string name() const override { return "FactorizationPass"; }

    /// @copydoc OptimizerPass::tier
    /// Substitutes a fitted factorization for the tensor it approximates, under a tolerance it records through
    /// OptimizerPass::approximate. Never eligible for a default manager, which is what this tier means.
    [[nodiscard]] PassTier tier() const override { return PassTier::Lossy; }

    /// @brief Zero the per-apply counters.
    void reset_stats() override;

    /// @brief Apply, then emit the setup bodies the accepted plans asked for.
    /// @param[in,out] graph The graph to rewrite.
    /// @return True when anything was rewritten.
    bool run(Graph &graph) override;

    /// @brief How many contractions were re-associated around a provider's factors.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("num_factorized") [[nodiscard]] std::size_t num_factorized() const { return _num_factorized; }

    /**
     * @brief How many captured intermediates the cone flattening dissolved into a chosen tree.
     *
     * Reported because it is the difference between re-associating one contraction and
     * re-associating the cone it sits in, and a caller reading the node count would otherwise
     * see a statement disappear with nothing saying which pass took it.
     *
     * @return The count.
     */
    APIARY_EXPOSE APIARY_GETTER("num_dissolved") [[nodiscard]] std::size_t num_dissolved() const { return _num_dissolved; }

  protected:
    /// @copydoc RegionRewrite::rewrite
    bool rewrite(Graph &graph, Region const &region, TensorExpr &expr) override;

    /// @brief Nothing to do unless some tensor carries a tag a provider claims.
    /// @param[in] graph The graph.
    /// @return True when at least one does.
    [[nodiscard]] bool applicable(Graph const &graph) const override;

    /// @brief The pass's own report lines.
    /// @return One line when anything was factorized.
    [[nodiscard]] std::vector<std::string> describe() const override;

  private:
    /// A setup body an accepted plan still needs emitted. Held until @ref run's region loop is
    /// over, because adding a node while a region is open moves the region out from under the
    /// splice that is about to replace it.
    struct PendingSetup {
        std::string                                                          label;
        std::vector<TensorId>                                                factors;
        std::function<void(Graph &, Graph &, std::vector<TensorId> const &)> emit;
    };

    /// The registry to query, or null for the process-wide one.
    FactorizationRegistry *_registry{nullptr};

    std::vector<PendingSetup> _pending;
    std::size_t               _num_factorized{0};
    std::size_t               _num_dissolved{0};

    /// Every tagged tensor some contraction offered this pass, whatever became of it. What the
    /// end-of-run sweep subtracts from the tagged tensors the graph holds, so a tag nothing
    /// ever offered is reported rather than silently doing nothing.
    std::vector<TensorId> _considered;

    /// @brief The registry this pass queries.
    /// @return The caller's, or the process-wide one.
    [[nodiscard]] FactorizationRegistry &registry() const;
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
