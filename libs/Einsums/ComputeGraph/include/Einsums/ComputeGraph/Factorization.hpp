//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file Factorization.hpp
 * @brief What a factorization provider offers, and where the offers are collected.
 *
 * @par The shape, and why it is a binary split
 * A provider says: this tagged tensor equals a product of two factors summed over a new
 * index. DF says an ERI is @c B[Q,m,n] @c B[Q,p,q] summed over an auxiliary @c Q; Cholesky
 * says the same with a different fitting; a low-rank compression says it with a smaller new
 * index. THC's three-factor form is a chain of the same operation and is not expressible
 * here yet, which is stated rather than left to be discovered.
 *
 * Binary because that is what LOWERS. @ref lower_region declines a contraction with other
 * than two operands, and its comment says exactly why: the node set has no multi-operand
 * contraction for one to lower to, so a pass emitting one has to pair it with a binary
 * decomposition. Making the plan binary at the point a provider states it means the
 * decomposition is arithmetic the pass does rather than a shape a provider can get wrong.
 *
 * @par Where the win comes from
 * The substitution is not the optimization; the RE-ASSOCIATION it enables is. Replacing the
 * ERI in @c C[..] @c = @c eri[m,n,p,q] @c T[p,q,..] with its factors and then contracting
 * them in the other order turns one expensive contraction into two cheap ones, and that
 * regrouping is the whole of what DF buys. A pass that substituted and stopped would make
 * the arithmetic strictly worse, which is why @ref passes::FactorizationPass costs the
 * decomposed form symbolically and declines when it does not come out ahead.
 *
 * @par What a provider is, and is not
 * A provider is a live object in a process, not data in a file. It holds the callback that
 * captures the fitting, which is a closure and unsaveable by the rule Part 3 sets. What IS
 * saved is the graph the provider produced: the setup body, the factor tensors, the
 * re-associated contractions, and the @ref ApproximationRecord saying what it cost. A loaded
 * graph therefore needs no provider registered to replay, which is the property that matters.
 *
 * @see Passes/FactorizationPass.hpp for the pass that consumes these
 * @see Approximation.hpp for the accuracy statement a provider has to make
 */

#include <Einsums/Config.hpp>

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/Approximation.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

/**
 * @brief One factor tensor a provider introduces, described in the provider's own alphabet.
 *
 * Letters are the provider's, not the graph's. A provider describing DF writes the ERI's
 * axes as whatever it called them in @ref FactorizationPlan::tagged_letters and the new
 * auxiliary axis as a letter of its own; the pass renames all of them onto the letters the
 * consuming contraction actually uses. That is what lets one provider serve every contraction
 * a tagged tensor appears in, whatever letters those happen to be spelled with.
 * @versionadded{2.0.0}
 */
struct FactorTensor {
    /// Name for the tensor the pass will create. Made unique if it collides.
    std::string name;

    /// Its index letters, in axis order, in the provider's alphabet.
    std::vector<std::string> letters;

    /// Extents, in letter order. A letter shared with the tagged tensor must agree with that
    /// tensor's extent on the axis it names, which the pass checks rather than assumes.
    std::vector<std::size_t> dims;

    /// Index-space names, in letter order. An empty string is an unannotated axis, which is
    /// legal and yields a weaker cost comparison rather than no comparison.
    std::vector<std::string> spaces;

    /// Element type of the factor.
    packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Float64};
};

/**
 * @brief A provider's offer for one tagged tensor.
 *
 * @see FactorizationProvider::propose
 * @versionadded{2.0.0}
 */
struct FactorizationPlan {
    /// Who proposed it, for the report and for @ref ApproximationRecord::pass_name.
    std::string provider;

    /// The tagged tensor's axes, in order, named in the provider's alphabet. Its length must
    /// equal the tensor's rank.
    std::vector<std::string> tagged_letters;

    /// Exactly two factors whose letters, taken together, are @ref tagged_letters plus the
    /// new letters this provider introduces.
    std::vector<FactorTensor> factors;

    /// A scalar on the product, for a provider whose identity carries one.
    PrefactorScalar factor{double{1}};

    /// What the substitution costs in accuracy. @ref ApproximationRecord::pass_name is filled
    /// in by the pass; everything else is the provider's statement. A bound of exactly zero
    /// is a provider claiming an EXACT factorization, and is recorded like any other so that
    /// "this result is a Cholesky decomposition of the integrals" survives a save.
    ApproximationRecord accuracy;

    /// Capture the fitting into a setup body.
    ///
    /// Called with the PARENT graph, the body to capture into, and the ids of the created
    /// factors in @ref factors order. Under no capture guard: the callback opens its own,
    /// exactly as any other code that captures does. Whatever it records runs once per bound
    /// problem and is skipped by every replay after, which is the whole reason a factorization
    /// is affordable at all.
    ///
    /// The parent is passed because the factors are ITS tensors and a plan holds only their
    /// ids; capture takes tensor references, so a callback writing into a factor has to reach
    /// the object through the parent's handle for it, which carries the address. Handing over
    /// the id alone would make the one thing every provider must do the one thing it cannot.
    ///
    /// The expression is spelled in prose rather than as a call, because a cross-reference or
    /// an inline literal followed immediately by a parameter list renders as a literal whose
    /// start-string is never terminated, and the docs build treats that warning as an error.
    std::function<void(Graph &parent, Graph &body, std::vector<TensorId> const &factors)> emit_setup;
};

/**
 * @brief A way of rewriting one kind of tensor as a product of factors.
 *
 * Register one instance per method. The pass asks every provider claiming a tensor's
 * provenance tag, costs each offer, and takes the cheapest that is actually cheaper.
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT FactorizationProvider {
  public:
    FactorizationProvider()                                         = default;
    FactorizationProvider(FactorizationProvider const &)            = delete;
    FactorizationProvider &operator=(FactorizationProvider const &) = delete;
    FactorizationProvider(FactorizationProvider &&)                 = delete;
    FactorizationProvider &operator=(FactorizationProvider &&)      = delete;
    virtual ~FactorizationProvider()                                = default;

    /// @brief This provider's name, used in reports and in the approximation record.
    /// @return The name.
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] virtual std::string name() const = 0;

    /// @brief The provenance tag whose tensors this provider factorizes.
    /// @return The tag name, e.g. ``"eri"``.
    APIARY_EXPOSE APIARY_GETTER("tag") [[nodiscard]] virtual std::string tag() const = 0;

    /**
     * @brief Offer a factorization of @p tensor, or say why there is not one.
     *
     * @param[in] graph The graph, for the tensor's extents, spaces and tag attributes.
     * @param[in] tensor The tagged tensor.
     * @return The plan, or a short reason phrased for a skip tally.
     *
     * Called at optimize time, once per tagged tensor. A provider that needs a threshold the
     * tag does not carry reads it from its own state, which is why a provider is an object
     * rather than a function.
     */
    [[nodiscard]] virtual expected<FactorizationPlan, std::string> propose(Graph const &graph, TensorId tensor) const = 0;
};

/**
 * @brief The providers a process knows about.
 *
 * Process-global by default, in the manner of @ref SpaceRegistry, because a provider is a
 * capability of the build rather than of one graph. Registration is a startup act; a test
 * that adds one is expected to remove it, and @ref clear exists for exactly that.
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") EINSUMS_EXPORT FactorizationRegistry {
  public:
    APIARY_EXPOSE FactorizationRegistry() = default;

    /**
     * @brief Register a provider.
     * @param[in] provider The provider. Held by shared_ptr and must outlive nothing in
     *            particular: the registry owns a reference.
     * @throws std::invalid_argument When a provider of the same name is already registered,
     *         naming it. Two providers under one name would make which of them ran depend on
     *         registration order, and a pass whose result depends on that is not reproducible.
     */
    APIARY_EXPOSE void add(std::shared_ptr<FactorizationProvider> provider);

    /**
     * @brief Every provider claiming @p tag, in registration order.
     * @param[in] tag A provenance tag name.
     * @return The providers. Empty when nothing claims it.
     */
    APIARY_EXPOSE [[nodiscard]] std::vector<std::shared_ptr<FactorizationProvider>> for_tag(std::string_view tag) const;

    /// @brief Whether any provider claims @p tag.
    /// @param[in] tag A provenance tag name.
    /// @return True when at least one does.
    APIARY_EXPOSE [[nodiscard]] bool claims(std::string_view tag) const;

    /// @brief Remove a provider by name.
    /// @param[in] name The provider's @ref FactorizationProvider::name.
    /// @return True when one was removed.
    APIARY_EXPOSE bool remove(std::string_view name);

    /// @brief Forget every provider.
    APIARY_EXPOSE void clear();

    /// @brief How many providers are registered.
    /// @return The count.
    APIARY_EXPOSE APIARY_GETTER("size") [[nodiscard]] std::size_t size() const;

  private:
    std::vector<std::shared_ptr<FactorizationProvider>> _providers;
};

/**
 * @brief The process-wide provider registry.
 * @return The registry. Never null.
 * @versionadded{2.0.0}
 */
[[nodiscard]] APIARY_EXPOSE APIARY_MODULE("graph") APIARY_RVP(reference)
    EINSUMS_EXPORT FactorizationRegistry &global_factorization_registry();

EINSUMS_NAMESPACE_END(compute_graph)
