//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file ThcFactorization.hpp
 * @brief The second factorization provider: a four-index tensor as a five-factor grid chain.
 *
 * @par What it is, in the vocabulary of the layer it lives in
 * Given a collocation matrix @c X[m,P], the values of the @c m basis functions at the @c P grid
 * points, and a three-index tensor @c B[A,m,n] over the same basis, this offers
 *
 * @f[ T[m,n,p,q] \approx \sum_{PQ} X[m,P]\,X[n,P]\,Z[P,Q]\,X[p,Q]\,X[q,Q] @f]
 *
 * with @c Z fitted by least squares from @c B. Five factors over two new letters, which is what
 * a plan naming a factor LIST exists for: it cannot be written as a split of two.
 *
 * @par What chemistry calls it
 * Tensor hypercontraction, and this class registered on the tag @c "eri" is the ``AutoTHC`` the
 * design asks for. The name stays generic for the reason @ref MetricFitFactorization's does:
 * nothing in the class knows what an integral is, and the chemistry lives in the registration.
 * The two are ALTERNATIVES on one tag rather than a composition, and the registry's ranking by
 * profitability is what chooses between them per tagged tensor.
 *
 * @par The fit, and why it is fitted FROM the density fit
 * The least-squares solution of @f$B[A,m,n] \approx \sum_P C[A,P] X[m,P] X[n,P]@f$ is
 * @f$C = \tilde{B} S^{-1}@f$ with
 *
 * @f[ S[P,Q] = \left(\sum_m X[m,P] X[m,Q]\right)^2, \qquad
 *     \tilde{B}[A,P] = \sum_{mn} B[A,m,n] X[m,P] X[n,P] @f]
 *
 * the square being ELEMENTWISE, and the four-index fit follows as
 * @f$Z = S^{-1} \tilde{B}^{T} \tilde{B} S^{-1}@f$. Fitting from the three-index tensor rather
 * than from the exact integrals is what keeps this validatable offline against the same fixture
 * @ref MetricFitFactorization was: the exact four-index quantity comes from an integral engine
 * this library does not have.
 *
 * @c S is the Hadamard square of a Gram matrix and therefore positive semi-definite, so its
 * inverse goes through the same guarded route the metric fit uses, twice: @c inv_sqrt_or_zero
 * over the eigenvalues gives @f$S^{-1/2}@f$ and the square of that is @f$S^{-1}@f$. The drop
 * threshold is the caller's for the reason it is there, and reaches the node rather than the
 * process.
 *
 * @par The grid is a space, and its extent is symbolic
 * The factors carry an index space named by @ref grid_space_name, whose @c dim_symbol is
 * @ref grid_dim_symbol. That is what makes a saved THC graph replayed at a new geometry a
 * REBIND rather than a re-capture, and it is what makes the factorization pass's numeric veto
 * abstain: a grid is chosen per problem, so the number of points a capture happened to have is
 * a placeholder rather than a size anything runs at. Register the space with
 * @ref register_grid_space before applying the pass; a graph whose registry does not hold it
 * gets factors with no space and a weaker cost comparison rather than a wrong one.
 *
 * @par The accuracy statement
 * The bound is asserted, from @c einsums:graph:thc-epsilon or from the constructor, because an
 * @ref ApproximationRecord is written once at optimize time. What is MEASURED, per bind and
 * exactly as @ref MetricFitFactorization measures its dropped directions, is the least-squares
 * residual of the fit against the three-index tensor it was fitted from: two parameters, named
 * by @ref residual_param_name and @ref reference_param_name, whose ratio's square root is the
 * relative residual. The four-index error is not measurable without the tensor the fit exists
 * to avoid forming; this one is, and it costs one contraction more than the fit already does.
 *
 * @see Factorization.hpp for what a provider is
 * @see MetricFitFactorization.hpp for the provider this is an alternative to
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <string>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class SpaceRegistry;

/**
 * @brief Offer a four-index tensor as a fitted chain over two grid indices.
 *
 * @see ThcFactorization.hpp for the mathematics and for what chemistry calls it
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ThcFactorization : public FactorizationProvider {
  public:
    /// @brief The drop threshold a caller who states none gets.
    ///
    /// The same number @ref MetricFitFactorization uses and for the same reason: it is far
    /// below anything a well-conditioned grid produces and far above the noise a redundant one
    /// leaves behind.
    static constexpr double default_drop_threshold = 1.0e-10;

    /**
     * @brief Construct the provider.
     *
     * @param[in] tag The provenance tag this claims, e.g. ``"eri"``.
     * @param[in] three_index @c B[A,m,n], the density-fitted three-index tensor the grid fit is
     *            fitted FROM. Must outlive the provider and every graph it factorizes.
     * @param[in] collocation @c X[m,P], the basis functions' values at the grid points. Must
     *            outlive the same, and becomes an interface tensor of the transformed graph.
     * @param[in] bound The relative error the caller asserts this fit has, or zero to take
     *            ``einsums:graph:thc-epsilon``.
     * @param[in] drop_threshold Grid directions whose eigenvalue of @c S does not exceed this
     *            are dropped from the fit; see @ref default_drop_threshold.
     * @param[in] name The provider's name, which the approximation record and the report carry.
     * @throws std::invalid_argument When @p drop_threshold is negative or not finite, or when
     *         @p bound is negative. A negative threshold reads as "keep more than everything";
     *         the kernel floors it at zero anyway, and running at a threshold the caller did
     *         not ask for is the failure the parameter exists to prevent.
     */
    ThcFactorization(std::string tag, RuntimeTensorView<double> three_index, RuntimeTensorView<double> collocation, double bound = 0.0,
                     double drop_threshold = default_drop_threshold, std::string name = "Thc");

    /// @brief The same, taking runtime-rank tensors. This is the overload Python gets.
    ///
    /// @note The drop-threshold default is spelled with its class qualifier because the binding
    ///       generator copies a default argument through verbatim, and an unqualified
    ///       class-scope constant does not resolve where the generated code spells it.
    APIARY_EXPOSE ThcFactorization(std::string tag, RuntimeTensor<double> const &three_index, RuntimeTensor<double> const &collocation,
                                   double bound = 0.0, double drop_threshold = ThcFactorization::default_drop_threshold,
                                   std::string name = "Thc");

    /// @brief The same, taking compile-time-rank tensors.
    ///
    /// Kept as its own overload rather than left to the implicit conversion, because that
    /// conversion does not carry the NAME: a view built from a ``Tensor<double, N>`` is
    /// anonymous, the fitting captures it as an interface tensor, and a manifest binds by name.
    ThcFactorization(std::string tag, Tensor<double, 3> const &three_index, Tensor<double, 2> const &collocation, double bound = 0.0,
                     double drop_threshold = default_drop_threshold, std::string name = "Thc");

    /// @copydoc FactorizationProvider::name
    [[nodiscard]] std::string name() const override { return _name; }

    /// @copydoc FactorizationProvider::tag
    [[nodiscard]] std::string tag() const override { return _tag; }

    /**
     * @brief Offer the grid chain, or say why this tensor is not one.
     *
     * @param[in] graph The graph holding @p tensor.
     * @param[in] tensor The tagged tensor.
     * @return The plan, or the reason there is not one: a rank other than four, or extents that
     *         do not match the collocation matrix's basis axis.
     */
    [[nodiscard]] expected<FactorizationPlan, std::string> propose(Graph const &graph, TensorId tensor) const override;

    /// @brief The tolerance this provider asserts, taking the constructor over the option.
    /// @return The relative bound.
    APIARY_EXPOSE APIARY_GETTER("epsilon") [[nodiscard]] double epsilon() const;

    /// @brief The threshold below which this provider drops a grid direction.
    /// @return The number handed to the constructor, or @ref default_drop_threshold.
    APIARY_EXPOSE APIARY_GETTER("drop_threshold") [[nodiscard]] double drop_threshold() const noexcept { return _drop_threshold; }

    /// @brief The index space the fitted factors' grid axes carry.
    /// @return ``"grid"``.
    APIARY_EXPOSE [[nodiscard]] static std::string grid_space_name();

    /// @brief The symbolic extent a grid axis goes by.
    /// @return ``"ngrid"``.
    APIARY_EXPOSE [[nodiscard]] static std::string grid_dim_symbol();

    /**
     * @brief Register the grid space in @p graph's own registry, if it is not there already.
     *
     * A space is a statement about a graph rather than about a provider, and @ref propose is
     * handed a const graph precisely so a provider cannot make one. So a caller registers it,
     * and this is the one call that does: idempotent for a repeated identical declaration, as
     * @ref SpaceRegistry::register_space is.
     *
     * @param[in,out] graph The graph whose registry gains the space.
     * @return The space's id in that registry.
     */
    APIARY_EXPOSE static SpaceId register_grid_space(Graph &graph);

    /**
     * @brief The parameter the squared residual of the fit is reported under.
     *
     * @param[in] provider The provider's @ref name.
     * @param[in] tensor The tagged tensor's name.
     * @return The key to read out of the graph's parameter table after an execute.
     *
     * Divided by @ref reference_param_name and square-rooted, this is the RELATIVE least-squares
     * residual of the grid fit against the three-index tensor it was fitted from. Per bind and
     * not saved, because it is a property of the grid and the integrals currently bound rather
     * than of the structure.
     */
    APIARY_EXPOSE [[nodiscard]] static std::string residual_param_name(std::string const &provider, std::string const &tensor);

    /// @brief The parameter the squared norm of the three-index tensor is reported under.
    /// @param[in] provider The provider's @ref name.
    /// @param[in] tensor The tagged tensor's name.
    /// @return The key. See @ref residual_param_name for what the two are for.
    APIARY_EXPOSE [[nodiscard]] static std::string reference_param_name(std::string const &provider, std::string const &tensor);

  private:
    std::string               _tag;
    std::string               _name;
    RuntimeTensorView<double> _three_index;
    RuntimeTensorView<double> _collocation;
    double                    _bound;
    double                    _drop_threshold;
};

EINSUMS_NAMESPACE_END(compute_graph)
