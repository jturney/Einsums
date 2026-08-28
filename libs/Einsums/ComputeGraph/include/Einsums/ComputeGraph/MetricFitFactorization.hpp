//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file MetricFitFactorization.hpp
 * @brief The first real factorization provider: a four-index tensor as a fitted product.
 *
 * @par What it is, in the vocabulary of the layer it lives in
 * Given a three-index tensor @c R[P,m,n] and a symmetric positive-definite metric @c J[P,Q]
 * over the same first index, this offers
 *
 * @f[ T[m,n,p,q] \approx \sum_Q B[Q,m,n]\, B[Q,p,q], \qquad B = J^{-1/2} R @f]
 *
 * which is exact when @c T is the metric-fitted product of @c R with itself and approximate
 * otherwise. Nothing here knows what @c T means; a caller says which provenance tag it claims,
 * and everything else is linear algebra.
 *
 * @par What chemistry calls it
 * Density fitting, and this class registered on the tag @c "eri" is the ``AutoDF`` the design
 * asks for. The name stays generic because core does not learn what an orbital is: the
 * auxiliary index, the metric and the tag are all the caller's, and the same class serves any
 * factorization of that shape.
 *
 * @par The bound is ASSERTED, and it has to be
 * The fitting error is the difference between the fitted product and the exact tensor, and a
 * caller who has the exact tensor in memory did not need to fit anything. So there is nothing
 * to measure against: from the three-index tensor and the metric the library can compute the
 * FITTED product and nothing else, because the exact four-index quantity comes from an integral
 * engine this library does not have. The bound is the caller's statement, recorded as one.
 *
 * What CAN be measured is reported instead, and per bind rather than once: how many auxiliary
 * directions the fit threw away. A metric of a nearly linearly dependent auxiliary set has
 * eigenvalues at or below zero, the guarded inverse square root zeroes them, and the fitted
 * space is then smaller than the auxiliary set the caller supplied. That is a specific and
 * common way for accuracy to degrade quietly, it costs nothing extra because the eigenvalues
 * are computed anyway, and it is a real number rather than a restatement of an assertion.
 *
 * @see Factorization.hpp for what a provider is
 * @see Passes/FactorizationPass.hpp for what is done with the offer
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <string>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief Offer a four-index tensor as a metric-fitted product of one three-index factor.
 *
 * @see MetricFitFactorization.hpp for the mathematics and for what chemistry calls it
 * @versionadded{2.0.0}
 */
class EINSUMS_EXPORT MetricFitFactorization : public FactorizationProvider {
  public:
    /**
     * @brief Construct the provider.
     *
     * @param[in] tag The provenance tag this claims, e.g. ``"eri"``.
     * @param[in] three_index @c R[P,m,n]. Must outlive the provider and every graph it
     *            factorizes, because the fitting reads it on every bind.
     * @param[in] metric @c J[P,Q], symmetric and positive semi-definite. Must outlive the same.
     * @param[in] bound The relative error the caller asserts this fit has. Required, because
     *            nothing here can measure it; see the file note.
     * @param[in] name The provider's name, which the approximation record and the report carry.
     */
    MetricFitFactorization(std::string tag, Tensor<double, 3> const &three_index, Tensor<double, 2> const &metric, double bound,
                           std::string name = "MetricFit");

    /// @copydoc FactorizationProvider::name
    [[nodiscard]] std::string name() const override { return _name; }

    /// @copydoc FactorizationProvider::tag
    [[nodiscard]] std::string tag() const override { return _tag; }

    /**
     * @brief Offer the fitted split, or say why this tensor is not one.
     *
     * @param[in] graph The graph holding @p tensor.
     * @param[in] tensor The tagged tensor.
     * @return The plan, or the reason there is not one: a rank other than four, extents that
     *         do not match the three-index tensor's, or a metric that is not square over the
     *         auxiliary index.
     */
    [[nodiscard]] expected<FactorizationPlan, std::string> propose(Graph const &graph, TensorId tensor) const override;

    /**
     * @brief The parameter a fitted graph reports its dropped auxiliary directions under.
     *
     * @param[in] provider The provider's @ref name.
     * @param[in] tensor The tagged tensor's name.
     * @return The key to read out of the graph's parameter table after an execute.
     *
     * A function rather than a documented convention, so a caller reads the name from the same
     * place the graph writes it. Zero means the metric was well conditioned and the fit kept
     * every direction the caller supplied; anything else is accuracy lost in a way the asserted
     * bound does not describe.
     */
    [[nodiscard]] static std::string dropped_param_name(std::string const &provider, std::string const &tensor);

  private:
    std::string              _tag;
    std::string              _name;
    Tensor<double, 3> const *_three_index;
    Tensor<double, 2> const *_metric;
    double                   _bound;
};

EINSUMS_NAMESPACE_END(compute_graph)
