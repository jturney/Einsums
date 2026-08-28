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
 * @par The bound is MEASURED, not asserted
 * The fitting error depends on the auxiliary set and no formula the library has predicts it.
 * But the exact tensor is in hand at the moment the pass asks, so this forms the fitted product
 * once and reports the relative Frobenius error it actually has. That is a statement about the
 * problem it was fitted on, which is exactly what an @ref ApproximationRecord is, and it is a
 * far better number than one a caller guessed. A caller who would rather assert a bound than
 * pay for measuring one can supply it.
 *
 * @see Factorization.hpp for what a provider is
 * @see Passes/FactorizationPass.hpp for what is done with the offer
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <optional>
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
     * @param[in] declared_bound A relative bound to assert instead of measuring one, or an
     *            empty optional to measure. Measuring forms the fitted product once, which is
     *            the same order of work as one contraction against the tensor being replaced.
     * @param[in] name The provider's name, which the approximation record and the report carry.
     */
    MetricFitFactorization(std::string tag, Tensor<double, 3> const &three_index, Tensor<double, 2> const &metric,
                           std::optional<double> declared_bound = std::nullopt, std::string name = "MetricFit");

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

    /// @brief The relative Frobenius error the last @ref propose measured, or a negative number
    ///        when it asserted a bound instead.
    /// @return The error.
    [[nodiscard]] double measured_error() const { return _measured; }

  private:
    std::string              _tag;
    std::string              _name;
    Tensor<double, 3> const *_three_index;
    Tensor<double, 2> const *_metric;
    std::optional<double>    _declared;
    mutable double           _measured{-1.0};
};

EINSUMS_NAMESPACE_END(compute_graph)
