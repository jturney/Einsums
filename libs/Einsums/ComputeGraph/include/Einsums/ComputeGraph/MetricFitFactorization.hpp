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
 * @par The drop threshold, and why it is ABSOLUTE
 * Which directions are thrown away is a policy, so it is a constructor parameter with a
 * conventional default rather than a rule baked into the kernel. A guard testing @c x>0
 * catches the eigenvalue that has already gone negative and misses the one that has not
 * quite: at @c +1e-17 it passes, and @f$1/\sqrt{x}@f$ of it is about @c 3e8, which is worse
 * than an infinity because nothing downstream looks wrong and the dropped-directions count
 * reports zero. Every orthogonalization in practice drops below a threshold instead.
 *
 * The threshold is compared against the eigenvalue itself, not against the largest
 * eigenvalue, and the difference matters for a caller whose metric is scaled: the
 * conventional cutoff in quantum chemistry is RELATIVE, @f$\lambda < \tau\lambda_{max}@f$,
 * which is invariant to scaling the metric and this is not. What a relative one needs and
 * this does not have is a reduction over the eigenvalues feeding a value into the guard,
 * which is a runtime quantity, while a captured element op binds its policy number when the
 * executor is built. That is a mechanism to add rather than a number to change, and until
 * then a caller whose metric is scaled far from unity is expected to scale the threshold
 * with it; @ref default_drop_threshold documents what the default assumes.
 *
 * @see Factorization.hpp for what a provider is
 * @see Passes/FactorizationPass.hpp for what is done with the offer
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Factorization.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
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
     * @brief The drop threshold a caller who states none gets.
     *
     * An eigenvalue of a Coulomb-like metric in atomic units is @c O(1e-3) to @c O(1e3), so
     * @c 1e-10 is far below anything a well-conditioned auxiliary set produces and far above
     * the noise a linearly dependent one leaves behind. It is the magnitude the conventional
     * relative cutoff uses, applied absolutely; see the file note on the difference.
     */
    static constexpr double default_drop_threshold = 1.0e-10;

    /**
     * @brief Construct the provider.
     *
     * @param[in] tag The provenance tag this claims, e.g. ``"eri"``.
     * @param[in] three_index @c R[P,m,n]. Must outlive the provider and every graph it
     *            factorizes, because the fitting reads it on every bind.
     * @param[in] metric @c J[P,Q], symmetric and positive semi-definite. Must outlive the same.
     * @param[in] bound The relative error the caller asserts this fit has. Required, because
     *            nothing here can measure it; see the file note.
     * @param[in] drop_threshold Auxiliary directions whose metric eigenvalue does not exceed
     *            this are dropped from the fit and counted; see @ref default_drop_threshold
     *            and the file note on why the comparison is absolute. Zero reproduces the
     *            bare @c x>0 guard.
     * @param[in] name The provider's name, which the approximation record and the report carry.
     * @throws std::invalid_argument When @p drop_threshold is negative or not finite. A
     *         negative one reads as "keep more than everything" and would mean nothing; the
     *         kernel floors it at zero anyway, and silently running at a threshold the caller
     *         did not ask for is the failure this whole parameter exists to prevent.
     */
    /// @note The two tensors are taken as @ref RuntimeTensorView, which any dense rank-3 and
    ///       rank-2 double tensor converts to, ``Tensor<double, N>`` and
    ///       ``RuntimeTensor<double>`` alike. That is what lets a Python caller register this
    ///       provider at all, since the bindings carry runtime-rank tensors and nothing else.
    ///       A view also holds a reference to the storage it names, where the raw pointers
    ///       this used to keep did not.
    MetricFitFactorization(std::string tag, RuntimeTensorView<double> three_index, RuntimeTensorView<double> metric, double bound,
                           double drop_threshold = default_drop_threshold, std::string name = "MetricFit");

    /// @brief The same, taking compile-time-rank tensors.
    ///
    /// Kept as its own overload rather than left to the implicit conversion, because that
    /// conversion does not carry the NAME: a view built from a ``Tensor<double, N>`` is
    /// anonymous, the fitting captures it as an interface tensor, and a manifest binds by
    /// name, so two anonymous operands collide and the graph stops being saveable. This
    /// overload names the views after the tensors it was handed.
    MetricFitFactorization(std::string tag, Tensor<double, 3> const &three_index, Tensor<double, 2> const &metric, double bound,
                           double drop_threshold = default_drop_threshold, std::string name = "MetricFit");

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
     * @brief The threshold below which this provider drops an auxiliary direction.
     * @return The number handed to the constructor, or @ref default_drop_threshold.
     */
    [[nodiscard]] double drop_threshold() const noexcept { return _drop_threshold; }

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
    std::string               _tag;
    std::string               _name;
    RuntimeTensorView<double> _three_index;
    RuntimeTensorView<double> _metric;
    double                    _bound;
    double                    _drop_threshold;
};

EINSUMS_NAMESPACE_END(compute_graph)
