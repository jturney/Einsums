//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file LaplaceQuadrature.hpp
 * @brief The numerical quadrature behind @ref passes::LaplaceTransform, and the node that refits it.
 *
 * @par The identity
 * For a positive @f$x@f$,
 * @f[ \frac{1}{x} = \int_0^\infty e^{-xt}\,\mathrm{d}t, @f]
 * so a reciprocal of a SUM of per-axis quantities factorizes: with
 * @f$x = \sum_k \sigma_k \varepsilon_k[i_k]@f$ and @f$\sigma_k = \pm 1@f$,
 * @f[ \frac{1}{x} \approx \sum_j w_j \prod_k e^{-\sigma_k t_j \varepsilon_k[i_k]} . @f]
 * Every quadrature point is a product of one vector per axis, which is what lets a consumer
 * that couples all the axes at once be rewritten as a sum of decoupled terms.
 *
 * @par The rule, and why this one
 * The substitution @f$t = e^{s}@f$ turns the integral into
 * @f[ \frac{1}{x} = \int_{-\infty}^{\infty} \exp\!\left(s - x e^{s}\right)\mathrm{d}s, @f]
 * and the rule is the TRUNCATED TRAPEZOIDAL SUM of that integrand on a uniform grid in
 * @f$s@f$. Three properties earn it its place here over a Gauss rule or a minimax fit:
 *
 *  - Its points and weights are CLOSED FORM in the spectral range, so a bind refits them with
 *    arithmetic rather than with a Remez iteration. A rule that needed an iterative solve
 *    could not be a node, and a rule that could not be a node would have to be computed once
 *    at optimize time and would then be wrong at every later bind.
 *  - Its error is bounded ANALYTICALLY and uniformly over the whole interval, see below.
 *  - It is the classical exponential-grid Laplace transform of the electronic-structure
 *    literature rather than something invented here.
 *
 * A minimax rule needs roughly a third of the points for the same accuracy and is the natural
 * successor; it is not the first version, because it is an iterative fit and this one is a
 * formula. What the class buys by starting here is that a saved graph refits.
 *
 * @par The error bound
 * Write @f$x = x_{\min} p@f$ with @f$p \in [1, R]@f$ and @f$R = x_{\max}/x_{\min}@f$. The
 * three error terms of a truncated trapezoidal sum with step @f$h@f$ on @f$[s_0, s_1]@f$ are
 * bounded, RELATIVE to @f$1/x@f$, by
 *
 *  - discretization: the Poisson summation of the trapezoid rule gives
 *    @f$2\,|\Gamma(1 + 2\pi i/h)| \le (4\pi/\sqrt{h})\,e^{-\pi^2/h}@f$, which is independent
 *    of @f$p@f$ and is the whole reason the rule is uniform over the interval;
 *  - left truncation: @f$p\,e^{s_0} \le R\,e^{s_0}@f$;
 *  - right truncation: @f$\exp(-e^{s_1})@f$.
 *
 * Given a target @f$\varepsilon@f$, the three are given @f$\varepsilon/2@f$,
 * @f$\varepsilon/4@f$ and @f$\varepsilon/4@f$, which fixes @f$s_0@f$, @f$s_1@f$ and @f$h@f$,
 * and the point count follows. The bound is not the number that gets recorded: the rule is
 * cheap to evaluate, so the setup MEASURES its worst relative deviation over the interval and
 * records that instead. The analytic bound is what chooses the point count; the measurement
 * is what the approximation record says.
 *
 * @par What the node does
 * @ref OpKind::LaplaceQuadrature reads the orbital-energy vectors, computes the spectral range
 * of the signed sum they form, builds the rule at the point count the descriptor fixes, and
 * writes the points, the weights, the measured error and one exponential matrix per axis.
 * A node rather than a captured chain of element operations because the range is a REDUCTION
 * over the bound data and the grid is a transcendental function of it, neither of which the
 * node set can express; and a node rather than a closure because a closure cannot be saved.
 *
 * @see Passes/LaplaceTransform.hpp for the rewrite this feeds
 */

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::laplace)

/// @brief The smallest and largest value a signed sum of per-axis vectors can take.
/// @versionadded{2.0.0}
struct SpectralRange {
    double low{0};  ///< The smallest value of @f$\sum_k \sigma_k \varepsilon_k[i_k]@f$.
    double high{0}; ///< The largest.
};

/**
 * @brief The range of @f$\sum_k \sigma_k \varepsilon_k[i_k]@f$ from each axis's own range.
 *
 * @param[in] axis_low  Per axis, the smallest element of that axis's energy vector.
 * @param[in] axis_high Per axis, the largest.
 * @param[in] signs     Per axis, @c +1 or @c -1.
 * @return The range of the signed sum.
 * @throws std::invalid_argument When the three lists differ in length or are empty.
 *
 * Exact rather than conservative, because the axes are INDEPENDENT: every combination of one
 * index per axis occurs, so the extremes of the sum are the sums of the extremes.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT SpectralRange spectral_range(std::vector<double> const &axis_low, std::vector<double> const &axis_high,
                                                          std::vector<int> const &signs);

/// @brief One quadrature rule for @f$1/x@f$ on a bounded positive interval.
/// @versionadded{2.0.0}
struct QuadratureGrid {
    std::vector<double> points;    ///< @f$t_j@f$, the exponents.
    std::vector<double> weights;   ///< @f$w_j@f$.
    double              x_low{0};  ///< The interval this rule was built for.
    double              x_high{0}; ///< The other end of it.

    /// Whether the signed sum was uniformly NEGATIVE, so the rule was built for its negation.
    ///
    /// A denominator of orbital energies is negative as often as it is positive, depending
    /// only on which way round the caller wrote the difference, and refusing one of the two
    /// would be an arbitrary convention for a caller to trip over. The rule is built for
    /// @f$|x|@f$ and the weights carry the sign, so the caller's convention survives.
    bool negated{false};
};

/**
 * @brief How many points a target accuracy needs over a given range.
 *
 * @param[in] range   The spectral range of the signed sum. Must not contain zero.
 * @param[in] epsilon The target RELATIVE accuracy. Must be positive and below one.
 * @return The point count, at least two.
 * @throws std::invalid_argument When @p range straddles or touches zero, or @p epsilon is not
 *         a usable tolerance.
 *
 * The count is what a rewrite bakes into structure, so it is derived once, at optimize time,
 * from the range the capture-bound energies have. A later bind refits the rule AT THIS COUNT
 * over whatever range it then finds, which is the honest trade: a node set whose size depended
 * on run-time data would not be a node set, and a rule refitted at a fixed count over a wider
 * range reports the accuracy it actually reaches through its measured error.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT std::size_t quadrature_point_count(SpectralRange const &range, double epsilon);

/**
 * @brief Build the rule at a fixed point count.
 *
 * @param[in] range   The spectral range of the signed sum. Must not contain zero.
 * @param[in] epsilon The target relative accuracy, which fixes where the grid is truncated.
 * @param[in] points  How many points to use. Must be at least two.
 * @return The rule.
 * @throws std::invalid_argument When @p range straddles or touches zero, when @p epsilon is
 *         not a usable tolerance, or when @p points is below two.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT QuadratureGrid build_quadrature(SpectralRange const &range, double epsilon, std::size_t points);

/**
 * @brief The worst relative deviation of @p grid from @f$1/x@f$ over the interval it covers.
 *
 * @param[in] grid    The rule to measure.
 * @param[in] samples How many points to sample, logarithmically spaced. At least two.
 * @return The largest @f$|q(x) - 1/x| \cdot |x|@f$ found.
 *
 * A measurement rather than the analytic bound, and affordable precisely because the object
 * being approximated is a scalar function: the four-index tensor a factorization would have to
 * form to measure itself does not appear anywhere in this.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT double measure_quadrature_error(QuadratureGrid const &grid, std::size_t samples = 1024);

/**
 * @brief Capture an @ref OpKind::LaplaceQuadrature node.
 *
 * @param[in]  energies      The per-axis orbital-energy vectors, in axis order. Their storage
 *             must outlive every replay, since the node reads it on every bind.
 * @param[out] points        Receives @f$t_j@f$. Extent must be the descriptor's point count.
 * @param[out] weights       Receives @f$w_j@f$, sign included.
 * @param[out] error         Receives the measured relative error, as a one-element tensor.
 * @param[out] exponentials  Per axis, receives @f$E_k[j, i] = w_j^{[k=0]} e^{-\sigma_k t_j \varepsilon_k[i]}@f$.
 * @param[in]  descriptor    The tolerance, the point count and the per-axis signs.
 * @throws std::logic_error When called outside a capture.
 * @throws std::invalid_argument When the operand lists disagree with the descriptor.
 *
 * The weights ride on the FIRST axis's exponential matrix rather than being applied by a node
 * of their own. One multiply per quadrature point either way, and the consumer then needs no
 * statement whose only job is a scalar.
 * @versionadded{2.0.0}
 */
template <typename EnergyType, typename OutputType>
void laplace_quadrature(std::vector<EnergyType *> const &energies, OutputType *points, OutputType *weights, OutputType *error,
                        std::vector<OutputType *> const &exponentials, LaplaceQuadratureDescriptor descriptor) {
    using T = typename OutputType::ValueType;

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::laplace_quadrature is a captured node and has no eager form");
    }
    if (energies.size() != exponentials.size() || energies.size() != descriptor.signs.size() || energies.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "cg::laplace_quadrature: {} energy vector(s), {} exponential matrices and {} signs; the three lists "
                                "describe one axis each and must agree",
                                energies.size(), exponentials.size(), descriptor.signs.size());
    }

    std::vector<TensorId> inputs;
    inputs.reserve(energies.size());
    for (auto *energy : energies) {
        auto [id, slot] = ctx.get_slot(*energy);
        (void)slot;
        inputs.push_back(id);
    }

    // Points, weights and the measured error come first and the exponentials follow, which is
    // the order the builder reads them back in. Positional rather than named because a node's
    // operand lists are positional everywhere else in this module.
    std::vector<TensorId> outputs;
    outputs.reserve(exponentials.size() + 3);
    for (OutputType *target : {points, weights, error}) {
        auto [id, slot] = ctx.get_slot(*target);
        (void)slot;
        outputs.push_back(id);
    }
    for (auto *matrix : exponentials) {
        auto [id, slot] = ctx.get_slot(*matrix);
        (void)slot;
        outputs.push_back(id);
    }

    OpData op_data(std::move(descriptor));
    auto executor = build_executor(OpKind::LaplaceQuadrature, packed_gemm::get_scalar_type<T>(), 2, op_data, *ctx.graph(), inputs, outputs);

    ctx.record(OpKind::LaplaceQuadrature, "laplace_quadrature", inputs, outputs, std::move(executor), std::move(op_data));
}

EINSUMS_NAMESPACE_END(compute_graph::laplace)
