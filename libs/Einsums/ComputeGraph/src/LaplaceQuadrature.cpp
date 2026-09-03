//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/LaplaceQuadrature.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::laplace)

namespace {

/// The relative discretization error of the trapezoidal rule at step @p h.
///
/// Poisson summation turns the trapezoid error into the Fourier transform of the integrand
/// sampled at multiples of 2*pi/h, and that transform is `p^{-1-i w} Gamma(1 + i w)`. The
/// factor of `p` cancels against the quantity being approximated, which is what makes the
/// bound uniform over the interval; what is left is twice the gamma magnitude, and
/// `|Gamma(1+iy)| = sqrt(pi y / sinh(pi y))` is bounded above by `sqrt(2 pi y) exp(-pi y / 2)`.
double discretization_bound(double h) {
    double const pi = std::acos(-1.0);
    return (4.0 * pi / std::sqrt(h)) * std::exp(-pi * pi / h);
}

/// The largest step whose discretization bound does not exceed @p target.
///
/// Bisection rather than an inversion in closed form: the bound is a product of a power and an
/// exponential and has no elementary inverse, while it is strictly increasing in `h` over the
/// whole range anyone would use, so a bisection is exact to the last bit in fifty steps.
double step_for(double target) {
    double lo = 1.0e-4; // a step nothing would need, whose bound underflows
    double hi = 10.0;   // a step nothing would accept, whose bound is above one
    for (int iteration = 0; iteration < 200; ++iteration) {
        double const mid = 0.5 * (lo + hi);
        if (discretization_bound(mid) > target) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return lo;
}

/// Where the grid starts and ends, and the step, for a range ratio and a tolerance.
struct GridBounds {
    double s_low{0};
    double s_high{0};
    double h{0};
};

GridBounds grid_bounds(double ratio, double epsilon) {
    GridBounds out;
    // A quarter of the budget to each truncation and a half to the discretization. The split
    // is a choice rather than an optimum; what matters is that all three are bounded and that
    // the sum is the target, so the count the caller gets is one the bound actually supports.
    out.s_low  = std::log(0.25 * epsilon / ratio);
    out.s_high = std::log(std::log(4.0 / epsilon));
    out.h      = step_for(0.5 * epsilon);
    return out;
}

void check_epsilon(double epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0 || epsilon >= 1.0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "laplace quadrature: the target accuracy must be a finite number strictly between 0 and 1; got {}",
                                epsilon);
    }
}

/// The positive interval the rule is built over, and whether it came from a negated sum.
struct PositiveInterval {
    double low{0};
    double high{0};
    bool   negated{false};
};

PositiveInterval positive_interval(SpectralRange const &range) {
    if (!std::isfinite(range.low) || !std::isfinite(range.high) || range.low > range.high) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "laplace quadrature: the spectral range [{}, {}] is not an interval", range.low,
                                range.high);
    }
    if (range.low > 0.0) {
        return PositiveInterval{.low = range.low, .high = range.high, .negated = false};
    }
    if (range.high < 0.0) {
        return PositiveInterval{.low = -range.high, .high = -range.low, .negated = true};
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                            "laplace quadrature: the denominator's range [{}, {}] touches or straddles zero, so its reciprocal is "
                            "unbounded and no quadrature of the exponential integral represents it",
                            range.low, range.high);
}

} // namespace

SpectralRange spectral_range(std::vector<double> const &axis_low, std::vector<double> const &axis_high, std::vector<int> const &signs) {
    if (axis_low.empty() || axis_low.size() != axis_high.size() || axis_low.size() != signs.size()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "laplace quadrature: {} low bounds, {} high bounds and {} signs; one of each per axis is required",
                                axis_low.size(), axis_high.size(), signs.size());
    }
    SpectralRange out;
    for (std::size_t axis = 0; axis < axis_low.size(); ++axis) {
        if (signs[axis] >= 0) {
            out.low += axis_low[axis];
            out.high += axis_high[axis];
        } else {
            out.low -= axis_high[axis];
            out.high -= axis_low[axis];
        }
    }
    return out;
}

std::size_t quadrature_point_count(SpectralRange const &range, double epsilon) {
    check_epsilon(epsilon);
    PositiveInterval const interval = positive_interval(range);
    GridBounds const       bounds   = grid_bounds(interval.high / interval.low, epsilon);
    auto const             count    = static_cast<std::size_t>(std::ceil((bounds.s_high - bounds.s_low) / bounds.h)) + 1;
    return std::max<std::size_t>(count, 2);
}

QuadratureGrid build_quadrature(SpectralRange const &range, double epsilon, std::size_t points) {
    check_epsilon(epsilon);
    if (points < 2) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "laplace quadrature: a trapezoidal rule needs at least two points; got {}", points);
    }
    PositiveInterval const interval = positive_interval(range);
    GridBounds const       bounds   = grid_bounds(interval.high / interval.low, epsilon);

    // The step is recomputed from the count rather than taken from the bound, so the grid ends
    // exactly where the truncation analysis says it should. A count above what the bound asks
    // for therefore SHRINKS the step and buys accuracy; one below it stretches the step and
    // spends accuracy, which is what an explicit override is for and what the measured error
    // then reports.
    double const step = (bounds.s_high - bounds.s_low) / static_cast<double>(points - 1);
    double const sign = interval.negated ? -1.0 : 1.0;

    QuadratureGrid grid;
    grid.x_low   = interval.low;
    grid.x_high  = interval.high;
    grid.negated = interval.negated;
    grid.points.reserve(points);
    grid.weights.reserve(points);
    for (std::size_t j = 0; j < points; ++j) {
        double const s = bounds.s_low + static_cast<double>(j) * step;
        double const t = std::exp(s) / interval.low;
        grid.points.push_back(t);
        grid.weights.push_back(sign * step * std::exp(s) / interval.low);
    }
    return grid;
}

double measure_quadrature_error(QuadratureGrid const &grid, std::size_t samples) {
    if (grid.points.empty() || grid.x_low <= 0.0 || grid.x_high < grid.x_low) {
        return 0.0;
    }
    std::size_t const count = std::max<std::size_t>(samples, 2);
    double const      span  = std::log(grid.x_high / grid.x_low);
    double            worst = 0.0;
    for (std::size_t sample = 0; sample < count; ++sample) {
        // Logarithmically spaced, because the rule is built on a logarithmic grid and a
        // linear sweep would leave the small end, where the reciprocal is largest and the
        // truncation bites, described by a handful of points.
        double const x     = grid.x_low * std::exp(span * static_cast<double>(sample) / static_cast<double>(count - 1));
        double       value = 0.0;
        for (std::size_t j = 0; j < grid.points.size(); ++j) {
            value += grid.weights[j] * std::exp(-x * grid.points[j]);
        }
        // The rule approximates 1/x for the POSITIVE interval and carries the caller's sign in
        // its weights, so the exact value it is measured against carries the same sign.
        double const exact = (grid.negated ? -1.0 : 1.0) / x;
        worst              = std::max(worst, std::abs(value - exact) * x);
    }
    return worst;
}

EINSUMS_NAMESPACE_END(compute_graph::laplace)
