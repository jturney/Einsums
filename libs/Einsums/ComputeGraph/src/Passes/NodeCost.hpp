//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file NodeCost.hpp
 * @brief The model price of one node, for the passes that plan by cost.
 *
 * Private to the passes. ThreadPlanning prices every node it has no measurement for from here,
 * and a registered descriptor's @c cost hook (DescriptorHooks.hpp) answers for its own node.
 */

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// @brief What the model says a node costs at width 1 on a device, and how it widens.
struct ModelledCost {
    double       t1_us{0.0};                        ///< Serial time; 0 for a Loop or Conditional, priced by its body
    KernelFamily family{KernelFamily::Elementwise}; ///< The speedup curve it scales along
    std::size_t  bytes{0};                          ///< Working set, for the curve's size class
    CurveRoute   route{CurveRoute::Einsums};        ///< Whose threads widen it; see @ref CurveRoute
    /// Independent members a grouped or batched node runs across its threads; 0 for any other
    /// node. A node cannot keep more threads busy than it has members.
    std::size_t members{0};
    /// Extent of one member's GEMM for a batched GEMM or a sandwich, the cube root of its
    /// `m*n*k`; 0 for any other node. It picks the batched curve; see
    /// @ref EfficiencyCurve::member_extent.
    std::size_t member_extent{0};
    /// A registered descriptor's own speedup curve, overriding @ref family's; empty otherwise.
    std::function<double(unsigned)> speedup;
};

/**
 * @brief Contraction prices already computed in one pricing sweep, keyed by everything the price
 *        depends on: the index lists, each operand's dims and strides, and the output's dtype.
 *
 * A graph built per chunk of a sweep repeats a handful of contraction layouts across thousands of
 * nodes (the DLPNO (T0) triples price 88 distinct layouts across 1320 einsums), and describing a
 * layout to the packed engine was the larger part of pricing one. Valid for one profile only: the
 * owner keeps it for one sweep over one graph and prices with one profile throughout.
 */
struct ContractionPriceMemo {
    std::unordered_map<std::string, std::optional<double>> prices;
};

/// @brief The model price of @p node in @p graph on @p profile, from its registered @c cost hook
///        when it has one that answers, otherwise from its kind and shape. A contraction's price
///        is looked up in @p memo first and recorded there, when one is given.
[[nodiscard]] ModelledCost model_cost(Graph const &graph, Node const &node, DeviceProfile const &profile,
                                      ContractionPriceMemo *memo = nullptr);

/// @brief How many times faster than serial @p c runs at @p width. A grouped node's members are
///        its parallelism, so past that many threads it gains nothing: it runs as it does at its
///        member count.
///
/// @p c is a @ref ModelledCost or anything carrying the same fields.
template <typename Cost>
[[nodiscard]] double speedup_at(Cost const &c, DeviceProfile const &profile, unsigned width) {
    unsigned const w = c.members > 0 ? static_cast<unsigned>(std::min<std::size_t>(width, c.members)) : width;
    return c.speedup ? c.speedup(w) : profile.parallel_speedup(c.family, c.bytes, w, c.route, c.member_extent);
}

/// @brief What opening its region costs @p c at @p width, in microseconds, where the curve that
///        prices it is net of that cost; zero otherwise. See DeviceProfile::region_fee_us.
template <typename Cost>
[[nodiscard]] double region_fee_at(Cost const &c, DeviceProfile const &profile, unsigned width) {
    if (c.speedup || width <= 1) {
        return 0.0;
    }
    unsigned const w = c.members > 0 ? static_cast<unsigned>(std::min<std::size_t>(width, c.members)) : width;
    return profile.region_fee_us(c.family, w, c.route);
}

/// @brief The time of @p c on @p width threads, in microseconds.
///
/// The part of the serial time above what a kernel costs to start is divided by the speedup; the
/// start cost is not, and a node whose curve is net of its region's fork pays that fee on top,
/// once per call.
template <typename Cost>
[[nodiscard]] double time_at_width(Cost const &c, DeviceProfile const &profile, unsigned width) {
    if (width <= 1) {
        return c.t1_us;
    }
    double const overhead_us = profile.kernel_launch_overhead_us;
    double const scalable    = std::max(0.0, c.t1_us - overhead_us);
    double const speedup     = std::max(1e-9, speedup_at(c, profile, width));
    return scalable / speedup + std::min(overhead_us, c.t1_us) + region_fee_at(c, profile, width);
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
