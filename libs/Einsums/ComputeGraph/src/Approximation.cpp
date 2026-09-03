//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Approximation.hpp>
#include <Einsums/ComputeGraphTypes/EnumNames.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <array>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// The one name table. A spelling and its parse come from the same array, so the two cannot
/// drift the way two switch statements do.
constexpr EnumNames effect_names{std::array<std::pair<ApproximationEffect, std::string_view>, 3>{{
                                     {ApproximationEffect::ElementWise, "element-wise"},
                                     {ApproximationEffect::NormRelative, "norm-relative"},
                                     {ApproximationEffect::EnergyLike, "energy-like"},
                                 }},
                                 "unknown"};

/// The origin name table, a table for the same reason the effect one is.
constexpr EnumNames origin_names{std::array<std::pair<ApproximationOrigin, std::string_view>, 2>{{
                                     {ApproximationOrigin::Measured, "measured"},
                                     {ApproximationOrigin::Asserted, "asserted"},
                                 }},
                                 "unknown"};

} // namespace

std::string_view approximation_effect_name(ApproximationEffect effect) noexcept {
    return effect_names.name(effect);
}

std::optional<ApproximationEffect> approximation_effect_from_name(std::string_view name) noexcept {
    return effect_names.from_name(name);
}

std::string_view approximation_origin_name(ApproximationOrigin origin) noexcept {
    return origin_names.name(origin);
}

std::optional<ApproximationOrigin> approximation_origin_from_name(std::string_view name) noexcept {
    return origin_names.from_name(name);
}

ApproximationRecord make_approximation_record(std::string pass_name, ApproximationEffect effect, double tolerance, double bound,
                                              std::vector<std::string> outputs, std::vector<std::string> spaces, std::string setup,
                                              ApproximationOrigin origin) {
    return ApproximationRecord{.pass_name = std::move(pass_name),
                               .tolerance = tolerance,
                               .effect    = effect,
                               .bound     = bound,
                               .origin    = origin,
                               .outputs   = std::move(outputs),
                               .spaces    = std::move(spaces),
                               .setup     = std::move(setup)};
}

EINSUMS_NAMESPACE_END(compute_graph)
