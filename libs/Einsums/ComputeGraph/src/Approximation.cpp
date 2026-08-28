//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Approximation.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <array>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// The one name table. A spelling and its parse come from the same array, so the two cannot
/// drift the way two switch statements do.
constexpr std::array<std::pair<ApproximationEffect, std::string_view>, 3> effect_names{{
    {ApproximationEffect::ElementWise, "element-wise"},
    {ApproximationEffect::NormRelative, "norm-relative"},
    {ApproximationEffect::EnergyLike, "energy-like"},
}};

} // namespace

std::string_view approximation_effect_name(ApproximationEffect effect) noexcept {
    for (auto const &[value, name] : effect_names) {
        if (value == effect) {
            return name;
        }
    }
    return "unknown";
}

std::optional<ApproximationEffect> approximation_effect_from_name(std::string_view name) noexcept {
    for (auto const &[value, spelling] : effect_names) {
        if (spelling == name) {
            return value;
        }
    }
    return std::nullopt;
}

ApproximationRecord make_approximation_record(std::string pass_name, ApproximationEffect effect, double tolerance, double bound,
                                              std::vector<std::string> outputs, std::vector<std::string> spaces, std::string setup) {
    return ApproximationRecord{.pass_name = std::move(pass_name),
                               .tolerance = tolerance,
                               .effect    = effect,
                               .bound     = bound,
                               .outputs   = std::move(outputs),
                               .spaces    = std::move(spaces),
                               .setup     = std::move(setup)};
}

EINSUMS_NAMESPACE_END(compute_graph)
