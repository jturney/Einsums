//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ParamStore.cpp
/// @brief The gate-flag names a graph keeps, in name order.

#include <Einsums/ComputeGraph/Detail/ParamStore.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

auto ParamStore::gate_flags_named(std::string_view name) const noexcept -> FlagBuffer const * {
    auto const it = std::ranges::find_if(_gate_flags, [name](auto const &entry) { return entry.first == name; });
    return it == _gate_flags.end() ? nullptr : &it->second;
}

std::string const *ParamStore::gate_flag_name(FlagBuffer const &buffer) const noexcept {
    auto const it = std::ranges::find_if(_gate_flags, [&buffer](auto const &entry) { return entry.second == buffer; });
    return it == _gate_flags.end() ? nullptr : &it->first;
}

void ParamStore::add_gate_flags(std::string name, FlagBuffer buffer) {
    _gate_flags.emplace_back(std::move(name), std::move(buffer));
    std::ranges::sort(_gate_flags, [](auto const &lhs, auto const &rhs) { return lhs.first < rhs.first; });
}

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
