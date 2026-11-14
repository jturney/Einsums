//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BindingSession.cpp
/// @brief The record binds keep about the caller's problem.

#include <Einsums/ComputeGraph/Detail/BindingSession.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

std::string const *BindingSession::interface_name(TensorId id) const noexcept {
    auto const it = _interface_names.find(id);
    return it == _interface_names.end() ? nullptr : &it->second;
}

RaggedExtentTable const *BindingSession::find_ragged(std::string_view name, std::size_t axis) const noexcept {
    auto const it = std::ranges::find_if(_ragged, [&](RaggedExtentTable const &held) { return held.name == name && held.axis == axis; });
    return it == _ragged.end() ? nullptr : &*it;
}

void BindingSession::hold_ragged(RaggedExtentTable table) {
    auto const existing =
        std::ranges::find_if(_ragged, [&](RaggedExtentTable const &held) { return held.id == table.id && held.axis == table.axis; });
    if (existing != _ragged.end()) {
        existing->name    = std::move(table.name);
        existing->space   = std::move(table.space);
        existing->extents = std::move(table.extents);
        return;
    }
    _ragged.push_back(std::move(table));
}

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
