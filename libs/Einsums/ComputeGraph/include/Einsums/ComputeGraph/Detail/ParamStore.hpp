//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file ParamStore.hpp
/// @brief The run-time values a graph's nodes read by name: the parameter table it shares with
///        its pipeline and its bodies, and the gate-flag arrays it can name.

#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

/**
 * @brief A graph's named run-time state.
 *
 * The parameter table is shared: a pipeline hands its own to each stage, and a graph hands its
 * own to every body under it, so one write is seen by every node that reads the name. A
 * standalone graph starts with an empty table of its own.
 *
 * The gate-flag arrays are kept sorted by name, because the order is part of what a saved file
 * records and a hash is taken over: a map's iteration order is not a property of the graph.
 */
class EINSUMS_EXPORT ParamStore {
  public:
    using FlagBuffer = std::shared_ptr<std::vector<std::uint8_t>>;

    [[nodiscard]] std::shared_ptr<ParamTable> const &table() const noexcept { return _params; }
    void                                             set_table(std::shared_ptr<ParamTable> params) noexcept { _params = std::move(params); }

    /// Every named gate-flag array, sorted by name.
    [[nodiscard]] std::vector<std::pair<std::string, FlagBuffer>> const &named_gate_flags() const noexcept { return _gate_flags; }

    /// The array named @p name, or null.
    [[nodiscard]] FlagBuffer const *gate_flags_named(std::string_view name) const noexcept;

    /// The name @p buffer carries, or null when it carries none.
    [[nodiscard]] std::string const *gate_flag_name(FlagBuffer const &buffer) const noexcept;

    /// Name @p buffer; the caller has checked that neither the name nor the array is taken.
    void add_gate_flags(std::string name, FlagBuffer buffer);

  private:
    /// Runtime parameter table. ``View`` executors and the @c WriteParam node read and write
    /// through this.
    std::shared_ptr<ParamTable> _params{std::make_shared<ParamTable>()};

    std::vector<std::pair<std::string, FlagBuffer>> _gate_flags;
};

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
