//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file BudgetShare.hpp
/// @brief The share of the memory budget one tensor may hold, and the bytes a backed tensor holds
///        whole.
///
/// AxisTiling streams an intermediate over its share and SlabStreaming a backed tensor, and every
/// pass that would copy a backed tensor whole asks the same question before it does. One answer,
/// so a tensor one pass streams is never one another holds whole.

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/TensorBacking.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>

#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::detail)

/// @brief The memory budget, @c --einsums:max-memory, in bytes; zero when there is none.
[[nodiscard]] inline std::size_t memory_budget() {
    return string_util::memory_string(config::get(option::MaxMemory));
}

/// @brief The most one tensor may hold: @c --einsums:graph:tiling-budget-share of the budget, or
///        the whole budget when the share is off. Zero when there is no budget.
[[nodiscard]] inline std::size_t budget_share() {
    std::size_t const budget = memory_budget();
    if (budget == 0) {
        return 0;
    }
    double const fraction = config::get(option::GraphTilingBudgetShare);
    return fraction > 0.0 && fraction < 1.0 ? static_cast<std::size_t>(fraction * static_cast<double>(budget)) : budget;
}

/// @brief The bytes of the tensor a backing stores, whole.
[[nodiscard]] inline std::size_t stored_bytes(TensorBacking const &backing) {
    std::size_t bytes = compute_graph::detail::dispatch_scalar_type(backing.dtype, []<typename T>(T) { return sizeof(T); });
    for (std::size_t const extent : backing.dims) {
        bytes *= extent;
    }
    return bytes;
}

EINSUMS_NAMESPACE_END(compute_graph::passes::detail)
