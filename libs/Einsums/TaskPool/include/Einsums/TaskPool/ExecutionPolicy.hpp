//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <variant>

EINSUMS_NAMESPACE_BEGIN(task_pool::execution)

/// @brief Sequential execution policy: run on calling thread.
struct SequentialPolicy {};

/// @brief Parallel execution policy: distribute across TaskPool workers.
struct ParallelPolicy {
    size_t chunk_size{0}; ///< 0 = automatic (4x oversubscription)
};

/// @brief Pre-defined policy instances.
inline constexpr SequentialPolicy seq{};
inline constexpr ParallelPolicy   par{};

/// @brief Type-erased execution policy.
using Policy = std::variant<SequentialPolicy, ParallelPolicy>;

EINSUMS_NAMESPACE_END(task_pool::execution)
