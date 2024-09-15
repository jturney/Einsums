//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstdint>

namespace einsums::chrono::detail {
EINSUMS_DEVICE inline std::uint64_t timestamp_cuda() {
    std::uint64_t cur;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(cur));
    return cur;
}
} // namespace einsums::chrono::detail
