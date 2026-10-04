//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

// The buffer allocator's options: memory sizes such as "4MB", parsed by string_util::memory_string.

EINSUMS_NAMESPACE_BEGIN(detail)

/// Default for einsums:max-memory: 80% of physical RAM, as a memory string.
EINSUMS_EXPORT std::string default_max_memory();

// An inline forwarder, so the constant-initialized descriptor holds a link-time address; on Windows
// an exported function's address is not a constant outside the library.
inline std::string max_memory_provider() {
    return default_max_memory();
}

EINSUMS_NAMESPACE_END(detail)

EINSUMS_NAMESPACE_BEGIN(option)

/// Ceiling on the memory einsums plans against: chunked algorithms size to it and MemoryPool refuses
/// reservations past it. Ordinary allocations are not checked, so nothing throws in a kernel. 0 disables.
inline constinit cl::ConfigOption<std::string> MaxMemory = cl::config_opt_computed<std::string>(
    "einsums:max-memory", "Ceiling on the memory einsums plans against: chunked algorithms and memory pools budget to it. 0 disables.",
    "Buffer Allocator", &detail::max_memory_provider, "size");

/// Total memory the contraction buffers may hold. Catches a runaway temporary, with room for a
/// wide machine running many contractions at once.
inline constinit cl::ConfigOption<std::string> BufferSize = cl::config_opt<std::string>(
    "einsums:buffer-size", "Total size of buffers allocated for tensor contractions", "Buffer Allocator", "64MB", "size");

/// The largest piece of a disk-backed tensor a contraction reads into memory at once. Zero means an
/// eighth of --einsums:buffer-size.
inline constinit cl::ConfigOption<std::string> WorkBufferSize = cl::config_opt<std::string>(
    "einsums:work-buffer-size",
    "The largest piece of a disk-backed tensor a contraction reads into memory at once; it counts against --einsums:buffer-size. "
    "Zero means an eighth of --einsums:buffer-size.",
    "Buffer Allocator", "0", "size");

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the allocator's options their command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_BufferAllocator_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_BufferAllocator = register_Einsums_BufferAllocator_options();
}

EINSUMS_NAMESPACE_END()
