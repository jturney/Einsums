//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

/*
 * The buffer allocator's options. Both are memory sizes written the way a
 * person writes them ("4MB"), so they are strings parsed by
 * string_util::memory_string rather than integers.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// How much memory the tensor-contraction buffers may hold in total.
///
/// The ceiling exists to catch a RUNAWAY temporary with a named error, not to
/// ration legitimate workspace. 4MB served while one contraction ran at a
/// time, but the moldable executors run a team's worth of kernels at once and
/// each may hold transient TTGT scratch of a few hundred kilobytes: ten
/// concurrent contractions sat within a hair of the old ceiling, and which
/// side they landed on depended on how the scheduler packed them. 64MB keeps
/// the runaway protection - a leaked population still hits it - with room for
/// a wide machine's worth of honest workspace.
inline constinit cl::ConfigOption<std::string> BufferSize = cl::config_opt<std::string>(
    "einsums:buffer-size", "Total size of buffers allocated for tensor contractions", "Buffer Allocator", "64MB", "size");

/// The largest single work buffer. Zero lets the allocator decide.
inline constinit cl::ConfigOption<std::string> WorkBufferSize = cl::config_opt<std::string>(
    "einsums:work-buffer-size",
    "The largest buffer size to use for buffered contractions. Should be much smaller than the max buffer size. The maximum should be the "
    "value of --einsums:buffer-size divided by three times the number of threads. In reality, the program will need more space for other "
    "buffers, so the size should be much smaller than that. Setting to zero will let the program decide.",
    "Buffer Allocator", "0", "size");

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the allocator's options their command-line presence. Idempotent.
 *
 * Run from a namespace-scope initializer rather than from the module's
 * argument hook, because the hook fires part-way through `initialize()` and
 * anything that enumerates the registry earlier - the Python binding layer
 * building argv, for one - would not see these options at all.
 */
EINSUMS_EXPORT int register_Einsums_BufferAllocator_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_BufferAllocator = register_Einsums_BufferAllocator_options();
}

EINSUMS_NAMESPACE_END()
