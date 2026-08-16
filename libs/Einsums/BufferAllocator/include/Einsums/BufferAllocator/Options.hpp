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
inline constinit cl::ConfigOption<std::string> BufferSize = cl::config_opt<std::string>(
    "einsums:buffer-size", "Total size of buffers allocated for tensor contractions", "Buffer Allocator", "4MB");

/// The largest single work buffer. Zero lets the allocator decide.
inline constinit cl::ConfigOption<std::string> WorkBufferSize = cl::config_opt<std::string>(
    "einsums:work-buffer-size",
    "The largest buffer size to use for buffered contractions. Should be much smaller than the max buffer size. The maximum should be the "
    "value of --einsums:buffer-size divided by three times the number of threads. In reality, the program will need more space for other "
    "buffers, so the size should be much smaller than that. Setting to zero will let the program decide.",
    "Buffer Allocator", "0");

EINSUMS_NAMESPACE_END(option)
