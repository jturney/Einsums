//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <string>

#if defined(EINSUMS_HAVE_BACKTRACES)

EINSUMS_NAMESPACE_BEGIN(util)

/**
 * @brief Generate a backtrace.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT std::string backtrace(std::size_t frames_no = EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH);

EINSUMS_NAMESPACE_END(util)

#else

EINSUMS_NAMESPACE_BEGIN(util)

/**
 * @brief Generate a backtrace.
 *
 * @versionadded{1.0.0}
 */
inline std::string backtrace(std::size_t frames_no = 0) {
    return "";
}

EINSUMS_NAMESPACE_END(util)

#endif
