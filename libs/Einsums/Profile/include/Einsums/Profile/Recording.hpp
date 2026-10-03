//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile/Profile.hpp>

EINSUMS_NAMESPACE_BEGIN(profile)

/**
 * @brief Whether Einsums records profiling data right now.
 *
 * Constant false in a build configured with EINSUMS_WITH_PROFILER=OFF, so code gated on it compiles
 * to nothing there; otherwise the profiler's recording switch. Code that calls the profiler directly,
 * rather than through the zone macros, reads this once and does nothing when it is false.
 */
inline bool recording() noexcept {
#if defined(EINSUMS_HAVE_PROFILER)
    return waggle::Profiler::instance().enabled();
#else
    return false;
#endif
}

EINSUMS_NAMESPACE_END(profile)
