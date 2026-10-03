//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file
/// Einsums' entry to its profiler, Waggle: include this, not <Waggle/Waggle.hpp>, so that Einsums'
/// configuration reaches the instrumentation macros. Also the einsums.profile Python surface.

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

// Decided here, by a header every user of Einsums' headers includes, so that Einsums' own sources
// and its users' translation units expand the macros alike. Only the macros change: the profiler
// itself is always built and shared, whatever Einsums was configured with.
#if !defined(EINSUMS_HAVE_PROFILER) && !defined(WAGGLE_DISABLE)
#    define WAGGLE_DISABLE
#endif
#if defined(EINSUMS_HAVE_PROFILER_INTERNAL) && !defined(WAGGLE_DETAIL)
#    define WAGGLE_DETAIL
#endif

#include <Waggle/Waggle.hpp>

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

// Every zone written in namespace einsums, in its headers too, belongs to the einsums domain.
EINSUMS_NAMESPACE_BEGIN()
WAGGLE_DEFINE_DOMAIN("einsums")
EINSUMS_NAMESPACE_END()

EINSUMS_NAMESPACE_BEGIN(profile)

// ---------------------- Python bindings ----------------------
// einsums.profile is Waggle's own Python API (waggle._core). What Einsums adds is whether its build
// records: with EINSUMS_WITH_PROFILER=OFF, einsums.profile keeps every name callable and records
// nothing, so instrumented scripts run on that build too.

/// Whether this build records anything. ``False`` means einsums was built with
/// ``EINSUMS_WITH_PROFILER=OFF``: the API still exists but does nothing.
APIARY_EXPOSE APIARY_MODULE("profile") constexpr bool available() {
#if defined(EINSUMS_HAVE_PROFILER)
    return true;
#else
    return false;
#endif
}

EINSUMS_NAMESPACE_END(profile)
