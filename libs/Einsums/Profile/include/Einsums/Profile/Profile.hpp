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

EINSUMS_NAMESPACE_BEGIN(profile)

// ---------------------- Python bindings ----------------------
// Free functions over Waggle, so einsums.profile need not bind the non-copyable Profiler. With
// EINSUMS_WITH_PROFILER=OFF each does nothing, so instrumented scripts run on that build too.
// Parameters keep their names: they are the bindings' keyword arguments.

/// Whether this build records anything. ``False`` means einsums was built with
/// ``EINSUMS_WITH_PROFILER=OFF``: the API still exists but does nothing.
APIARY_EXPOSE APIARY_MODULE("profile") constexpr bool available() {
#if defined(EINSUMS_HAVE_PROFILER)
    return true;
#else
    return false;
#endif
}

/// Attach a string annotation to the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void annotate([[maybe_unused]] std::string_view key,
                                                            [[maybe_unused]] std::string_view value) {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::annotate(key, value);
#endif
}

/// Attach an integer annotation to the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void annotate([[maybe_unused]] std::string_view key, [[maybe_unused]] int64_t value) {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::annotate(key, value);
#endif
}

/// Attach a floating-point annotation to the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void annotate([[maybe_unused]] std::string_view key, [[maybe_unused]] double value) {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::annotate(key, value);
#endif
}

/// Record a memory allocation in the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void mem_alloc([[maybe_unused]] int64_t bytes) {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::mem_alloc(bytes);
#endif
}

/// Record a memory deallocation in the current profiling zone.
APIARY_EXPOSE APIARY_MODULE("profile") inline void mem_free([[maybe_unused]] int64_t bytes) {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::mem_free(bytes);
#endif
}

/// Begin a profile region. Pair it with ``pop()``, usually through the
/// ``einsums.profile.section(name)`` context manager.
APIARY_EXPOSE APIARY_MODULE("profile") inline void push([[maybe_unused]] std::string const &name,
                                                        [[maybe_unused]] std::string const &file = "", [[maybe_unused]] int line = 0,
                                                        [[maybe_unused]] std::string const &func = "") {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::Profiler::instance().push(name, file, line, func);
#endif
}

/// End the innermost profile region.
APIARY_EXPOSE APIARY_MODULE("profile") inline void pop() {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::Profiler::instance().pop();
#endif
}

/// Drain the per-thread ring buffers into the aggregated tree, so ``print_report``
/// and ``export_json`` see recent events.
APIARY_EXPOSE APIARY_MODULE("profile") inline void flush() {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::Profiler::instance().flush();
#endif
}

/// Print the compact (or detailed) report to standard output.
APIARY_EXPOSE APIARY_MODULE("profile") inline void print_report([[maybe_unused]] bool detailed = false) {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::Profiler::instance().print(detailed);
    // Flush so pytest's capfd and non-tty stdout see the report before returning.
    std::cout.flush();
#endif
}

/// Write the aggregated profile to JSON. Returns the resolved path on
/// success or ``None`` on failure.
APIARY_EXPOSE APIARY_MODULE("profile") inline std::optional<std::string>
export_json([[maybe_unused]] std::string const &path = "einsums_profile.json") {
#if defined(EINSUMS_HAVE_PROFILER)
    return waggle::Profiler::instance().export_json(path);
#else
    return std::nullopt;
#endif
}

/// Set a human-readable name for the calling thread.
APIARY_EXPOSE APIARY_MODULE("profile") inline void set_thread_name([[maybe_unused]] std::string const &name) {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::Profiler::instance().set_thread_name(name);
#endif
}

/// Return the profiler's thread id for the calling thread.
APIARY_EXPOSE APIARY_MODULE("profile") inline uint32_t current_thread_id() {
#if defined(EINSUMS_HAVE_PROFILER)
    return waggle::Profiler::current_thread_id();
#else
    return 0;
#endif
}

/// Average per-call overhead of ``push`` in nanoseconds.
APIARY_EXPOSE APIARY_MODULE("profile") inline double avg_push_overhead_ns() {
#if defined(EINSUMS_HAVE_PROFILER)
    return waggle::Profiler::instance().avg_push_overhead_ns();
#else
    return 0.0;
#endif
}

/// Average per-call overhead of ``pop`` in nanoseconds.
APIARY_EXPOSE APIARY_MODULE("profile") inline double avg_pop_overhead_ns() {
#if defined(EINSUMS_HAVE_PROFILER)
    return waggle::Profiler::instance().avg_pop_overhead_ns();
#else
    return 0.0;
#endif
}

/// Total number of ``push`` calls observed since process start.
APIARY_EXPOSE APIARY_MODULE("profile") inline uint64_t total_push_count() {
#if defined(EINSUMS_HAVE_PROFILER)
    return waggle::Profiler::instance().total_push_count();
#else
    return 0;
#endif
}

/// Total number of ``pop`` calls observed since process start.
APIARY_EXPOSE APIARY_MODULE("profile") inline uint64_t total_pop_count() {
#if defined(EINSUMS_HAVE_PROFILER)
    return waggle::Profiler::instance().total_pop_count();
#else
    return 0;
#endif
}

EINSUMS_NAMESPACE_END(profile)
