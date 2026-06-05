//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Unreachable.hpp
/// @brief C++23 std::unreachable() backport for C++20.
///
/// Marks a code path as unreachable. In release builds, this enables
/// compiler optimizations. In debug builds, it triggers an assertion.
///
/// When C++23 is available, define EINSUMS_USE_STD_UNREACHABLE to use
/// the standard library version.

#include <Einsums/Config.hpp>

#if defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#    include <utility>
#endif

namespace einsums {

/**
 * @brief Marks a code path as unreachable.
 *
 * Invokes undefined behavior if reached. The compiler uses this to optimize
 * away impossible branches (e.g., default cases in exhaustive switches).
 *
 * In debug builds (NDEBUG not defined), triggers __builtin_trap() for
 * immediate crash with a debuggable stack trace.
 *
 * @par Example
 * @code
 * switch (kind) {
 * case OpKind::Einsum:  return "Einsum";
 * case OpKind::Scale:   return "Scale";
 * // ... all cases covered ...
 * }
 * einsums::unreachable();
 * @endcode
 */
[[noreturn]] inline void unreachable() {
#if defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
    std::unreachable();
#elif defined(NDEBUG)
    // Release: UB hint for optimizer
#    if defined(__GNUC__) || defined(__clang__)
    __builtin_unreachable();
#    elif defined(_MSC_VER)
    __assume(false);
#    endif
#else
    // Debug: crash immediately for diagnosis
#    if defined(__GNUC__) || defined(__clang__)
    __builtin_trap();
#    elif defined(_MSC_VER)
    __debugbreak();
#    endif
#endif
}

} // namespace einsums
