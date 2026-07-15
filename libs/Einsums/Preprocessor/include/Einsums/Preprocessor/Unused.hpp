//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Marking device code: the sink below must be callable from GPU kernels when a
// device compiler is active, and stay annotation-free everywhere else.
#if defined(__CUDACC__) || defined(__HIPCC__)
#    define EINSUMS_DETAIL_UNUSED_HD __host__ __device__
#else
#    define EINSUMS_DETAIL_UNUSED_HD
#endif

namespace einsums::util {

/// Sink for silencing unused-variable/parameter warnings.
///
/// A variadic no-op function: takes any number of arguments by forwarding
/// reference and does nothing. The emitted code contains no copies, moves, or
/// calls - the references bind and the empty body is inlined away - while the
/// compiler sees every argument as used. This is a leaner descendant of the
/// pre-restructuring ``einsums::util::detail::UnusedType`` assignment sink
/// (type_support/unused.hpp), rebuilt as a plain function per review feedback
/// on PR #268.
template <typename... T>
EINSUMS_DETAIL_UNUSED_HD constexpr void unused(T &&...) noexcept {
}

} // namespace einsums::util

/// Silence unused warnings for one or more variables: ``EINSUMS_UNUSED(a, b)``.
/// The macro supplies the trailing semicolon (existing call sites omit it).
#define EINSUMS_UNUSED(...) ::einsums::util::unused(__VA_ARGS__);
