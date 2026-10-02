//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>

// The sink must be callable from GPU kernels under a device compiler.
#if defined(__CUDACC__) || defined(__HIPCC__)
#    define EINSUMS_DETAIL_UNUSED_HD __host__ __device__
#else
#    define EINSUMS_DETAIL_UNUSED_HD
#endif

EINSUMS_NAMESPACE_BEGIN(util)

/// Marks its arguments as used, silencing unused warnings. Binds references only, so it compiles
/// to nothing.
template <typename... T>
EINSUMS_DETAIL_UNUSED_HD constexpr void unused(T &&...) noexcept {
}

EINSUMS_NAMESPACE_END(util)

/// Silence unused warnings for one or more variables: ``EINSUMS_UNUSED(a, b)``.
/// The macro supplies the trailing semicolon (existing call sites omit it).
#define EINSUMS_UNUSED(...) ::einsums::util::unused(__VA_ARGS__);
