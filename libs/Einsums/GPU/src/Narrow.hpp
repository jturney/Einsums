//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS/Types.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <concepts>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(gpu::detail)

/**
 * @brief @p value as the integer type a backend routine takes, or a DimensionError naming @p name.
 *
 * The GPU entry points take int64_t extents. The routines they forward to take narrower ones:
 * cuBLAS, cuSOLVER, hipBLAS and MPS take int, and the CPU vendor layer takes blas::int_t, which is
 * int unless the build is ILP64. A static_cast wrapped an extent past the target's range into a
 * small or negative one, and the routine then ran on it.
 */
template <std::integral To, std::integral From>
To narrow_extent(From value, char const *name) {
    if (!std::in_range<To>(value)) {
        EINSUMS_THROW_EXCEPTION(DimensionError, "{} = {} does not fit the {}-bit integer this backend routine takes", name, value,
                                8 * sizeof(To));
    }
    return static_cast<To>(value);
}

/// An extent for a CPU vendor routine (the mock backend, and MPS's non-float32 fallbacks).
template <std::integral From>
::einsums::blas::int_t vendor_extent(From value, char const *name) {
    return narrow_extent<::einsums::blas::int_t>(value, name);
}

/// An extent for a routine that takes int: cuBLAS, cuSOLVER, hipBLAS, hipSOLVER and MPS.
template <std::integral From>
int int_extent(From value, char const *name) {
    return narrow_extent<int>(value, name);
}

EINSUMS_NAMESPACE_END(gpu::detail)
