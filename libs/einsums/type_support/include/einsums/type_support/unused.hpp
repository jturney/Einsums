//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// clang-format off
#include <einsums/config.hpp>
#if defined(EINSUMS_MSVC)
#pragma warning(push)
#pragma warning(disable: 4522)
#endif
// clang-format on

namespace einsums::util::detail {

struct UnusedType {
    constexpr EINSUMS_FORCEINLINE UnusedType() noexcept = default;

    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE UnusedType(UnusedType const &) noexcept = default;
    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE UnusedType(UnusedType &&) noexcept      = default;

    template <typename T>
    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE UnusedType(T const &) noexcept {}

    template <typename T>
    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE UnusedType const &operator=(T const &) const noexcept {
        return *this;
    }
    template <typename T>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE UnusedType &operator=(T const &) noexcept {
        return *this;
    }

    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE UnusedType const &operator=(UnusedType const &) const noexcept {
        return *this;
    }
    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE UnusedType const &operator=(UnusedType &&) const noexcept {
        return *this;
    }

    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE UnusedType &operator=(UnusedType const &) noexcept { return *this; }
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE UnusedType &operator=(UnusedType &&) noexcept { return *this; }
};

#if defined(EINSUMS_MSVC_NVCC)
EINSUMS_CONSTANT
#endif
constexpr UnusedType Unused = UnusedType();

} // namespace einsums::util::detail

//////////////////////////////////////////////////////////////////////////////
// Use this to silence compiler warnings related to unused function arguments.
#if defined(__CUDA_ARCH__)
#    define EINSUMS_UNUSED(x) (void)x
#else
#    define EINSUMS_UNUSED(x) ::einsums::util::detail::Unused = (x);
#endif

//////////////////////////////////////////////////////////////
// Use this to silence compiler warnings for global variables.
#define EINSUMS_MAYBE_UNUSED [[maybe_unused]]

#if defined(EINSUMS_MSVC)
#    pragma warning(pop)
#endif