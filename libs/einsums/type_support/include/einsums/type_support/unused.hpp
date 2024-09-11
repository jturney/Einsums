//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

namespace einsums::type_support::detail {

struct unused_t {
    constexpr EINSUMS_FORCEINLINE unused_t() noexcept = default;

    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE unused_t(unused_t const &) noexcept {}
    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE unused_t(unused_t &&) noexcept {}

    template <typename T>
    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE unused_t(T const &) noexcept {}

    template <typename T>
    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE unused_t const &operator=(T const &) const noexcept {
        return *this;
    }

    template <typename T>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE unused_t &operator=(T const &) noexcept {
        return *this;
    }

    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE unused_t const &operator=(unused_t const &) const noexcept { return *this; }
    constexpr EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE unused_t const &operator=(unused_t &&) const noexcept { return *this; }

    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE unused_t &operator=(unused_t const &) noexcept { return *this; }
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE unused_t &operator=(unused_t &&) noexcept { return *this; }
};

constexpr unused_t unused = unused_t();

} // namespace einsums::type_support::detail

#if defined(__CUDA_ARCH__)
#    define EINSUMS_UNUSED(x) (void)x
#else
#    define EINSUMS_UNUSED(x) ::einsums::type_support::detail::unused = (x)
#endif

#define EINSUMS_MAYBE_UNUSED [[maybe_unused]]
