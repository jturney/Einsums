//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <cstdlib>

namespace einsums::detail {
template <std::uint64_t N>
struct fibhash_helper;

template <>
struct fibhash_helper<0> {
    static constexpr int log2 = -1;
};

template <std::uint64_t N>
struct fibhash_helper {
    static constexpr std::uint64_t log2         = fibhash_helper<(N >> 1)>::log2 + 1;
    static constexpr std::uint64_t shift_amount = 64 - log2;
};

inline constexpr std::uint64_t golden_ratio = 11400714819323198485llu;

// This function calculates the hash based on a multiplicative Fibonacci
// scheme
template <std::uint64_t N>
constexpr std::uint64_t fibhash(std::uint64_t i) noexcept {
    using helper = fibhash_helper<N>;
    static_assert(N != 0, "This algorithm only works with N != 0");
    static_assert((1 << helper::log2) == N, "N must be a power of two"); // -V104
    return (detail::golden_ratio * (i ^ (i >> helper::shift_amount))) >> helper::shift_amount;
}
} // namespace einsums::detail
