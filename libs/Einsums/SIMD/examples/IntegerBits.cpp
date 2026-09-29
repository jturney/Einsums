//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file IntegerBits.cpp
/// @brief Integer Vecs: shifts, bitwise logic, equality masks, and a horizontal sum.
///
/// Two small jobs on uint32_t and int32_t data:
///
///   - A hash of every element (xorshift: x ^= x << 13; x ^= x >> 17; x ^= x << 5), which is
///     nothing but shift_left, shift_right and bitwise_xor, lanes at a time. The shift count is
///     a template argument, because every ISA encodes it in the instruction.
///   - Counting the elements equal to a key. cmp_eq gives -1 (all ones) in matching lanes and 0
///     elsewhere, so subtracting the mask from a counter adds one per match without a branch, and
///     reduce_add totals the counters at the end.
///
/// AVX without AVX2 has no 256-bit integer instructions: on such a build these operations are
/// missing and the program fails to link, rather than silently running something else.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Reduce.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace einsums::simd;

namespace {

std::uint32_t xorshift(std::uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

Vec<std::uint32_t> xorshift(Vec<std::uint32_t> x) {
    x = bitwise_xor(x, shift_left<13>(x));
    x = bitwise_xor(x, shift_right<17>(x));
    x = bitwise_xor(x, shift_left<5>(x));
    return x;
}

/// How many of the n elements of @p v equal @p key.
std::int32_t count_equal(std::size_t n, std::int32_t const *v, std::int32_t key) {
    constexpr std::size_t   L     = lanes<std::int32_t>;
    Vec<std::int32_t> const vkey  = broadcast(key);
    Vec<std::int32_t>       count = broadcast(std::int32_t{0});
    std::size_t             i     = 0;
    for (; i + L <= n; i += L) {
        count = sub(count, cmp_eq(loadu(v + i), vkey)); // a match is -1, so this adds one
    }
    std::int32_t total = reduce_add(count);
    for (; i < n; ++i) {
        total += v[i] == key;
    }
    return total;
}

} // namespace

int einsums_main() {
    constexpr std::size_t      L = lanes<std::uint32_t>;
    std::size_t const          n = 16 * L;
    std::vector<std::uint32_t> seeds(n), hashed(n);
    for (std::size_t i = 0; i < n; ++i) {
        seeds[i] = static_cast<std::uint32_t>(i * 2654435761u + 1);
    }
    for (std::size_t i = 0; i < n; i += L) {
        storeu(hashed.data() + i, xorshift(loadu(seeds.data() + i)));
    }
    int failures = 0;
    for (std::size_t i = 0; i < n; ++i) {
        failures += hashed[i] != xorshift(seeds[i]);
    }
    std::cout << "xorshift of " << n << " seeds, " << L << " at a time: " << (failures ? "FAILED" : "ok") << "\n";

    std::vector<std::int32_t> values(1003);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<std::int32_t>(i % 7) - 3;
    }
    std::int32_t expect = 0;
    for (std::int32_t const x : values) {
        expect += x == 2;
    }
    std::int32_t const got = count_equal(values.size(), values.data(), 2);
    std::cout << "elements equal to 2: " << got << " (expected " << expect << ")\n";
    failures += got != expect;
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
