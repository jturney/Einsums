//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Round-trip tests for the integer Vec specializations. Currently covers
// load/store/broadcast for int32_t / uint32_t / int64_t / uint64_t. The
// element-wise arithmetic (add/sub/bitwise/shifts/compare-eq) lands in
// follow-up commits and gets its own test cases here.

#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Platform.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cstdint>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

TEMPLATE_TEST_CASE("integer Vec lane count matches native register width", "[simd][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    // Lane count derives from native_bits and sizeof(TestType); same rule
    // as float/double. Skip the assertion on the scalar fallback (lanes==1)
    // because native_bits is 0 there and the formula would divide by zero.
    if (native_bits == 0) {
        CHECK(Vec<TestType>::lanes == 1);
    } else {
        CHECK(Vec<TestType>::lanes == native_bits / (8 * sizeof(TestType)));
    }
}

TEMPLATE_TEST_CASE("broadcast: every lane holds the input value", "[simd][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    constexpr int                      N   = Vec<TestType>::lanes;
    TestType const                     val = static_cast<TestType>(42);
    auto                               v   = broadcast(val);
    alignas(native_alignment) TestType buf[N];
    storeu(buf, v);
    for (int i = 0; i < N; ++i)
        CHECK(buf[i] == val);
}

TEMPLATE_TEST_CASE("loadu / storeu round-trip preserves all lanes", "[simd][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    constexpr int N = Vec<TestType>::lanes;
    TestType      input[N];
    for (int i = 0; i < N; ++i)
        input[i] = static_cast<TestType>(i + 1);

    auto     v = loadu<TestType>(input);
    TestType output[N];
    storeu(output, v);

    for (int i = 0; i < N; ++i)
        CHECK(output[i] == input[i]);
}

TEMPLATE_TEST_CASE("loada / storea round-trip preserves all lanes", "[simd][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    constexpr int                      N = Vec<TestType>::lanes;
    alignas(native_alignment) TestType input[N];
    for (int i = 0; i < N; ++i)
        input[i] = static_cast<TestType>((i + 1) * 7);

    auto                               v = loada<TestType>(input);
    alignas(native_alignment) TestType output[N];
    storea(output, v);

    for (int i = 0; i < N; ++i)
        CHECK(output[i] == input[i]);
}

TEMPLATE_TEST_CASE("integer Vec respects sign for boundary values", "[simd][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    // Signed types should round-trip negative values. Unsigned types
    // should round-trip the high bit set.
    constexpr int N = Vec<TestType>::lanes;
    TestType      input[N];
    for (int i = 0; i < N; ++i) {
        if constexpr (std::is_signed_v<TestType>) {
            input[i] = (i % 2 == 0) ? std::numeric_limits<TestType>::min() : std::numeric_limits<TestType>::max();
        } else {
            // Unsigned: alternate 0 and ~0 to exercise full range without
            // signed overflow concerns.
            input[i] = (i % 2 == 0) ? TestType(0) : std::numeric_limits<TestType>::max();
        }
    }

    auto     v = loadu<TestType>(input);
    TestType output[N];
    storeu(output, v);

    for (int i = 0; i < N; ++i)
        CHECK(output[i] == input[i]);
}

// ─── Arithmetic ───────────────────────────────────────────────────────────
namespace {
// Helper that loads two arrays into Vecs, applies a binary op, stores
// back, and returns the result array. Used by the arithmetic tests.
template <typename T, typename Op>
std::vector<T> apply_binop(std::vector<T> const &a, std::vector<T> const &b, Op op) {
    constexpr int N = Vec<T>::lanes;
    REQUIRE(a.size() == size_t(N));
    REQUIRE(b.size() == size_t(N));
    auto           va = loadu<T>(a.data());
    auto           vb = loadu<T>(b.data());
    auto           vc = op(va, vb);
    std::vector<T> out(N);
    storeu(out.data(), vc);
    return out;
}
} // namespace

TEMPLATE_TEST_CASE("add: lane-wise sum, two's-complement wraparound", "[simd][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> a(N), b(N), expected(N);
    for (int i = 0; i < N; ++i) {
        a[i]        = static_cast<TestType>(i + 1);
        b[i]        = static_cast<TestType>(10 * (i + 1));
        expected[i] = static_cast<TestType>(a[i] + b[i]);
    }
    auto out = apply_binop<TestType>(a, b, [](Vec<TestType> x, Vec<TestType> y) { return add(x, y); });
    for (int i = 0; i < N; ++i)
        CHECK(out[i] == expected[i]);
}

TEMPLATE_TEST_CASE("sub: lane-wise difference", "[simd][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> a(N), b(N), expected(N);
    for (int i = 0; i < N; ++i) {
        a[i]        = static_cast<TestType>(100 + i);
        b[i]        = static_cast<TestType>(i + 1);
        expected[i] = static_cast<TestType>(a[i] - b[i]);
    }
    auto out = apply_binop<TestType>(a, b, [](Vec<TestType> x, Vec<TestType> y) { return sub(x, y); });
    for (int i = 0; i < N; ++i)
        CHECK(out[i] == expected[i]);
}

// `mul` is only available for 32-bit integer types on NEON (no 64-bit
// vector multiply); test those paths only.
TEMPLATE_TEST_CASE("mul (32-bit only): lane-wise low product", "[simd][integer]", int32_t, uint32_t) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> a(N), b(N), expected(N);
    for (int i = 0; i < N; ++i) {
        a[i]        = static_cast<TestType>(i + 1);
        b[i]        = static_cast<TestType>(7);
        expected[i] = static_cast<TestType>(a[i] * b[i]);
    }
    auto out = apply_binop<TestType>(a, b, [](Vec<TestType> x, Vec<TestType> y) { return mul(x, y); });
    for (int i = 0; i < N; ++i)
        CHECK(out[i] == expected[i]);
}

// ─── Bitwise ──────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("bitwise_and / bitwise_or / bitwise_xor lane-wise", "[simd][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> a(N), b(N);
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<TestType>(0xF0F0F0F0u + i);
        b[i] = static_cast<TestType>(0x0F0F0F0Fu + 2 * i);
    }
    auto and_out = apply_binop<TestType>(a, b, [](Vec<TestType> x, Vec<TestType> y) { return bitwise_and(x, y); });
    auto or_out  = apply_binop<TestType>(a, b, [](Vec<TestType> x, Vec<TestType> y) { return bitwise_or(x, y); });
    auto xor_out = apply_binop<TestType>(a, b, [](Vec<TestType> x, Vec<TestType> y) { return bitwise_xor(x, y); });
    for (int i = 0; i < N; ++i) {
        CHECK(and_out[i] == static_cast<TestType>(a[i] & b[i]));
        CHECK(or_out[i] == static_cast<TestType>(a[i] | b[i]));
        CHECK(xor_out[i] == static_cast<TestType>(a[i] ^ b[i]));
    }
}

// ─── Shifts ───────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("shift_left<N> and shift_right<N> are logical (zero-fill)", "[simd][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> input(N);
    for (int i = 0; i < N; ++i) {
        // High-bit-set patterns to exercise the logical-vs-arithmetic
        // shift distinction on signed types.
        input[i] = static_cast<TestType>(~TestType(0) - i);
    }
    auto v = loadu<TestType>(input.data());

    {
        auto                  shifted = shift_left<3>(v);
        std::vector<TestType> out(N);
        storeu(out.data(), shifted);
        for (int i = 0; i < N; ++i) {
            // Cast through the unsigned counterpart so signed left-shift
            // overflow doesn't trip undefined-behavior sanitizers.
            using U = std::make_unsigned_t<TestType>;
            CHECK(static_cast<U>(out[i]) == static_cast<U>(static_cast<U>(input[i]) << 3));
        }
    }
    {
        auto                  shifted = shift_right<5>(v);
        std::vector<TestType> out(N);
        storeu(out.data(), shifted);
        for (int i = 0; i < N; ++i) {
            // Logical shift right: result must NOT sign-extend even on signed inputs.
            using U = std::make_unsigned_t<TestType>;
            CHECK(static_cast<U>(out[i]) == static_cast<U>(input[i]) >> 5);
        }
    }
}

// ─── Compare ──────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("cmp_eq returns all-1s mask in matching lanes", "[simd][integer]", int32_t, uint32_t, int64_t, uint64_t) {
    constexpr int         N = Vec<TestType>::lanes;
    std::vector<TestType> a(N), b(N);
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<TestType>(i);
        b[i] = static_cast<TestType>((i % 2 == 0) ? i : i + 1); // even lanes equal, odd lanes differ
    }
    auto                  va    = loadu<TestType>(a.data());
    auto                  vb    = loadu<TestType>(b.data());
    auto                  vmask = cmp_eq(va, vb);
    std::vector<TestType> mask(N);
    storeu(mask.data(), vmask);
    using U = std::make_unsigned_t<TestType>;
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0)
            CHECK(static_cast<U>(mask[i]) == ~U(0));
        else
            CHECK(static_cast<U>(mask[i]) == U(0));
    }
}
