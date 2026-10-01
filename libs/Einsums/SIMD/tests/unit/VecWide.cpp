//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Vec<T, N> and Mask<T, N> wider than one register: moving between the FP32 and FP64 tiers of a
// mixed-precision kernel lane for lane (convert and mask_cast across widths), table lookup with an
// index of either width, masked memory at every tail length, and the two tiers of a Boys-style
// interpolation sharing one index vector. The elementwise operations are checked against the scalar
// instantiation in GenericKernel.

#include <Einsums/SIMD/Generic.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <catch2/catch_all.hpp>

#if defined(__linux__)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace simd = einsums::simd;

namespace {

constexpr int LF = simd::lanes<float>;

/// The FP64 tier at the FP32 tier's width.
using DoubleTier = simd::Vec<double, LF>;

template <typename T>
using bits_t = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;

template <typename T>
bool same(T got, T want) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::bit_cast<bits_t<T>>(got) == std::bit_cast<bits_t<T>>(want) || (std::isnan(got) && std::isnan(want));
    } else {
        return got == want;
    }
}

template <typename T, int N>
concept valid_vec = requires { sizeof(simd::Vec<T, N>); };

} // namespace

TEST_CASE("Vec<T> is the native vector, and wide lane counts are whole registers", "[simd][wide]") {
    STATIC_CHECK(std::is_same_v<simd::Vec<float>, simd::Vec<float, LF>>);
    STATIC_CHECK(std::is_same_v<simd::Mask<double>, simd::Mask<double, simd::lanes<double>>>);
    STATIC_CHECK(simd::Vec<float>::native);
    STATIC_CHECK(DoubleTier::lanes == LF);
    STATIC_CHECK(DoubleTier::native == (simd::lanes<double> == LF));
    STATIC_CHECK(sizeof(simd::Vec<double, 4 * simd::lanes<double>>) == 4 * sizeof(simd::Vec<double>));
    STATIC_CHECK(valid_vec<float, 3 * LF>);
    STATIC_CHECK_FALSE(valid_vec<float, 0>);
    if constexpr (LF > 1) {
        STATIC_CHECK_FALSE(valid_vec<float, LF + 1>);
    }
}

TEST_CASE("convert moves the same lanes between float and double", "[simd][wide][convert]") {
    // Widening is exact for every float; narrowing rounds to nearest, ties to even, as a C++ cast.
    std::vector<float> values{0.0f,
                              -0.0f,
                              1.0f,
                              -1.5f,
                              3.0e38f,
                              std::numeric_limits<float>::denorm_min(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN(),
                              0.1f,
                              16777217.0f};
    for (int i = 0; i < 64; ++i)
        values.push_back(static_cast<float>(i * i) * 0.37f - 300.0f);
    for (size_t o = 0; o < values.size(); ++o) {
        float in[LF], back[LF];
        for (int i = 0; i < LF; ++i)
            in[i] = values[(o + static_cast<size_t>(i)) % values.size()];
        DoubleTier const d = simd::convert<double>(simd::load<simd::Vec<float>>(in));
        double           wide[LF];
        simd::store(wide, d);
        simd::store(back, simd::convert<float>(d));
        for (int i = 0; i < LF; ++i) {
            INFO("lane " << i << ": " << in[i]);
            CHECK(same(wide[i], static_cast<double>(in[i])));
            CHECK(same(back[i], in[i]));
        }
    }

    std::vector<double> wides{
        1.0 + 0x1p-24, 1.0 + 3 * 0x1p-24, -(1.0 + 0x1p-24), 3.5e38, -3.5e38, 1e-50, -1e-50, std::numeric_limits<double>::quiet_NaN(), 0.1,
        -0.0,          1.0 / 3.0};
    for (size_t o = 0; o < wides.size(); ++o) {
        double in[LF];
        float  out[LF];
        for (int i = 0; i < LF; ++i)
            in[i] = wides[(o + static_cast<size_t>(i)) % wides.size()];
        simd::store(out, simd::convert<float>(simd::load<DoubleTier>(in)));
        for (int i = 0; i < LF; ++i) {
            INFO("lane " << i << ": " << in[i]);
            CHECK(same(out[i], static_cast<float>(in[i])));
        }
    }
}

TEST_CASE("convert moves the same lanes between double and the integers", "[simd][wide][convert]") {
    double in[LF];
    for (int i = 0; i < LF; ++i)
        in[i] = (i % 2 ? -1.0 : 1.0) * (static_cast<double>(i) * 1000.5 + 0.75);
    DoubleTier const d = simd::load<DoubleTier>(in);

    int32_t i32[LF];
    double  from32[LF];
    simd::store(i32, simd::convert<int32_t>(d));
    simd::store(from32, simd::convert<double>(simd::load<simd::Vec<int32_t, LF>>(i32)));
    for (int i = 0; i < LF; ++i) {
        CHECK(i32[i] == static_cast<int32_t>(in[i]));
        CHECK(from32[i] == static_cast<double>(i32[i]));
    }

    int64_t i64[LF];
    double  from64[LF];
    simd::store(i64, simd::convert<int64_t>(d));
    simd::store(from64, simd::convert<double>(simd::load<simd::Vec<int64_t, LF>>(i64)));
    for (int i = 0; i < LF; ++i) {
        CHECK(i64[i] == static_cast<int64_t>(in[i]));
        CHECK(from64[i] == static_cast<double>(i64[i]));
    }
}

TEST_CASE("mask_cast keeps lane i in lane i across widths", "[simd][wide][mask]") {
    // Every pattern for small lane counts, a spread of them otherwise.
    for (uint64_t bits = 0; bits < (uint64_t{1} << LF) && bits < 70000; bits = LF <= 8 ? bits + 1 : bits * 5 + 1) {
        simd::Mask<float> const      f = simd::mask_from_bits<float>(bits);
        simd::Mask<double, LF> const d = simd::mask_cast<double>(f);
        INFO("bits " << bits);
        CHECK(simd::to_bits(d) == bits);
        CHECK(simd::to_bits(simd::mask_cast<float>(d)) == bits);
        CHECK(simd::to_bits(simd::mask_cast<int64_t>(simd::mask_cast<int32_t>(f))) == bits);
        CHECK(simd::to_bits(simd::mask_cast<int32_t>(simd::mask_cast<int64_t>(simd::mask_cast<int32_t>(f)))) == bits);
    }
    // A comparison in one tier steers the other.
    float  x[LF];
    double y[LF], out[LF];
    for (int i = 0; i < LF; ++i) {
        x[i] = static_cast<float>(i) - 2.5f;
        y[i] = static_cast<double>(i);
    }
    auto const steer = simd::mask_cast<double>(simd::cmp_lt(simd::load<simd::Vec<float>>(x), simd::splat<simd::Vec<float>>(0.0f)));
    simd::store(out, simd::select(steer, simd::load<DoubleTier>(y), simd::splat<DoubleTier>(-1.0)));
    for (int i = 0; i < LF; ++i)
        CHECK(out[i] == (x[i] < 0.0f ? y[i] : -1.0));
}

TEMPLATE_TEST_CASE("first_n, to_bits and the reductions span the parts of a wide mask", "[simd][wide][mask]", float, double, int32_t,
                   int64_t) {
    using T         = TestType;
    constexpr int N = 4 * simd::lanes<T>;
    for (std::size_t n = 0; n <= static_cast<std::size_t>(N) + 1; ++n) {
        auto const     m    = simd::first_n<T, N>(n);
        std::size_t    set  = n < static_cast<std::size_t>(N) ? n : static_cast<std::size_t>(N);
        uint64_t const want = set >= 64 ? ~uint64_t{0} : (uint64_t{1} << set) - 1u;
        INFO("n = " << n);
        CHECK(simd::to_bits(m) == want);
        CHECK(simd::count(m) == static_cast<int>(set));
        CHECK(simd::any(m) == (set > 0));
        CHECK(simd::all(m) == (set == static_cast<std::size_t>(N)));
        CHECK(simd::to_bits(simd::mask_from_bits<T, N>(want)) == want);
    }
}

TEMPLATE_TEST_CASE("lookup on a wide vector takes either index width", "[simd][wide][gather]", float, double) {
    using T          = TestType;
    constexpr int  N = 2 * LF;
    std::vector<T> table(1000);
    for (size_t k = 0; k < table.size(); ++k)
        table[k] = T(0.5) * static_cast<T>(k) - T(9);
    int32_t at32[N];
    int64_t at64[N];
    for (int i = 0; i < N; ++i) {
        at32[i] = static_cast<int32_t>((i * 389 + 17) % 1000);
        at64[i] = at32[i];
    }
    T out[N];
    simd::store(out, simd::lookup(table.data(), simd::load<simd::Vec<int32_t, N>>(at32)));
    for (int i = 0; i < N; ++i)
        CHECK(out[i] == table[static_cast<size_t>(at32[i])]);
    simd::store(out, simd::lookup(table.data(), simd::load<simd::Vec<int64_t, N>>(at64)));
    for (int i = 0; i < N; ++i)
        CHECK(out[i] == table[static_cast<size_t>(at64[i])]);

    // Masked: inactive lanes' indices are out of range and must never be read.
    for (int i = 1; i < N; i += 2)
        at32[i] = 1 << 28;
    auto const m = simd::mask_from_bits<T, N>(0x5555555555555555ull);
    simd::store(out, simd::lookup(table.data(), simd::load<simd::Vec<int32_t, N>>(at32), m));
    for (int i = 0; i < N; ++i)
        CHECK(out[i] == (i % 2 == 0 ? table[static_cast<size_t>(at32[i])] : T(0)));
}

#if defined(__linux__)
TEMPLATE_TEST_CASE("wide masked loads and stores never touch an inactive lane's memory", "[simd][wide][mask]", float, double) {
    using T                = TestType;
    constexpr int     N    = 2 * LF;
    std::size_t const page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    void *const       base = mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(base != MAP_FAILED);
    REQUIRE(mprotect(static_cast<char *>(base) + page, page, PROT_NONE) == 0);
    for (std::size_t n = 0; n <= static_cast<std::size_t>(N); ++n) {
        INFO("n = " << n);
        T *const p = reinterpret_cast<T *>(static_cast<char *>(base) + page) - n;
        for (std::size_t i = 0; i < n; ++i)
            p[i] = static_cast<T>(i + 1);
        auto const m = simd::first_n<T, N>(n);
        T          out[N];
        simd::store(out, simd::loadu(p, m));
        for (int i = 0; i < N; ++i)
            CHECK(out[i] == (static_cast<std::size_t>(i) < n ? static_cast<T>(i + 1) : T(0)));
        simd::storeu(p, simd::splat<simd::Vec<T, N>>(T(7)), m);
        for (std::size_t i = 0; i < n; ++i)
            CHECK(p[i] == T(7));
    }
    munmap(base, 2 * page);
}
#endif

TEST_CASE("the two tiers of an interpolation share one index vector", "[simd][wide]") {
    // The FP32 tier computes the grid index once; the FP64 tier, at the same width, gathers its doubles
    // with that index and evaluates the same Taylor series in double precision.
    float const         dx = 0.125f, inv_dx = 8.0f;
    std::vector<float>  f0(400), f1(400);
    std::vector<double> d0(400), d1(400);
    for (size_t k = 0; k < 400; ++k) {
        double const g = static_cast<double>(k) * dx;
        d0[k]          = std::exp(-g);
        d1[k]          = -d0[k];
        f0[k]          = static_cast<float>(d0[k]);
        f1[k]          = static_cast<float>(d1[k]);
    }
    for (int start = 0; start < 30; ++start) {
        float x[LF];
        for (int i = 0; i < LF; ++i)
            x[i] = 0.0131f * static_cast<float>((start * 97 + i * 31) % 3700);
        simd::Vec<float> const   xf   = simd::load<simd::Vec<float>>(x);
        simd::Vec<float> const   grid = simd::floor(xf * inv_dx);
        simd::Vec<int32_t> const idx  = simd::convert<int32_t>(grid);

        // FP32 tier.
        simd::Vec<float> const df = simd::fnmadd(grid, simd::splat<simd::Vec<float>>(dx), xf);
        float                  r32[LF];
        simd::store(r32, simd::fmadd(simd::lookup(f1.data(), idx), df, simd::lookup(f0.data(), idx)));

        // FP64 tier, same lanes, same index register.
        DoubleTier const xd = simd::convert<double>(xf);
        DoubleTier const dd = simd::fnmadd(simd::convert<double>(grid), simd::splat<DoubleTier>(dx), xd);
        double           r64[LF];
        simd::store(r64, simd::fmadd(simd::lookup(d1.data(), idx), dd, simd::lookup(d0.data(), idx)));

        for (int i = 0; i < LF; ++i) {
            INFO("x = " << x[i]);
            double const g    = std::floor(static_cast<double>(x[i]) * inv_dx);
            double const want = d0[static_cast<size_t>(g)] + d1[static_cast<size_t>(g)] * (static_cast<double>(x[i]) - g * dx);
            CHECK(r64[i] == Catch::Approx(want).epsilon(1e-14));
            CHECK(static_cast<double>(r32[i]) == Catch::Approx(r64[i]).epsilon(1e-6));
        }
    }
}
