//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The stream-fusion inner kernel against a scalar loop, for every stride
// triple it vectorizes and for lengths on both sides of every vector-block
// boundary. The kernel is compiled once per SIMD rung and chosen at run time,
// so the per-rung registrations (EINSUMS_SIMD_ARCH) are what cover the wide
// paths; each rung's vector length and its four-accumulator dot unroll put
// the boundaries in different places, which is why the lengths run past 4
// vectors of the widest rung.

#include <Einsums/ComputeGraph/Passes/StreamKernel.hpp>

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

template <typename T>
T value(int64_t k, int salt) {
    auto const r = static_cast<double>((k * 7 + salt * 13) % 17) / 8.0 - 1.0;
    if constexpr (requires { typename T::value_type; }) {
        using U = typename T::value_type;
        return T(static_cast<U>(r), static_cast<U>(static_cast<double>((k * 5 + salt) % 11) / 4.0 - 1.25));
    } else {
        return static_cast<T>(r);
    }
}

template <typename T>
double tolerance() {
    if constexpr (requires { typename T::value_type; }) {
        return std::is_same_v<typename T::value_type, float> ? 1e-4 : 1e-12;
    } else {
        return std::is_same_v<T, float> ? 1e-4 : 1e-12;
    }
}

template <typename T>
void check_triple(int64_t ds, int64_t dc, int64_t dw) {
    auto const kernel = compute_graph::passes::stream_inner_entry<T>();
    REQUIRE(kernel != nullptr);

    // Offsets keep the streams unaligned and away from the buffer starts.
    int64_t const co = 3, si = 1, wo = 2;
    T const       alpha = value<T>(5, 9);

    for (int64_t n : {0, 1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129}) {
        std::vector<T> s(static_cast<size_t>(si + ds * n + 1)), w(static_cast<size_t>(wo + dw * n + 1)),
            c(static_cast<size_t>(co + dc * n + 1));
        for (size_t k = 0; k < s.size(); ++k)
            s[k] = value<T>(static_cast<int64_t>(k), 1);
        for (size_t k = 0; k < w.size(); ++k)
            w[k] = value<T>(static_cast<int64_t>(k), 2);
        for (size_t k = 0; k < c.size(); ++k)
            c[k] = value<T>(static_cast<int64_t>(k), 3);

        std::vector<T> expected = c;
        for (int64_t i = 0; i < n; ++i) {
            expected[static_cast<size_t>(co + i * dc)] += alpha * s[static_cast<size_t>(si + i * ds)] * w[static_cast<size_t>(wo + i * dw)];
        }

        kernel(c.data(), s.data(), w.data(), alpha, n, co, si, wo, ds, dc, dw);

        for (size_t k = 0; k < c.size(); ++k) {
            INFO("n = " << n << ", element " << k);
            REQUIRE(std::abs(c[k] - expected[k]) <= tolerance<T>() * (1.0 + static_cast<double>(n)));
        }
    }
}

} // namespace

TEMPLATE_TEST_CASE("StreamKernel - every stride triple matches the scalar loop", "[ComputeGraph][StreamKernel]", float, double,
                   std::complex<float>, std::complex<double>) {
    SECTION("(1,1,0) scaled AXPY") {
        check_triple<TestType>(1, 1, 0);
    }
    SECTION("(1,1,1) Hadamard FMA") {
        check_triple<TestType>(1, 1, 1);
    }
    SECTION("(1,0,1) dot reduction") {
        check_triple<TestType>(1, 0, 1);
    }
    SECTION("(2,1,1) strided fallback") {
        check_triple<TestType>(2, 1, 1);
    }
}
