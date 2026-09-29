//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Stream.cpp
/// @brief stream_contract, and the route that sends a GEMV-shaped contraction with interleaved axes
///        to it: every term against a brute-force loop, the privatized and the partitioned walks, a
///        call from inside a parallel region, and the route pins.

#include <Einsums/PackedGemm/PackedGemm.hpp>
#include <Einsums/PackedGemm/Stream.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <cmath>
#include <complex>
#include <random>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

using namespace einsums;
using namespace einsums::index;
using namespace einsums::tensor_algebra;
using einsums::testing::reference_einsum;
namespace pg = einsums::packed_gemm;

namespace {

template <typename T>
std::vector<T> random_values(size_t n, unsigned seed) {
    std::mt19937                           gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<T>                         v(n);
    for (auto &x : v) {
        if constexpr (IsComplexV<T>) {
            x = T{static_cast<typename T::value_type>(dist(gen)), static_cast<typename T::value_type>(dist(gen))};
        } else {
            x = static_cast<T>(dist(gen));
        }
    }
    return v;
}

pg::StreamLayout column_major(std::vector<int64_t> dims) {
    pg::StreamLayout l{.dims = std::move(dims), .strides = {}};
    int64_t          s = 1;
    for (int64_t const d : l.dims) {
        l.strides.push_back(s);
        s *= d;
    }
    return l;
}

// The exchange contraction K(i,j) = c_pf K(i,j) + alpha sum_{k,l} S(i,k,j,l) W(k,l): the output and
// summed axes of S alternate, so no matrix view of S exists.
template <typename T>
struct ExchangeCase {
    int64_t          n;
    pg::StreamLayout s_layout, w_layout, c_layout;
    std::vector<T>   s, w, c, expected;
    T                alpha, c_pf;

    ExchangeCase(int64_t n_, T alpha_, T c_pf_, pg::StreamLayout c_layout_)
        : n(n_), s_layout(column_major({n_, n_, n_, n_})), w_layout(column_major({n_, n_})), c_layout(std::move(c_layout_)), alpha(alpha_),
          c_pf(c_pf_) {
        s            = random_values<T>(static_cast<size_t>(n * n * n * n), 1);
        w            = random_values<T>(static_cast<size_t>(n * n), 2);
        int64_t span = 1;
        for (size_t d = 0; d < c_layout.dims.size(); d++) {
            span += (c_layout.dims[d] - 1) * c_layout.strides[d];
        }
        c        = random_values<T>(static_cast<size_t>(span), 3);
        expected = c;
        for (int64_t j = 0; j < n; j++) {
            for (int64_t i = 0; i < n; i++) {
                T sum{0};
                for (int64_t l = 0; l < n; l++) {
                    for (int64_t k = 0; k < n; k++) {
                        sum += s[i + n * (k + n * (j + n * l))] * w[k + n * l];
                    }
                }
                T &e = expected[i * c_layout.strides[0] + j * c_layout.strides[1]];
                e    = c_pf * e + alpha * sum;
            }
        }
    }

    pg::StreamTerm<T> term() {
        return {.c        = c.data(),
                .c_layout = c_layout,
                .w        = w.data(),
                .w_layout = w_layout,
                .c_axis   = {0, -1, 1, -1},
                .w_axis   = {-1, 0, -1, 1},
                .alpha    = alpha,
                .c_pf     = c_pf};
    }

    // Every element, including the gaps of a strided output, which must be untouched.
    void check() const {
        using R = RemoveComplexT<T>;
        for (size_t e = 0; e < c.size(); e++) {
            REQUIRE(std::abs(c[e] - expected[e]) <= R(100) * std::numeric_limits<R>::epsilon() * static_cast<R>(n * n));
        }
    }
};

} // namespace

TEMPLATE_TEST_CASE("stream_contract - interleaved GEMV-shaped term matches the loop", "[PackedGemm][Stream]", float, double,
                   std::complex<float>, std::complex<double>) {
    int64_t const n = 12;
    SECTION("privatized output, scaled prior contents") {
        ExchangeCase<TestType> tc(n, TestType{0.75}, TestType{-0.5}, column_major({n, n}));
        pg::stream_contract<TestType>(tc.s.data(), tc.s_layout, {tc.term()}, {});
        tc.check();
    }
    SECTION("partitioned walk writing the output in place") {
        ExchangeCase<TestType> tc(n, TestType{2}, TestType{0}, column_major({n, n}));
        // Both output axes of S are offered; the kernel partitions the one with the larger stride.
        pg::stream_contract<TestType>(tc.s.data(), tc.s_layout, {tc.term()}, {0, 2});
        tc.check();
    }
    SECTION("strided output keeps its gaps") {
        // Column stride 2n: every other column of a wider parent.
        ExchangeCase<TestType> tc(n, TestType{1}, TestType{1}, pg::StreamLayout{.dims = {n, n}, .strides = {1, 2 * n}});
        pg::stream_contract<TestType>(tc.s.data(), tc.s_layout, {tc.term()}, {});
        tc.check();
    }
}

// Defends the segfault in the fused-stream kernel: it split the work by omp_get_max_threads() and
// then assumed the region had forked that wide. Called from inside another parallel region, the
// region does not fork, only thread 0 ran, and the reduction read the private buffers of the threads
// that never existed.
TEST_CASE("stream_contract - a call from inside a parallel region computes every element", "[PackedGemm][Stream]") {
    int64_t const n = 12;
    for (bool const partition : {false, true}) {
        ExchangeCase<double> tc(n, 1.0, 0.0, column_major({n, n}));
        auto const           term = tc.term();
#ifdef _OPENMP
#    pragma omp parallel num_threads(2)
#    pragma omp single
#endif
        {
            pg::stream_contract<double>(tc.s.data(), tc.s_layout, {term}, partition ? std::vector<int>{0, 2} : std::vector<int>{});
        }
        tc.check();
    }
}

TEST_CASE("stream_contract - terms sharing an output accumulate", "[PackedGemm][Stream]") {
    int64_t const        n = 10;
    ExchangeCase<double> k(n, -1.0, 0.0, column_major({n, n}));

    // J(i,j) = 2 sum S(i,j,k,l) W(k,l), added into the same output as K.
    std::vector<double> expected = k.expected;
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < n; i++) {
            double sum = 0;
            for (int64_t l = 0; l < n; l++) {
                for (int64_t kk = 0; kk < n; kk++) {
                    sum += k.s[i + n * (j + n * (kk + n * l))] * k.w[kk + n * l];
                }
            }
            expected[i + n * j] += 2.0 * sum;
        }
    }
    auto j_term   = k.term();
    j_term.c_axis = {0, 1, -1, -1};
    j_term.w_axis = {-1, -1, 0, 1};
    j_term.alpha  = 2.0;
    j_term.c_pf   = 1.0;

    pg::stream_contract<double>(k.s.data(), k.s_layout, {k.term(), j_term}, {});
    for (size_t e = 0; e < expected.size(); e++) {
        REQUIRE_THAT(k.c[e], Catch::Matchers::WithinAbs(expected[e], 1e-11));
    }

    SECTION("a later term that rescales the output is rejected") {
        j_term.c_pf = 0.5;
        REQUIRE_THROWS_AS(pg::stream_contract<double>(k.s.data(), k.s_layout, {k.term(), j_term}, {}), std::invalid_argument);
    }
}

TEST_CASE("stream_contract - an empty stream still applies the output prefactor", "[PackedGemm][Stream]") {
    std::vector<double>          c{1.0, 2.0, 3.0, 4.0};
    pg::StreamTerm<double> const term{.c        = c.data(),
                                      .c_layout = column_major({2, 2}),
                                      .w        = nullptr,
                                      .w_layout = column_major({0}),
                                      .c_axis   = {0, -1, 1},
                                      .w_axis   = {-1, 0, -1},
                                      .alpha    = 1.0,
                                      .c_pf     = 3.0};
    pg::stream_contract<double>(nullptr, column_major({2, 0, 2}), {term}, {});
    REQUIRE(c == std::vector<double>{3.0, 6.0, 9.0, 12.0});
}

TEMPLATE_TEST_CASE("Stream route - the exchange contraction streams and matches the reference", "[PackedGemm][Stream]", float, double,
                   std::complex<float>, std::complex<double>) {
    size_t const n   = 16; // 65536 elements in the streamed operand, above kStreamMinElems
    auto         TEI = create_random_tensor<TestType>("TEI", n, n, n, n);
    auto         D   = create_random_tensor<TestType>("D", n, n);

    Tensor<TestType, 2> K("K", n, n), K_ref("K_ref", n, n);
    K.zero();
    K_ref.zero();
    reference_einsum("ij <- ikjl ; kl", TestType{0}, &K_ref, TestType{-1}, TEI, D);

    pg::last_contraction_route() = "none";
    einsum(0.0, Indices{i, j}, &K, -1.0, Indices{i, k, j, l}, TEI, Indices{k, l}, D);
    REQUIRE(std::string(pg::last_contraction_route()) == "stream");

    using R = RemoveComplexT<TestType>;
    for (size_t a = 0; a < n; a++) {
        for (size_t b = 0; b < n; b++) {
            REQUIRE(std::abs(K(a, b) - K_ref(a, b)) <= R(1000) * std::numeric_limits<R>::epsilon() * static_cast<R>(n * n));
        }
    }
}

TEST_CASE("Stream route - shapes it must leave alone", "[PackedGemm][Stream]") {
    SECTION("a contiguous GEMV stays on the direct gemv") {
        size_t const      n   = 16;
        auto              TEI = create_random_tensor<double>("TEI", n, n, n, n);
        auto              D   = create_random_tensor<double>("D", n, n);
        Tensor<double, 2> J("J", n, n), J_ref("J_ref", n, n);
        J_ref.zero();
        reference_einsum("ij <- ijkl ; kl", 0.0, &J_ref, 2.0, TEI, D);
        pg::last_contraction_route() = "none";
        einsum(0.0, Indices{i, j}, &J, 2.0, Indices{i, j, k, l}, TEI, Indices{k, l}, D);
        REQUIRE(std::string(pg::last_contraction_route()) != "stream");
        for (size_t a = 0; a < n; a++) {
            for (size_t b = 0; b < n; b++) {
                REQUIRE_THAT(J(a, b), Catch::Matchers::WithinAbs(J_ref(a, b), 1e-10));
            }
        }
    }
    SECTION("below kStreamMinElems") {
        size_t const      n   = 6; // 1296 elements
        auto              TEI = create_random_tensor<double>("TEI", n, n, n, n);
        auto              D   = create_random_tensor<double>("D", n, n);
        Tensor<double, 2> K("K", n, n), K_ref("K_ref", n, n);
        K_ref.zero();
        reference_einsum("ij <- ikjl ; kl", 0.0, &K_ref, -1.0, TEI, D);
        pg::last_contraction_route() = "none";
        einsum(0.0, Indices{i, j}, &K, -1.0, Indices{i, k, j, l}, TEI, Indices{k, l}, D);
        REQUIRE(std::string(pg::last_contraction_route()) != "stream");
        for (size_t a = 0; a < n; a++) {
            for (size_t b = 0; b < n; b++) {
                REQUIRE_THAT(K(a, b), Catch::Matchers::WithinAbs(K_ref(a, b), 1e-12));
            }
        }
    }
}
