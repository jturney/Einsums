//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase D.1 — runtime-tensor coverage for linear_algebra::gesv (linear solve).
// Mirrors gesv.cpp's static-rank LAPACKE example so we can directly compare
// the runtime overload (added in C.11) against the same reference values.

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <cstring>
#include <tuple>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

template <typename T>
auto runtime_data(RuntimeTensor<T> const &t) -> std::vector<T> {
    std::vector<T> out(t.size());
    std::memcpy(out.data(), t.data(), t.size() * sizeof(T));
    return out;
}

template <typename T>
void fill_runtime(RuntimeTensor<T> &t, std::vector<T> const &flat) {
    REQUIRE(t.size() == flat.size());
    std::memcpy(t.data(), flat.data(), flat.size() * sizeof(T));
}

} // namespace

TEMPLATE_TEST_CASE("RuntimeTensor gesv — LAPACKE example, both layouts produce same X", "[linear-algebra][runtime]", float, double) {
    constexpr int N{5};
    constexpr int NRHS{3};
    using T = TestType;

    // Column-major (matches the static-rank gesv.cpp test's primary path).
    RuntimeTensor<T> a("a", {N, N}, /*row_major=*/false);
    RuntimeTensor<T> b("b", {N, NRHS}, /*row_major=*/false);

    fill_runtime(a, std::vector<T>{T(6.80),  T(-2.11), T(5.66),  T(5.97),  T(8.23),  T(-6.05), T(-3.30), T(5.36), T(-4.44),
                                   T(1.08),  T(-0.45), T(2.58),  T(-2.70), T(0.27),  T(9.04),  T(8.32),  T(2.71), T(4.35),
                                   T(-7.17), T(2.14),  T(-9.67), T(-5.14), T(-7.26), T(6.08),  T(-6.87)});
    fill_runtime(b, std::vector<T>{T(4.02), T(6.19), T(-8.22), T(-7.57), T(-3.03), T(-1.56), T(4.00), T(-8.67), T(1.75), T(2.86), T(9.81),
                                   T(-4.09), T(-4.57), T(-8.61), T(8.99)});

    // Mirror the same data into a row-major pair to confirm both layouts
    // converge on the same numerical solution.
    RuntimeTensor<T> a_rm("a_rm", {N, N}, /*row_major=*/true);
    RuntimeTensor<T> b_rm("b_rm", {N, NRHS}, /*row_major=*/true);
    a_rm = a;
    b_rm = b;

    std::ignore = linear_algebra::gesv(&a, &b);

    CHECK_THAT(runtime_data(b),
               Catch::Matchers::Approx(std::vector<T>{T(-0.80071403), T(-0.69524338), T(0.59391499), T(1.32172561), T(0.56575620),
                                                      T(-0.38962139), T(-0.55442713), T(0.84222739), T(-0.10380185), T(0.10571095),
                                                      T(0.95546491), T(0.22065963), T(1.90063673), T(5.35766149), T(4.04060266)})
                   .margin(0.00001));

    std::ignore = linear_algebra::gesv(&a_rm, &b_rm);

    // The (i, j) entries of the row-major and column-major solutions should match.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < NRHS; ++j) {
            CHECK_THAT(b_rm(i, j), Catch::Matchers::WithinRel(b(i, j), T(1e-5)));
        }
    }
}

TEST_CASE("RuntimeTensor gesv — round-trip A * X = B verification", "[linear-algebra][runtime]") {
    // Solve A @ X = B, then verify A @ X == B numerically. Layout-agnostic
    // since the assertion runs entirely on the runtime tensor's element
    // access (no flat-buffer assumption).
    constexpr int         N    = 4;
    constexpr int         NRHS = 2;
    RuntimeTensor<double> A("A", {N, N}, /*row_major=*/false);
    RuntimeTensor<double> B("B", {N, NRHS}, /*row_major=*/false);
    fill_runtime(A, {2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 5.0});
    fill_runtime(B, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

    // Save originals to verify the round-trip.
    RuntimeTensor<double> A0 = A;
    RuntimeTensor<double> B0 = B;

    std::ignore = linear_algebra::gesv(&A, &B);
    // After gesv, B holds X. Compute A0 @ X and compare against B0.
    RuntimeTensor<double> Bcheck("Bcheck", {N, NRHS}, /*row_major=*/false);
    linear_algebra::gemm<false, false>(1.0, A0, B, 0.0, &Bcheck);

    auto const reconstructed = runtime_data(Bcheck);
    auto const reference     = runtime_data(B0);
    REQUIRE(reconstructed.size() == reference.size());
    for (size_t i = 0; i < reconstructed.size(); ++i) {
        CHECK_THAT(reconstructed[i], Catch::Matchers::WithinRel(reference[i], 1e-9));
    }
}

TEST_CASE("RuntimeTensor gesv — non-square A throws", "[linear-algebra][runtime]") {
    RuntimeTensor<double> A("A", {3, 4});
    RuntimeTensor<double> b("b", {3});
    REQUIRE_THROWS(linear_algebra::gesv(&A, &b));
}
