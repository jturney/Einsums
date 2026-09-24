//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// cg::khatri_rao. The expected values are reference_einsum's rank-N product, reshaped into the
// matrix by hand from the definition, so neither the library's einsum nor its view arithmetic
// checks itself.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/KhatriRao.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <complex>
#include <stdexcept>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;
using einsums::testing::reference_einsum;

namespace {

template <typename T>
constexpr double tolerance = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>> ? 1.0e-5 : 1.0e-12;

template <typename T>
void check_close(T got, T want) {
    CHECK(std::abs(got - want) <= tolerance<T> * (1.0 + std::abs(want)));
}

} // namespace

TEMPLATE_TEST_CASE("khatri_rao - one shared index is the column-wise Kronecker product", "[ComputeGraph][khatri-rao]", float, double,
                   std::complex<float>, std::complex<double>) {
    size_t const I = 4, M = 3, R = 5;
    auto const   A = create_random_tensor<TestType>("A", I, R);
    auto const   B = create_random_tensor<TestType>("B", M, R);

    auto const KR = cg::khatri_rao("ir", A, "mr", B);
    REQUIRE(KR.dim(0) == I * M);
    REQUIRE(KR.dim(1) == R);

    auto expected = create_zero_tensor<TestType>("expected", I, M, R);
    reference_einsum("imr <- ir ; mr", &expected, A, B);
    for (size_t i = 0; i < I; ++i) {
        for (size_t m = 0; m < M; ++m) {
            for (size_t r = 0; r < R; ++r) {
                check_close(KR(i + I * m, r), expected(i, m, r));
            }
        }
    }
}

TEST_CASE("khatri_rao - index lists follow the einsum operand grammar", "[ComputeGraph][khatri-rao]") {
    auto const A = create_random_tensor<double>("A", 4, 5);
    auto const B = create_random_tensor<double>("B", 3, 5);

    auto const chars = cg::khatri_rao("ir", A, "mr", B);
    auto const words = cg::khatri_rao("mu,rho", A, "nu,rho", B);
    REQUIRE(words.dim(0) == chars.dim(0));
    REQUIRE(words.dim(1) == chars.dim(1));
    for (size_t row = 0; row < chars.dim(0); ++row) {
        for (size_t col = 0; col < chars.dim(1); ++col) {
            CHECK(words(row, col) == chars(row, col));
        }
    }
}

TEST_CASE("khatri_rao - the general form", "[ComputeGraph][khatri-rao]") {
    SECTION("two shared indices make two column axes") {
        size_t const I = 3, M = 2, R = 4, S = 2;
        auto const   A = create_random_tensor<double>("A", I, R, S);
        auto const   B = create_random_tensor<double>("B", M, R, S);

        auto const KR = cg::khatri_rao("irs", A, "mrs", B);
        REQUIRE(KR.dim(0) == I * M);
        REQUIRE(KR.dim(1) == R * S);

        auto expected = create_zero_tensor<double>("expected", I, M, R, S);
        reference_einsum("imrs <- irs ; mrs", &expected, A, B);
        for (size_t i = 0; i < I; ++i) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t r = 0; r < R; ++r) {
                    for (size_t s = 0; s < S; ++s) {
                        check_close(KR(i + I * m, r + R * s), expected(i, m, r, s));
                    }
                }
            }
        }
    }

    SECTION("no shared index is the outer product, as one column") {
        size_t const I = 3, M = 4;
        auto const   A = create_random_tensor<double>("A", I);
        auto const   B = create_random_tensor<double>("B", M);

        auto const KR = cg::khatri_rao("i", A, "m", B);
        REQUIRE(KR.dim(0) == I * M);
        REQUIRE(KR.dim(1) == 1);
        for (size_t i = 0; i < I; ++i) {
            for (size_t m = 0; m < M; ++m) {
                check_close(KR(i + I * m, 0), A(i) * B(m));
            }
        }
    }

    SECTION("an operand of higher rank contributes all its own axes to the rows") {
        size_t const I = 2, J = 3, M = 2, R = 3;
        auto const   A = create_random_tensor<double>("A", I, J, R);
        auto const   B = create_random_tensor<double>("B", M, R);

        auto const KR = cg::khatri_rao("ijr", A, "mr", B);
        REQUIRE(KR.dim(0) == I * J * M);
        for (size_t i = 0; i < I; ++i) {
            for (size_t j = 0; j < J; ++j) {
                for (size_t m = 0; m < M; ++m) {
                    for (size_t r = 0; r < R; ++r) {
                        check_close(KR(i + I * (j + J * m), r), A(i, j, r) * B(m, r));
                    }
                }
            }
        }
    }
}

TEST_CASE("khatri_rao - rejections", "[ComputeGraph][khatri-rao]") {
    auto const A = create_random_tensor<double>("A", 4, 5);
    auto const B = create_random_tensor<double>("B", 3, 5);
    auto const W = create_random_tensor<double>("W", 3, 6);

    CHECK_THROWS_AS(cg::khatri_rao("irs", A, "mr", B), RankError);
    CHECK_THROWS_AS(cg::khatri_rao("ii", A, "mr", B), std::invalid_argument);
    CHECK_THROWS_AS(cg::khatri_rao("i@", A, "mr", B), std::invalid_argument);
    CHECK_THROWS_AS(cg::khatri_rao("ir", A, "mr", W), DimensionError);

    SECTION("during capture, as a returning form") {
        cg::Graph              graph("khatri_rao");
        cg::CaptureGuard const guard(graph);
        CHECK_THROWS_AS(cg::khatri_rao("ir", A, "mr", B), std::logic_error);
    }
}
