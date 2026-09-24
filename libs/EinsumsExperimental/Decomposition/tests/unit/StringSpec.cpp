//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The string-spec copies of CP and Tucker against the originals, which spell their contractions
// with compile-time index types. Each copy must meet the original's accuracy bound on the
// original's inputs and agree with it; the unfolding and the Khatri-Rao product each copy is built
// on are checked against the templated ones element by element.

#include <Einsums/ComputeGraph/KhatriRao.hpp>
#include <Einsums/Decomposition/CP.hpp>
#include <Einsums/Decomposition/StringSpec/CP.hpp>
#include <Einsums/Decomposition/StringSpec/Tucker.hpp>
#include <Einsums/Decomposition/Tucker.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/RMSD.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace ss = einsums::decomposition::string_spec;

namespace {

// The inputs of CP.cpp and Tucker.cpp. CP.cpp fills them row-major and Tucker.cpp column-major,
// and each file's accuracy bounds hold for its own layout.
template <typename T>
Tensor<T, 3> test_tensor_333(bool row_major) {
    auto t          = create_tensor<T>(row_major, "test 1", 3, 3, 3);
    t.vector_data() = {0.94706517, 0.3959549,  0.14122476, 0.83665482, 0.27340639, 0.29811429, 0.1823041,  0.66556282, 0.73178046,
                       0.72504222, 0.58360409, 0.68301135, 0.8316929,  0.66955444, 0.25182224, 0.24108674, 0.09582611, 0.93056666,
                       0.60919366, 0.97363788, 0.24531965, 0.23757898, 0.43426057, 0.64763913, 0.61224901, 0.86068415, 0.12051599};
    return t;
}

template <typename T>
Tensor<T, 3> test_tensor_342(bool row_major) {
    auto t          = create_tensor<T>(row_major, "test 2", 3, 4, 2);
    t.vector_data() = {0.29945093, 0.0090937,  0.99788559, 0.0821231,  0.29625705, 0.80278977, 0.15189681, 0.35832086,
                       0.09648153, 0.39398175, 0.49662056, 0.83101396, 0.84288292, 0.48603425, 0.93286471, 0.47101289,
                       0.32736096, 0.50067919, 0.49932342, 0.91922942, 0.44777189, 0.23009644, 0.34874549, 0.19356636};
    return t;
}

template <typename T>
Tensor<T, 4> test_tensor_3232(bool row_major) {
    auto t          = create_tensor<T>(row_major, "test 3", 3, 2, 3, 2);
    t.vector_data() = {0.37001224, 0.77676895, 0.17589323, 0.02762156, 0.21037116, 0.83686174, 0.35042434, 0.19117270, 0.58095640,
                       0.99220655, 0.33536840, 0.15210615, 0.95033534, 0.73212124, 0.31346639, 0.83961596, 0.15418801, 0.58927303,
                       0.46744825, 0.44001279, 0.50372353, 0.09696069, 0.96449749, 0.71151666, 0.72334792, 0.98646368, 0.13764230,
                       0.95949904, 0.07774470, 0.18239083, 0.82591821, 0.40939436, 0.22088749, 0.90281597, 0.37465773, 0.02541923};
    return t;
}

template <typename T>
constexpr double agreement = std::is_same_v<T, float> ? 1.0e-4 : 1.0e-10;

template <typename T, size_t Rank>
void check_parafac(Tensor<T, Rank> const &tensor, size_t rank, double bound) {
    auto const original = decomposition::parafac(tensor, rank, 50, 1.0e-6);
    auto const copy     = ss::parafac(tensor, rank, 50, 1.0e-6);

    double const original_error = rmsd(tensor, decomposition::parafac_reconstruct<Rank>(original));
    double const copy_error     = rmsd(tensor, ss::parafac_reconstruct<Rank>(copy));
    CAPTURE(original_error, copy_error);
    CHECK(copy_error <= bound);
    CHECK(std::abs(copy_error - original_error) <= agreement<T>);
}

template <typename T, size_t Rank>
void check_tucker(Tensor<T, Rank> const &tensor, std::vector<size_t> ranks, double svd_bound, double oi_bound) {
    {
        auto [g_original, f_original] = decomposition::tucker_ho_svd(tensor, ranks);
        auto [g_copy, f_copy]         = ss::tucker_ho_svd(tensor, ranks);
        double const original_error   = rmsd(tensor, decomposition::tucker_reconstruct(g_original, f_original));
        double const copy_error       = rmsd(tensor, ss::tucker_reconstruct(g_copy, f_copy));
        CAPTURE(original_error, copy_error);
        CHECK(copy_error <= svd_bound);
        CHECK(std::abs(copy_error - original_error) <= agreement<T>);
    }
    {
        auto [g_original, f_original] = decomposition::tucker_ho_oi(tensor, ranks, 50, 1.0e-6);
        auto [g_copy, f_copy]         = ss::tucker_ho_oi(tensor, ranks, 50, 1.0e-6);
        double const original_error   = rmsd(tensor, decomposition::tucker_reconstruct(g_original, f_original));
        double const copy_error       = rmsd(tensor, ss::tucker_reconstruct(g_copy, f_copy));
        CAPTURE(original_error, copy_error);
        CHECK(copy_error <= oi_bound);
        CHECK(std::abs(copy_error - original_error) <= agreement<T>);
    }
}

} // namespace

TEMPLATE_TEST_CASE("string spec - the unfolding matches the templated one", "[decomposition][string-spec]", float, double) {
    auto const tensor = test_tensor_3232<TestType>(false);
    for_sequence<4>([&](auto mode) {
        CAPTURE(static_cast<size_t>(mode));
        auto const original = tensor_algebra::unfold<mode>(tensor);
        auto const copy     = ss::detail::unfolded(mode, tensor);
        REQUIRE(copy.dim(0) == original.dim(0));
        REQUIRE(copy.dim(1) == original.dim(1));
        for (size_t row = 0; row < original.dim(0); ++row) {
            for (size_t col = 0; col < original.dim(1); ++col) {
                CHECK(copy(row, col) == original(row, col));
            }
        }
    });
}

TEMPLATE_TEST_CASE("string spec - the Khatri-Rao product matches the templated one", "[decomposition][string-spec]", float, double) {
    using namespace einsums::index;
    auto const A = create_random_tensor<TestType>("A", 4, 3);
    auto const B = create_random_tensor<TestType>("B", 5, 3);

    auto const original = tensor_algebra::khatri_rao(Indices{I, r}, A, Indices{M, r}, B);
    auto const copy     = compute_graph::khatri_rao("ir", A, "mr", B);
    REQUIRE(copy.dim(0) == original.dim(0));
    REQUIRE(copy.dim(1) == original.dim(1));
    for (size_t row = 0; row < original.dim(0); ++row) {
        for (size_t col = 0; col < original.dim(1); ++col) {
            CHECK(std::abs(copy(row, col) - original(row, col)) <= agreement<TestType> * (1.0 + std::abs(original(row, col))));
        }
    }
}

TEMPLATE_TEST_CASE("string spec - CP matches the original", "[decomposition][string-spec]", float, double) {
    SECTION("3 x 3 x 3") {
        check_parafac(test_tensor_333<TestType>(true), 2, 0.17392);
    }
    SECTION("3 x 4 x 2") {
        check_parafac(test_tensor_342<TestType>(true), 2, 0.122492);
    }
    SECTION("3 x 2 x 3 x 2") {
        check_parafac(test_tensor_3232<TestType>(true), 2, 0.228200);
    }
}

TEST_CASE("string spec - CP at rank 24 reconstructs the tensor", "[decomposition][string-spec]") {
    // CP.cpp's "CP 4": an over-complete decomposition, which only a correctly oriented update solves.
    auto const tensor = test_tensor_342<double>(true);
    auto const copy   = ss::parafac(tensor, 24, 100, 1.0e-6);
    CHECK(rmsd(tensor, ss::parafac_reconstruct<3>(copy)) <= 1.0e-4);
}

TEST_CASE("string spec - weighted CP matches the original", "[decomposition][string-spec]") {
    auto const tensor  = test_tensor_342<double>(true);
    auto       weights = create_tensor<double>("weights", 3);
    weights(0)         = 1.0;
    weights(1)         = 0.5;
    weights(2)         = 2.0;

    auto const   original       = decomposition::weighted_parafac(tensor, weights, 2, 50, 1.0e-6);
    auto const   copy           = ss::weighted_parafac(tensor, weights, 2, 50, 1.0e-6);
    double const original_error = rmsd(tensor, decomposition::parafac_reconstruct<3>(original));
    double const copy_error     = rmsd(tensor, ss::parafac_reconstruct<3>(copy));
    CAPTURE(original_error, copy_error);
    CHECK(std::abs(copy_error - original_error) <= 1.0e-10);
}

TEMPLATE_TEST_CASE("string spec - Tucker matches the original", "[decomposition][string-spec]", float, double) {
    SECTION("3 x 3 x 3") {
        check_tucker(test_tensor_333<TestType>(false), {2, 2, 2}, 0.178837, 0.173911);
    }
    SECTION("3 x 4 x 2") {
        check_tucker(test_tensor_342<TestType>(false), {2, 3, 2}, 0.110250, 0.108301);
    }
    SECTION("3 x 2 x 3 x 2") {
        check_tucker(test_tensor_3232<TestType>(false), {2, 2, 2, 2}, 0.196843, 0.192402);
    }
}

TEST_CASE("string spec - CP reports a singular ALS system", "[decomposition][string-spec]") {
    // The same guard as the original's: a zero tensor zeroes the first factor, and the next mode's
    // normal-equations matrix is exactly singular.
    auto zero = create_tensor<double>("zero", 3, 4, 2);
    zero.zero();
    CHECK_THROWS_AS(ss::parafac(zero, 2, 10, 1.0e-6), std::runtime_error);
}
