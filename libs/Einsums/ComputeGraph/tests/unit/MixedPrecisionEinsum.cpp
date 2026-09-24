//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// cg::einsum over operands of different element types, checked against the brute-force reference.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;
using einsums::testing::reference_einsum;

namespace {

template <typename T>
constexpr bool single_precision = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;

/// Tolerance: a float anywhere in the chain limits the agreement to a float's precision.
template <typename TC, typename TA, typename TB>
constexpr double tolerance = (single_precision<TC> || single_precision<TA> || single_precision<TB>) ? 1.0e-5 : 1.0e-12;

template <typename TC, typename TA, typename TB, typename CT>
void check_against(CT const &got, CT const &want) {
    for (size_t n = 0; n < want.size(); ++n) {
        CAPTURE(n);
        CHECK(std::abs(got.data()[n] - want.data()[n]) <= tolerance<TC, TA, TB> * (1.0 + std::abs(want.data()[n])));
    }
}

} // namespace

// Every combination the rules allow whose types are not all the same: 36 of them. A complex
// operand with a real output is the combination left out, because it does not compile.
using MixedTriples =
    std::tuple<std::tuple<float, float, double>, std::tuple<float, double, float>, std::tuple<float, double, double>,
               std::tuple<double, float, float>, std::tuple<double, float, double>, std::tuple<double, double, float>,
               std::tuple<std::complex<float>, float, float>, std::tuple<std::complex<float>, float, double>,
               std::tuple<std::complex<float>, float, std::complex<float>>, std::tuple<std::complex<float>, float, std::complex<double>>,
               std::tuple<std::complex<float>, double, float>, std::tuple<std::complex<float>, double, double>,
               std::tuple<std::complex<float>, double, std::complex<float>>, std::tuple<std::complex<float>, double, std::complex<double>>,
               std::tuple<std::complex<float>, std::complex<float>, float>, std::tuple<std::complex<float>, std::complex<float>, double>,
               std::tuple<std::complex<float>, std::complex<float>, std::complex<double>>,
               std::tuple<std::complex<float>, std::complex<double>, float>, std::tuple<std::complex<float>, std::complex<double>, double>,
               std::tuple<std::complex<float>, std::complex<double>, std::complex<float>>,
               std::tuple<std::complex<float>, std::complex<double>, std::complex<double>>, std::tuple<std::complex<double>, float, float>,
               std::tuple<std::complex<double>, float, double>, std::tuple<std::complex<double>, float, std::complex<float>>,
               std::tuple<std::complex<double>, float, std::complex<double>>, std::tuple<std::complex<double>, double, float>,
               std::tuple<std::complex<double>, double, double>, std::tuple<std::complex<double>, double, std::complex<float>>,
               std::tuple<std::complex<double>, double, std::complex<double>>, std::tuple<std::complex<double>, std::complex<float>, float>,
               std::tuple<std::complex<double>, std::complex<float>, double>,
               std::tuple<std::complex<double>, std::complex<float>, std::complex<float>>,
               std::tuple<std::complex<double>, std::complex<float>, std::complex<double>>,
               std::tuple<std::complex<double>, std::complex<double>, float>,
               std::tuple<std::complex<double>, std::complex<double>, double>,
               std::tuple<std::complex<double>, std::complex<double>, std::complex<float>>>;

TEMPLATE_LIST_TEST_CASE("mixed precision einsum - every allowed combination, eagerly", "[ComputeGraph][MixedPrecision]", MixedTriples) {
    using TC = std::tuple_element_t<0, TestType>;
    using TA = std::tuple_element_t<1, TestType>;
    using TB = std::tuple_element_t<2, TestType>;
    using TR = cg::detail::PromoteT<TA, TB>;

    auto const A        = create_random_tensor<TA>("A", 4, 5);
    auto const B        = create_random_tensor<TB>("B", 5, 3);
    auto       C        = create_random_tensor<TC>("C", 4, 3);
    auto       expected = C;

    cg::einsum("ij <- ik ; kj", TC{0.5}, &C, TR{1.5}, A, B);
    CHECK(std::string(cg::dispatch::last_dispatch_route()) == "generic_loop_mixed_precision");

    reference_einsum("ij <- ik ; kj", TC{0.5}, &expected, TR{1.5}, A, B);
    check_against<TC, TA, TB>(C, expected);
}

TEST_CASE("mixed precision einsum - complex<float> times double keeps both the imaginary part and the precision",
          "[ComputeGraph][MixedPrecision]") {
    // The pair has no operator* between them, and BiggestTypeT would name double for it. The
    // accumulator here is complex<double>.
    using cf = std::complex<float>;
    using cd = std::complex<double>;
    auto a   = create_zero_tensor<cf>("a", 2);
    auto b   = create_zero_tensor<double>("b", 2);
    auto c   = create_zero_tensor<cd>("c", 1);
    a(0)     = cf{1.0f, 2.0f};
    a(1)     = cf{3.0f, -1.0f};
    b(0)     = 1.0 / 3.0;
    b(1)     = 2.0;
    cg::einsum(" <- i ; i", &c, a, b);
    CHECK(c(0).real() == Catch::Approx(1.0 / 3.0 + 6.0).epsilon(1e-15));
    CHECK(c(0).imag() == Catch::Approx(2.0 / 3.0 - 2.0).epsilon(1e-15));
}

TEST_CASE("mixed precision einsum - conjugation, runtime-rank operands and other specs", "[ComputeGraph][MixedPrecision]") {
    using cd = std::complex<double>;
    using cf = std::complex<float>;

    SECTION("conjugating a complex operand, and a no-op on a real one") {
        auto const A        = create_random_tensor<cf>("A", 3, 4);
        auto const B        = create_random_tensor<double>("B", 4, 2);
        auto       C        = create_zero_tensor<cd>("C", 3, 2);
        auto       expected = C;
        cg::einsum("ij <- ik ; kj", cd{0}, &C, cd{1}, A, B, /*conj_a=*/true, /*conj_b=*/true);
        reference_einsum("ij <- ik ; kj", cd{0}, &expected, cd{1}, A, B, true, true);
        check_against<cd, cf, double>(C, expected);
    }

    SECTION("runtime-rank operands") {
        RuntimeTensor<double> A        = create_random_tensor<double>("A", {3, 4});
        RuntimeTensor<float>  B        = create_random_tensor<float>("B", {4, 5});
        RuntimeTensor<double> C        = create_zero_tensor<double>("C", {3, 5});
        auto                  expected = C;
        cg::einsum("ij <- ik ; kj", &C, A, B);
        reference_einsum("ij <- ik ; kj", &expected, A, B);
        check_against<double, double, float>(C, expected);
    }

    SECTION("an outer product, a diagonal and a lone summed letter") {
        auto const A = create_random_tensor<float>("A", 3, 3);
        auto const B = create_random_tensor<double>("B", 3, 4);
        auto const x = create_random_tensor<double>("x", 3);
        for (auto spec : {"ijk <- ij ; k", "i <- ii ; i", "i <- ij ; i"}) {
            CAPTURE(spec);
            std::string const s = spec;
            if (s == "ijk <- ij ; k") {
                auto C = create_zero_tensor<double>("C", 3, 3, 3), expected = C;
                cg::einsum(cg::EinsumFormatString(s), &C, A, x);
                reference_einsum(s, &expected, A, x);
                check_against<double, float, double>(C, expected);
            } else {
                auto C = create_zero_tensor<double>("C", 3), expected = C;
                cg::einsum(cg::EinsumFormatString(s), &C, A, x);
                reference_einsum(s, &expected, A, x);
                check_against<double, float, double>(C, expected);
            }
        }
    }
}

TEST_CASE("mixed precision einsum - the rules string_einsum enforces still hold", "[ComputeGraph][MixedPrecision]") {
    SECTION("an empty contraction scales C once") {
        auto const A = create_zero_tensor<float>("A", 3, 0);
        auto const B = create_zero_tensor<double>("B", 0, 2);
        auto       C = create_zero_tensor<double>("C", 3, 2);
        C.set_all(4.0);
        cg::einsum("ij <- ik ; kj", 0.5, &C, 1.0, A, B);
        CHECK(C(2, 1) == 2.0);
        CHECK(std::string(cg::dispatch::last_dispatch_route()) == "empty_input_scale_only");
    }

    SECTION("an output overlapping an input is rejected") {
        auto C = create_random_tensor<double>("C", 4, 4);
        auto B = create_random_tensor<float>("B", 4, 4);
        CHECK_THROWS_AS(cg::einsum("ij <- ik ; kj", &C, C, B), std::invalid_argument);
    }

    SECTION("permutation operators are not supported") {
        auto const A = create_random_tensor<float>("A", 3, 3);
        auto const B = create_random_tensor<double>("B", 3, 3);
        auto       C = create_zero_tensor<double>("C", 3, 3);
        CHECK_THROWS_AS(cg::einsum("ij <- P(ij) ik ; kj", &C, A, B), std::invalid_argument);
    }

    SECTION("capturing is not supported") {
        auto const             A = create_random_tensor<float>("A", 3, 3);
        auto const             B = create_random_tensor<double>("B", 3, 3);
        auto                   C = create_zero_tensor<double>("C", 3, 3);
        cg::Graph              graph("mixed");
        cg::CaptureGuard const guard(graph);
        CHECK_THROWS_AS(cg::einsum("ij <- ik ; kj", &C, A, B), std::invalid_argument);
    }
}
