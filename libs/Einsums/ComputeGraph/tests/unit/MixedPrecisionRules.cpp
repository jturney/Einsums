//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The arithmetic rules of a mixed-precision einsum, on their own: what R is for every pair of
// operand types, that the run-time rule agrees with the compile-time one, and what is rejected.

#include <Einsums/ComputeGraph/Detail/MixedPrecision.hpp>

#include <complex>
#include <type_traits>

#include <Einsums/Testing.hpp>

using einsums::compute_graph::detail::check_mixed_einsum;
using einsums::compute_graph::detail::promote;
using einsums::compute_graph::detail::PromoteT;
using einsums::compute_graph::detail::storable_v;
using einsums::packed_gemm::get_scalar_type;
using einsums::packed_gemm::ScalarType;

using cf = std::complex<float>;
using cd = std::complex<double>;

// complex<float> with double is the pair a size comparison gets wrong: both are 8 bytes.
static_assert(std::is_same_v<PromoteT<cf, double>, cd>);
static_assert(std::is_same_v<PromoteT<double, cf>, cd>);
static_assert(std::is_same_v<PromoteT<float, double>, double>);
static_assert(std::is_same_v<PromoteT<float, float>, float>);
static_assert(std::is_same_v<PromoteT<cf, float>, cf>);
static_assert(std::is_same_v<PromoteT<cf, cd>, cd>);
static_assert(std::is_same_v<PromoteT<cd, float>, cd>);

static_assert(storable_v<cd, cf>); // narrowing precision is allowed
static_assert(storable_v<double, float>);
static_assert(storable_v<float, cd>);   // widening into complex is allowed
static_assert(!storable_v<cf, double>); // dropping the imaginary part is not

TEMPLATE_PRODUCT_TEST_CASE("mixed precision - the run-time promotion agrees with the compile-time one", "[ComputeGraph][MixedPrecision]",
                           std::pair,
                           ((float, float), (float, double), (float, cf), (float, cd), (double, float), (double, double), (double, cf),
                            (double, cd), (cf, float), (cf, double), (cf, cf), (cf, cd), (cd, float), (cd, double), (cd, cf), (cd, cd))) {
    using A = typename TestType::first_type;
    using B = typename TestType::second_type;
    CHECK(promote(get_scalar_type<A>(), get_scalar_type<B>()) == get_scalar_type<PromoteT<A, B>>());
}

TEST_CASE("mixed precision - what is rejected when the einsum is built", "[ComputeGraph][MixedPrecision]") {
    // Allowed: every mix that stores its result without losing an imaginary part.
    CHECK_NOTHROW(check_mixed_einsum(ScalarType::Float64, ScalarType::Float32, ScalarType::Float64, false, "test"));
    CHECK_NOTHROW(check_mixed_einsum(ScalarType::Float32, ScalarType::Float64, ScalarType::Float64, false, "test")); // narrowing
    CHECK_NOTHROW(check_mixed_einsum(ScalarType::Complex128, ScalarType::Complex64, ScalarType::Float64, false, "test"));
    CHECK_NOTHROW(check_mixed_einsum(ScalarType::Complex64, ScalarType::Complex128, ScalarType::Float64, true, "test"));
    CHECK_NOTHROW(check_mixed_einsum(ScalarType::Complex128, ScalarType::Float64, ScalarType::Float64, true, "test"));

    // A complex operand makes the product complex; a real C cannot hold it.
    CHECK_THROWS_AS(check_mixed_einsum(ScalarType::Float64, ScalarType::Complex64, ScalarType::Float64, false, "test"),
                    std::invalid_argument);
    // A complex prefactor cannot scale a real C, even when every operand is real.
    CHECK_THROWS_AS(check_mixed_einsum(ScalarType::Float64, ScalarType::Float32, ScalarType::Float64, true, "test"), std::invalid_argument);
    // Types the rules do not cover.
    CHECK_THROWS_AS(check_mixed_einsum(ScalarType::Unknown, ScalarType::Float64, ScalarType::Float64, false, "test"),
                    std::invalid_argument);
}
