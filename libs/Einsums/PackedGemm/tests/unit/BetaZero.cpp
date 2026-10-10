//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// A zero C prefactor through the packed engine's write paths.
//
// beta == 0 means C is overwritten without being read, so a NaN already in C must not survive. The
// engine meets that without a clearing pass: on the first K block the tile kernel overwrites (C
// directly, or the scratch buffer a scattered C is written back from), and every later K block adds.
// A first-block store that ran on every block would drop all but the last block's sum, so the
// direct-C case uses a K several blocks long and checks the whole sum arrives.

#include <Einsums/PackedGemm/PackedGemm.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorAlgebra/Detail/PackedGemmIndices.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;

namespace {

template <typename T>
T nan_value() {
    if constexpr (IsComplexV<T>) {
        using R = RemoveComplexT<T>;
        return T{std::numeric_limits<R>::quiet_NaN(), std::numeric_limits<R>::quiet_NaN()};
    } else {
        return std::numeric_limits<T>::quiet_NaN();
    }
}

template <typename T, size_t Rank>
void fill_nan(Tensor<T, Rank> &C) {
    std::fill(C.data(), C.data() + C.size(), nan_value<T>());
}

/// Every element finite and within @p tol_per_k * k of the reference.
template <typename T, size_t Rank>
void check_close(Tensor<T, Rank> const &C, Tensor<T, Rank> const &C_ref, int64_t k) {
    double const tol = 64.0 * static_cast<double>(k) * std::numeric_limits<RemoveComplexT<T>>::epsilon();
    for (size_t idx = 0; idx < C.size(); ++idx) {
        T const got  = C.data()[idx];
        T const want = C_ref.data()[idx];
        REQUIRE(std::isfinite(std::abs(got)));
        REQUIRE(std::abs(got - want) <= tol * (1.0 + std::abs(want)));
    }
}

} // namespace

// Single M and single N with a B that no BLAS call can address (no unit stride), so the packed tile
// engine writes straight into C. K spans several K blocks.
TEMPLATE_TEST_CASE("beta zero: direct C overwrites on the first K block and adds after", "[PackedGemm][BetaZero]", float, double) {
    using T              = TestType;
    constexpr int64_t nm = 37;
    constexpr int64_t nn = 29;
    constexpr int64_t nk = 9000; // past the SME rung's 4096 KC hint, so at least three K blocks

    auto const A  = create_random_tensor<T>("A", nm, nk);
    auto       B3 = create_random_tensor<T>("B3", 2, nk, nn);
    auto const B  = B3(1, All, All); // strides (2, 2 nk): no unit stride, not BLAS-addressable
    auto       C  = create_random_tensor<T>("C", nm, nn);
    fill_nan(C);
    Tensor<T, 2> C_ref = C;

    testing::reference_einsum("ij <- ik ; kj", T{0}, &C_ref, T{1}, A, B);

    packed_gemm::last_contraction_route() = "";
    packed_gemm::last_packed_blocking()   = {};
    bool const handled =
        tensor_algebra::detail::try_packed_gemm_indices<false, false>(T{0}, Indices{i, j}, &C, T{1}, Indices{i, k}, A, Indices{k, j}, B);
    REQUIRE(handled);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");
    INFO("engine " << packed_gemm::last_packed_engine() << ", KC " << packed_gemm::last_packed_blocking().kc);
    REQUIRE(packed_gemm::last_packed_blocking().kc < nk); // more than one K block
    check_close(C, C_ref, nk);
}

// The CCSD ring: C's M and N groups interleave, so the result is scattered through a scratch buffer
// (tile engine) or a vendor GEMM's output (block engine), and complex runs the 1m route where the
// rung has one. Each must leave no trace of the NaN that C held.
TEMPLATE_TEST_CASE("beta zero: scattered C drops what C held", "[PackedGemm][BetaZero]", float, double, std::complex<double>) {
    using T        = TestType;
    auto const A   = create_random_tensor<T>("A", 24, 12, 13, 9);
    auto const B   = create_random_tensor<T>("B", 12, 22, 9, 11);
    auto       C   = create_random_tensor<T>("C", 24, 22, 13, 11);
    T const    two = T{2};
    fill_nan(C);
    Tensor<T, 4> C_ref = C;

    testing::reference_einsum("abij <- aeim ; ebmj", T{0}, &C_ref, two, A, B);

    packed_gemm::last_contraction_route() = "";
    bool const handled = tensor_algebra::detail::try_packed_gemm_indices<false, false>(T{0}, Indices{a, b, i, j}, &C, two,
                                                                                       Indices{a, e, i, m}, A, Indices{e, b, m, j}, B);
    REQUIRE(handled);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");
    INFO("engine " << packed_gemm::last_packed_engine());
    check_close(C, C_ref, 12 * 9);
}
