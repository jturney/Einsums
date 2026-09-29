//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The packed engine's complex contractions with --einsums:packed-gemm:complex-1m on and off.
//
// With the flag on, an x86 rung from V2 up runs complex on the REAL vector tile of the underlying
// type by the 1m method (see MicroKernelShape::use_1m); with it off, complex takes the block
// strategy on the scatter path. Both are checked here against the brute-force reference, over
// every element type (the flag must leave real types alone), with conjugated operands and complex
// prefactors. The engine each case ran is asserted, so a case that silently fell back to another
// engine cannot pass for the one it names.
//
// try_packed_gemm is called directly so every case exercises this backend whatever einsum dispatch
// would prefer. The file is re-registered once per SIMD rung, so each rung's 1m route runs on its
// own tile.

#include <Einsums/Options/Get.hpp>
#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/PackedGemm/Options.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorAlgebra/Detail/PackedGemmIndices.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <cmath>
#include <complex>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;

namespace {

/// Sets the complex-1m flag for one scope and restores it after.
class Complex1mFlag {
  public:
    explicit Complex1mFlag(bool value) : _previous{config::get(option::PackedGemmComplex1m)} {
        config::set(option::PackedGemmComplex1m, value);
    }
    ~Complex1mFlag() { config::set(option::PackedGemmComplex1m, _previous); }

    Complex1mFlag(Complex1mFlag const &)            = delete;
    Complex1mFlag &operator=(Complex1mFlag const &) = delete;

  private:
    bool _previous;
};

/// Whether the rung this process selected has a 1m route to opt into.
bool rung_has_1m() {
#if defined(__x86_64__) || defined(_M_X64)
    return einsums::simd::selected_arch() >= einsums::simd::InstructionSet::V2;
#else
    return einsums::simd::selected_arch() == einsums::simd::InstructionSet::Sme;
#endif
}

/// The engine the packed loops should have run for element type @p T with the flag at @p flag.
template <typename T>
std::string expected_engine(bool flag) {
    if constexpr (IsComplexV<T>) {
        if (einsums::simd::selected_arch() == einsums::simd::InstructionSet::Sme || (flag && rung_has_1m())) {
            return "1m";
        }
        return packed_gemm::micro_kernel_shape<T>().use_3m ? "3m" : "block_gemm";
    } else {
        // A rung without a vector tile kernel for T runs real types through block_gemm too: the
        // aarch64 NEON rung does, which a forced EINSUMS_SIMD_ARCH=baseline selects on an SME machine.
        return packed_gemm::micro_kernel_shape<T>().block_gemm ? "block_gemm" : "tile";
    }
}

/// Compares @p C against the reference and names the engine that produced it.
///
/// Every element is a sum of @p k products of operands drawn from the unit box (complex ones have
/// modulus below sqrt(2)), so no term exceeds 2 in modulus and the result's rounding is bounded by
/// the element type's tolerance times the sum of the terms' moduli. A wrong sign or a dropped
/// conjugate in the 1m packing is off by the size of the terms themselves, far outside it.
template <typename T, size_t Rank>
void check_against(Tensor<T, Rank> const &C, Tensor<T, Rank> const &C_ref, int64_t k, T ab_pf, T c_pf) {
    double const magnitude = 2.0 * static_cast<double>(k) * std::abs(ab_pf) + std::sqrt(2.0) * std::abs(c_pf);
    auto const  *got       = C.data();
    auto const  *want      = C_ref.data();
    for (size_t e = 0; e < C.size(); e++) {
        CAPTURE(e);
        REQUIRE_THAT(got[e], CheckWithinMagnitude(want[e], magnitude));
    }
}

/// Runs one prefactor pair through the ring-like scatter C(a,b,i,j) = A(a,e,i,m) * B(e,b,m,j).
///
/// C's two index groups interleave, M = (a, i) and N = (b, j), so the contraction takes the scatter
/// path; K = (e, m). The extents are odd so the last tile in every direction is a partial one.
template <typename T, bool ConjA, bool ConjB>
void ring(int64_t na, int64_t ne, int64_t ni, int64_t nm, int64_t nb, int64_t nj, T c_pf, T ab_pf, bool flag) {
    Complex1mFlag const guard{flag};

    auto A = create_random_tensor<T>("A", na, ne, ni, nm);
    auto B = create_random_tensor<T>("B", ne, nb, nm, nj);
    auto C = create_random_tensor<T>("C", na, nb, ni, nj);

    Tensor<T, 4> C_ref = C;
    testing::reference_einsum("abij <- aeim ; ebmj", c_pf, &C_ref, ab_pf, A, B, ConjA, ConjB);

    bool const handled = tensor_algebra::detail::try_packed_gemm_indices<ConjA, ConjB>(c_pf, Indices{a, b, i, j}, &C, ab_pf,
                                                                                       Indices{a, e, i, m}, A, Indices{e, b, m, j}, B);
    REQUIRE(handled);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");
    CHECK(std::string(packed_gemm::last_packed_engine()) == expected_engine<T>(flag));

    check_against(C, C_ref, ne * nm, ab_pf, c_pf);
}

/// A GEMM's worth of work on the scatter path: C(a,b,c,d) = A(a,c,k) * B(k,b,d), with M = (a, c)
/// and N = (b, d) interleaved in C and one K index.
template <typename T, bool ConjA, bool ConjB>
void gemm_like(int64_t na, int64_t nc, int64_t nk, int64_t nb, int64_t nd, T c_pf, T ab_pf, bool flag) {
    Complex1mFlag const guard{flag};

    auto A = create_random_tensor<T>("A", na, nc, nk);
    auto B = create_random_tensor<T>("B", nk, nb, nd);
    auto C = create_random_tensor<T>("C", na, nb, nc, nd);

    Tensor<T, 4> C_ref = C;
    testing::reference_einsum("abcd <- ack ; kbd", c_pf, &C_ref, ab_pf, A, B, ConjA, ConjB);

    bool const handled = tensor_algebra::detail::try_packed_gemm_indices<ConjA, ConjB>(c_pf, Indices{a, b, c, d}, &C, ab_pf,
                                                                                       Indices{a, c, k}, A, Indices{k, b, d}, B);
    REQUIRE(handled);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");
    CHECK(std::string(packed_gemm::last_packed_engine()) == expected_engine<T>(flag));

    check_against(C, C_ref, nk, ab_pf, c_pf);
}

} // namespace

TEMPLATE_LIST_TEST_CASE("complex-1m: ring scatter against the reference", "[PackedGemm][Complex1m]", testing::AllScalarTypes) {
    using T = TestType;
    for (bool const flag : {false, true}) {
        CAPTURE(flag);
        // C's prefactor 0 stores rather than accumulates, 1 accumulates onto C, and a genuinely
        // complex one scales C first; each is a separate branch of the 1m write-back.
        for (T const c_pf : {T{0}, T{1}, testing::prefactor<T>(0.3, -0.7)}) {
            CAPTURE(c_pf);
            T const ab_pf = testing::prefactor<T>(1.1, 0.4);
            SECTION("plain") {
                ring<T, false, false>(5, 3, 3, 2, 7, 3, c_pf, ab_pf, flag);
            }
            SECTION("conj A") {
                ring<T, true, false>(5, 3, 3, 2, 7, 3, c_pf, ab_pf, flag);
            }
            SECTION("conj B") {
                ring<T, false, true>(5, 3, 3, 2, 7, 3, c_pf, ab_pf, flag);
            }
            SECTION("conj A and B") {
                ring<T, true, true>(5, 3, 3, 2, 7, 3, c_pf, ab_pf, flag);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("complex-1m: GEMM-like scatter against the reference", "[PackedGemm][Complex1m]", testing::AllScalarTypes) {
    using T       = TestType;
    T const c_pf  = testing::prefactor<T>(-0.6, 0.2);
    T const ab_pf = testing::prefactor<T>(0.9, -1.3);
    for (bool const flag : {false, true}) {
        CAPTURE(flag);
        gemm_like<T, false, false>(3, 5, 17, 4, 5, c_pf, ab_pf, flag);
        gemm_like<T, true, true>(3, 5, 17, 4, 5, c_pf, ab_pf, flag);
    }
}

TEMPLATE_LIST_TEST_CASE("complex-1m: blocks in K, M and N", "[PackedGemm][Complex1m]", testing::ComplexScalarTypes) {
    // Large enough that the 1m route's own K and M blocks (derived from the cache model for the
    // real kernel) and the outer N block each split the contraction more than once, so the beta
    // prescale on the first K block, the accumulation over the later ones and the edge tiles of
    // every block all run. K = 20 * 21 = 420 complex, 840 real; M = 9 * 11 = 99; N = 13 * 9 = 117.
    using T       = TestType;
    T const c_pf  = testing::prefactor<T>(0.3, -0.7);
    T const ab_pf = testing::prefactor<T>(1.1, 0.4);
    ring<T, false, true>(9, 20, 11, 21, 13, 9, c_pf, ab_pf, true);
    ring<T, true, false>(9, 20, 11, 21, 13, 9, T{0}, ab_pf, true);
}

TEMPLATE_LIST_TEST_CASE("complex-1m: the flag selects the real tile on x86 from V2 up", "[PackedGemm][Complex1m]",
                        testing::ComplexScalarTypes) {
    using T     = TestType;
    using RealT = RemoveComplexT<T>;
    {
        Complex1mFlag const guard{true};
        auto const          shape = packed_gemm::micro_kernel_shape<T>();
        CHECK(shape.use_1m == rung_has_1m());
        if (rung_has_1m()) {
            // 1m advertises the real kernel's geometry: blis_contraction packs with it.
            auto const real = packed_gemm::micro_kernel_shape<RealT>();
            CHECK(shape.mr == real.mr);
            CHECK(shape.nr == real.nr);
            CHECK_FALSE(shape.block_gemm);
        }
    }
#if defined(__x86_64__) || defined(_M_X64)
    // Off by default, and off means the block strategy the flag replaces.
    Complex1mFlag const guard{false};
    CHECK_FALSE(packed_gemm::micro_kernel_shape<T>().use_1m);
    CHECK(packed_gemm::micro_kernel_shape<T>().block_gemm);
#endif
}
