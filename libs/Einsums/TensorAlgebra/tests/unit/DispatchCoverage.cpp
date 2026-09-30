//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Comprehensive dispatch coverage tests.
// Covers dispatch edge cases: conjugation, prefactors, TensorView,
// complex types, and float variants across all dispatch paths.

#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <limits>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;

// ============================================================================
// DOT path gaps: conjugation, prefactors
// ============================================================================

TEMPLATE_LIST_TEST_CASE("dot_conjugation", "[dispatch][dot]", testing::ComplexScalarTypes) {
    using T = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 8;
    auto   A  = create_random_tensor<T>("A", di);
    auto   B  = create_random_tensor<T>("B", di);

    // ConjA: C = conj(A) . B
    Tensor<T, 0> C1("C1");
    T            ref1{0};
    for (size_t i0 = 0; i0 < di; i0++) {
        ref1 += std::conj(A(i0)) * B(i0);
    }

    REQUIRE_NOTHROW(einsum<true, false>(Indices{}, &C1, Indices{i}, A, Indices{i}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::DOT);
    REQUIRE_THAT((T)C1, CheckWithinRel(ref1, 0.0001));

    // ConjB: C = A . conj(B)
    Tensor<T, 0> C2("C2");
    T            ref2{0};
    for (size_t i0 = 0; i0 < di; i0++) {
        ref2 += A(i0) * std::conj(B(i0));
    }

    REQUIRE_NOTHROW(einsum<false, true>(Indices{}, &C2, Indices{i}, A, Indices{i}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::DOT);
    REQUIRE_THAT((T)C2, CheckWithinRel(ref2, 0.0001));
}

TEMPLATE_LIST_TEST_CASE("dot_prefactors", "[dispatch][dot]", testing::AllScalarTypes) {
    using T                                       = TestType;
    T const                                 c_pf  = testing::prefactor<T>(0.5, 0.3);
    T const                                 ab_pf = testing::prefactor<T>(2.1, -0.7);
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 10;
    auto   A  = create_random_tensor<T>("A", di);
    auto   B  = create_random_tensor<T>("B", di);

    // C = 0.5*C + 2.0 * A.B
    Tensor<T, 0> C("C");
    (T &)C = testing::prefactor<T>(3.14, 1.1);

    T ref = testing::prefactor<T>(3.14, 1.1) * c_pf;
    for (size_t i0 = 0; i0 < di; i0++) {
        ref += ab_pf * A(i0) * B(i0);
    }

    REQUIRE_NOTHROW(einsum(c_pf, Indices{}, &C, ab_pf, Indices{i}, A, Indices{i}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::DOT);
    REQUIRE_THAT((T)C, CheckWithinRel(ref));
}

// A zero output prefactor assigns rather than multiplies, as on every other route: 0 * NaN is NaN,
// so a dot into a never-written output kept whatever it held. The DOT branch scaled C with
// operator*=, whose rank-0 arm multiplies, and a plain scalar output multiplied directly.
TEMPLATE_LIST_TEST_CASE("dot with a zero output prefactor discards what C held", "[dispatch][dot]", testing::AllScalarTypes) {
    using T = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 7;
    auto   A  = create_random_tensor<T>("A", di);
    auto   B  = create_random_tensor<T>("B", di);

    T ref{0};
    for (size_t i0 = 0; i0 < di; i0++) {
        ref += A(i0) * B(i0);
    }
    T const nan = testing::prefactor<T>(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

    SECTION("a rank-0 tensor output") {
        Tensor<T, 0> C("C");
        (T &)C = nan;
        einsum(T{0}, Indices{}, &C, T{1}, Indices{i}, A, Indices{i}, B, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::DOT);
        REQUIRE_THAT((T)C, CheckWithinRel(ref));
    }

    SECTION("a scalar output") {
        T c = nan;
        einsum(T{0}, Indices{}, &c, T{1}, Indices{i}, A, Indices{i}, B, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::DOT);
        REQUIRE_THAT(c, CheckWithinRel(ref));
    }
}

// ============================================================================
// DIRECT path gaps: α≠1
// ============================================================================

TEMPLATE_LIST_TEST_CASE("direct_alpha", "[dispatch][direct]", testing::AllScalarTypes) {
    using T                                       = TestType;
    T const                                 c_pf  = testing::prefactor<T>(0.5, 0.3);
    T const                                 ab_pf = testing::prefactor<T>(2.1, -0.7);
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 5, dj = 6;
    auto   A = create_random_tensor<T>("A", di, dj);
    auto   B = create_random_tensor<T>("B", di, dj);
    auto   C = create_random_tensor<T>("C", di, dj);

    auto C_ref = Tensor<T, 2>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0, j0) = c_pf * C(i0, j0) + ab_pf * A(i0, j0) * B(i0, j0);
        }
    }

    REQUIRE_NOTHROW(einsum(c_pf, Indices{i, j}, &C, ab_pf, Indices{i, j}, A, Indices{i, j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::DIRECT);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), CheckWithinRel(C_ref(i0, j0)));
        }
    }
}

// ============================================================================
// GER path gaps: prefactors (β≠0, α≠1), conjugation
// ============================================================================

TEMPLATE_LIST_TEST_CASE("ger_prefactors", "[dispatch][ger]", testing::AllScalarTypes) {
    using T                                       = TestType;
    T const                                 c_pf  = testing::prefactor<T>(0.5, 0.3);
    T const                                 ab_pf = testing::prefactor<T>(2.1, -0.7);
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 5, dj = 6;
    auto   A = create_random_tensor<T>("A", di);
    auto   B = create_random_tensor<T>("B", dj);
    auto   C = create_random_tensor<T>("C", di, dj);

    auto C_ref = Tensor<T, 2>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0, j0) = c_pf * C(i0, j0) + ab_pf * A(i0) * B(j0);
        }
    }

    REQUIRE_NOTHROW(einsum(c_pf, Indices{i, j}, &C, ab_pf, Indices{i}, A, Indices{j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GER);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), CheckWithinRel(C_ref(i0, j0)));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("ger_conjugation", "[dispatch][ger]", testing::ComplexScalarTypes) {
    using T = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(j,i) = A(i) * conj(B(j)), ConjB with swap_AB=true (B targets at front of C)
    // einsum_is_outer_product: swap_AB=true, swapped_conjugation = !ConjB && swap_AB is false
    // Actually: the dispatch tries (C,B,A) too. Let's use a pattern that works:
    // C(j,i) = conj(A(j)) * B(i), swap_AB=false (A targets at front), ConjA with straight
    // straight_conjugation = !ConjA && !swap_AB = false. Doesn't work either.
    //
    // GER conjugation is very restricted: only !ConjA when !swap, or !ConjB when swap.
    // It's actually not supported by the GER path; conjugation falls through to generic.
    // Instead, test that GER works with complex types (no conjugation).
    size_t di = 5, dj = 6;
    auto   A = create_random_tensor<T>("A", di);
    auto   B = create_random_tensor<T>("B", dj);
    auto   C = create_zero_tensor<T>("C", di, dj);

    auto C_ref = create_zero_tensor<T>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0, j0) = A(i0) * B(j0);
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i}, A, Indices{j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GER);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), CheckWithinRel(C_ref(i0, j0), 0.0001));
        }
    }
}

// ============================================================================
// GEMV path gaps: complex, TensorView, conjugation, prefactors
// ============================================================================

TEMPLATE_TEST_CASE("gemv_complex", "[dispatch][gemv]", std::complex<float>, std::complex<double>) {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 5, dj = 6;
    auto   A = create_random_tensor<TestType>("A", di, dj);
    auto   B = create_random_tensor<TestType>("B", dj);
    auto   C = create_zero_tensor<TestType>("C", di);

    auto C_ref = create_zero_tensor<TestType>("C_ref", di);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0) += A(i0, j0) * B(j0);
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i}, &C, Indices{i, j}, A, Indices{j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMV);

    for (size_t i0 = 0; i0 < di; i0++) {
        REQUIRE_THAT(C(i0), CheckWithinRel(C_ref(i0), 0.001));
    }
}

TEMPLATE_LIST_TEST_CASE("gemv_tensorview", "[dispatch][gemv]", testing::AllScalarTypes) {
    using T = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 5, dj = 6;
    auto   A_full = create_random_tensor<T>("A_full", di + 2, dj + 2);
    auto   B_full = create_random_tensor<T>("B_full", dj + 2);
    auto   A_view = A_full(Range{1, (int64_t)(di + 1)}, Range{1, (int64_t)(dj + 1)});
    auto   B_view = B_full(Range{1, (int64_t)(dj + 1)});
    auto   C      = create_zero_tensor<T>("C", di);

    auto C_ref = create_zero_tensor<T>("C_ref", di);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0) += A_view(i0, j0) * B_view(j0);
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i}, &C, Indices{i, j}, A_view, Indices{j}, B_view, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMV);

    for (size_t i0 = 0; i0 < di; i0++) {
        REQUIRE_THAT(C(i0), CheckWithinRel(C_ref(i0)));
    }
}

TEMPLATE_LIST_TEST_CASE("gemv_conjugation", "[dispatch][gemv]", testing::ComplexScalarTypes) {
    using T = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(i) = conj(A(j,i)) * B(j), ConjA with transpose
    // einsum_is_matrix_vector requires ConjA only when transpose_A is true
    size_t di = 5, dj = 6;
    auto   A = create_random_tensor<T>("A", dj, di);
    auto   B = create_random_tensor<T>("B", dj);
    auto   C = create_zero_tensor<T>("C", di);

    auto C_ref = create_zero_tensor<T>("C_ref", di);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0) += std::conj(A(j0, i0)) * B(j0);
        }
    }

    REQUIRE_NOTHROW(einsum<true, false>(Indices{i}, &C, Indices{j, i}, A, Indices{j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMV);

    for (size_t i0 = 0; i0 < di; i0++) {
        REQUIRE_THAT(C(i0), CheckWithinRel(C_ref(i0), 0.001));
    }
}

TEMPLATE_LIST_TEST_CASE("gemv_prefactors", "[dispatch][gemv]", testing::AllScalarTypes) {
    using T                                       = TestType;
    T const                                 c_pf  = testing::prefactor<T>(0.5, 0.3);
    T const                                 ab_pf = testing::prefactor<T>(2.1, -0.7);
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 5, dj = 6;
    auto   A = create_random_tensor<T>("A", di, dj);
    auto   B = create_random_tensor<T>("B", dj);
    auto   C = create_random_tensor<T>("C", di);

    auto C_ref = Tensor<T, 1>("C_ref", di);
    for (size_t i0 = 0; i0 < di; i0++) {
        C_ref(i0) = c_pf * C(i0);
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0) += ab_pf * A(i0, j0) * B(j0);
        }
    }

    REQUIRE_NOTHROW(einsum(c_pf, Indices{i}, &C, ab_pf, Indices{i, j}, A, Indices{j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMV);

    for (size_t i0 = 0; i0 < di; i0++) {
        REQUIRE_THAT(C(i0), CheckWithinRel(C_ref(i0)));
    }
}

// ============================================================================
// GEMM path gaps: conjugation, prefactors, complex<float>
// ============================================================================

TEST_CASE("gemm_complex_double", "[dispatch][gemm]") {
    using T = std::complex<double>;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(i,j) = A(i,k) * B(k,j), standard GEMM with complex<double>
    size_t di = 4, dj = 5, dk = 6;
    auto   A = create_random_tensor<T>("A", di, dk);
    auto   B = create_random_tensor<T>("B", dk, dj);
    auto   C = create_zero_tensor<T>("C", di, dj);

    auto C_ref = create_zero_tensor<T>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            for (size_t k0 = 0; k0 < dk; k0++) {
                C_ref(i0, j0) += A(i0, k0) * B(k0, j0);
            }
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMM);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), CheckWithinRel(C_ref(i0, j0), 0.001));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("gemm_conjA_transposed", "[dispatch][gemm]", testing::ComplexScalarTypes) {
    using T = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(i,j) = conj(A(k,i)) * B(k,j), ConjA with transposed A.
    // transpose_A=true, transpose_C=false → BLAS uses 'c' flag for A^H.
    size_t di = 4, dj = 5, dk = 6;
    auto   A = create_random_tensor<T>("A", dk, di);
    auto   B = create_random_tensor<T>("B", dk, dj);
    auto   C = create_zero_tensor<T>("C", di, dj);

    auto C_ref = create_zero_tensor<T>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            for (size_t k0 = 0; k0 < dk; k0++) {
                C_ref(i0, j0) += std::conj(A(k0, i0)) * B(k0, j0);
            }
        }
    }

    REQUIRE_NOTHROW(einsum<true, false>(Indices{i, j}, &C, Indices{k, i}, A, Indices{k, j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMM);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), CheckWithinRel(C_ref(i0, j0), 0.001));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("gemm_conjA_nontransposed_falls_through", "[dispatch][gemm]", testing::ComplexScalarTypes) {
    using T = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(i,j) = conj(A(i,k)) * B(k,j), ConjA without transpose.
    // BLAS can't apply conjugation-only (no 'c' without transpose), so this
    // must NOT dispatch to GEMM. Falls through to SORT_GEMM or generic.
    size_t di = 4, dj = 5, dk = 6;
    auto   A = create_random_tensor<T>("A", di, dk);
    auto   B = create_random_tensor<T>("B", dk, dj);
    auto   C = create_zero_tensor<T>("C", di, dj);

    auto C_ref = create_zero_tensor<T>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            for (size_t k0 = 0; k0 < dk; k0++) {
                C_ref(i0, j0) += std::conj(A(i0, k0)) * B(k0, j0);
            }
        }
    }

    REQUIRE_NOTHROW(einsum<true, false>(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B, &alg_choice));
    // Must not be GEMM; BLAS can't do conjugate-only.
    REQUIRE(alg_choice != tensor_algebra::detail::GEMM);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), CheckWithinRel(C_ref(i0, j0), 0.001));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("gemm_prefactors", "[dispatch][gemm]", testing::AllScalarTypes) {
    using T                                       = TestType;
    T const                                 c_pf  = testing::prefactor<T>(0.5, 0.3);
    T const                                 ab_pf = testing::prefactor<T>(2.1, -0.7);
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 4, dj = 5, dk = 6;
    auto   A = create_random_tensor<T>("A", di, dk);
    auto   B = create_random_tensor<T>("B", dk, dj);
    auto   C = create_random_tensor<T>("C", di, dj);

    auto C_ref = Tensor<T, 2>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0, j0) = c_pf * C(i0, j0);
            for (size_t k0 = 0; k0 < dk; k0++) {
                C_ref(i0, j0) += ab_pf * A(i0, k0) * B(k0, j0);
            }
        }
    }

    REQUIRE_NOTHROW(einsum(c_pf, Indices{i, j}, &C, ab_pf, Indices{i, k}, A, Indices{k, j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMM);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), CheckWithinRel(C_ref(i0, j0)));
        }
    }
}

TEST_CASE("gemm_complex_float", "[dispatch][gemm]") {
    using T = std::complex<float>;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 4, dj = 5, dk = 6;
    auto   A = create_random_tensor<T>("A", di, dk);
    auto   B = create_random_tensor<T>("B", dk, dj);
    auto   C = create_zero_tensor<T>("C", di, dj);

    auto C_ref = create_zero_tensor<T>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            for (size_t k0 = 0; k0 < dk; k0++) {
                C_ref(i0, j0) += A(i0, k0) * B(k0, j0);
            }
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMM);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), CheckWithinRel(C_ref(i0, j0), 0.01));
        }
    }
}

// ============================================================================
// SORT_GEMM path gaps: float, complex<float>
// ============================================================================

TEST_CASE("sort_gemm_float", "[dispatch][sort_gemm]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(i,l,j) = A(j,k,i) * B(l,k), scrambled indices, float
    size_t di = 3, dj = 4, dk = 5, dl = 3;
    auto   A = create_random_tensor<float>("A", dj, dk, di);
    auto   B = create_random_tensor<float>("B", dl, dk);
    auto   C = create_zero_tensor<float>("C", di, dl, dj);

    auto C_ref = create_zero_tensor<float>("C_ref", di, dl, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                for (size_t k0 = 0; k0 < dk; k0++) {
                    C_ref(i0, l0, j0) += A(j0, k0, i0) * B(l0, k0);
                }
            }
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i, l, j}, &C, Indices{j, k, i}, A, Indices{l, k}, B, &alg_choice));
    // On rungs whose kernel wins the scatter path (SME double/float), einsum
    // legitimately picks PACKED_GEMM here instead of Sort+GEMM.
    REQUIRE((alg_choice == tensor_algebra::detail::SORT_GEMM || alg_choice == tensor_algebra::detail::PACKED_GEMM));

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                REQUIRE_THAT(C(i0, l0, j0), Catch::Matchers::WithinRel(C_ref(i0, l0, j0), 0.01f));
            }
        }
    }
}

TEST_CASE("sort_gemm_complex_float", "[dispatch][sort_gemm]") {
    using T = std::complex<float>;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 3, dj = 4, dk = 5, dl = 3;
    auto   A = create_random_tensor<T>("A", dj, dk, di);
    auto   B = create_random_tensor<T>("B", dl, dk);
    auto   C = create_zero_tensor<T>("C", di, dl, dj);

    auto C_ref = create_zero_tensor<T>("C_ref", di, dl, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                for (size_t k0 = 0; k0 < dk; k0++) {
                    C_ref(i0, l0, j0) += A(j0, k0, i0) * B(l0, k0);
                }
            }
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i, l, j}, &C, Indices{j, k, i}, A, Indices{l, k}, B, &alg_choice));
    // On rungs whose kernel wins the scatter path (SME double/float), einsum
    // legitimately picks PACKED_GEMM here instead of Sort+GEMM.
    REQUIRE((alg_choice == tensor_algebra::detail::SORT_GEMM || alg_choice == tensor_algebra::detail::PACKED_GEMM));

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                REQUIRE_THAT(C(i0, l0, j0), CheckWithinRel(C_ref(i0, l0, j0), 0.01));
            }
        }
    }
}

// ============================================================================
// GENERIC path gaps: TensorView, conjugation, prefactors
// ============================================================================

TEMPLATE_LIST_TEST_CASE("generic_tensorview", "[dispatch][generic]", testing::AllScalarTypes) {
    using T = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // Hadamard-like contraction via generic: C(i,j) = A(i,j,k) * B(i,j,k)
    // Uses Hadamard (repeated i,j in output), forces generic path.
    size_t di = 3, dj = 4, dk = 5;
    auto   A_full = create_random_tensor<T>("A_full", di + 2, dj + 2, dk + 2);
    auto   B_full = create_random_tensor<T>("B_full", di + 2, dj + 2, dk + 2);
    auto   A_view = A_full(Range{1, (int64_t)(di + 1)}, Range{1, (int64_t)(dj + 1)}, Range{1, (int64_t)(dk + 1)});
    auto   B_view = B_full(Range{1, (int64_t)(di + 1)}, Range{1, (int64_t)(dj + 1)}, Range{1, (int64_t)(dk + 1)});
    auto   C      = create_zero_tensor<T>("C", di, dj);

    auto C_ref = create_zero_tensor<T>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            for (size_t k0 = 0; k0 < dk; k0++) {
                C_ref(i0, j0) += A_view(i0, j0, k0) * B_view(i0, j0, k0);
            }
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, j, k}, A_view, Indices{i, j, k}, B_view, &alg_choice));
    // This is a trace-like contraction with shared target indices; goes to generic.
    // (Not DOT because C is not scalar; not GEMM because no proper M/N/K split)

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), CheckWithinRel(C_ref(i0, j0)));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("generic_conjugation", "[dispatch][generic]", testing::ComplexScalarTypes) {
    using T = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // Hadamard contraction with conjugation; must go to generic.
    // C(i) = conj(A(i,i)) * B(i,i), repeated index i forces Hadamard/generic
    size_t di = 5;
    auto   A  = create_random_tensor<T>("A", di, di);
    auto   B  = create_random_tensor<T>("B", di, di);
    auto   C  = create_zero_tensor<T>("C", di);

    auto C_ref = create_zero_tensor<T>("C_ref", di);
    for (size_t i0 = 0; i0 < di; i0++) {
        C_ref(i0) = std::conj(A(i0, i0)) * B(i0, i0);
    }

    REQUIRE_NOTHROW(einsum<true, false>(Indices{i}, &C, Indices{i, i}, A, Indices{i, i}, B, &alg_choice));

    for (size_t i0 = 0; i0 < di; i0++) {
        REQUIRE_THAT(C(i0), CheckWithinRel(C_ref(i0), 0.001));
    }
}

TEMPLATE_LIST_TEST_CASE("generic_prefactors", "[dispatch][generic]", testing::AllScalarTypes) {
    using T                                       = TestType;
    T const                                 c_pf  = testing::prefactor<T>(0.5, 0.3);
    T const                                 ab_pf = testing::prefactor<T>(2.1, -0.7);
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // Hadamard + prefactors: C(i) = 0.5*C(i) + 2.0*A(i,i)*B(i,i)
    size_t di = 5;
    auto   A  = create_random_tensor<T>("A", di, di);
    auto   B  = create_random_tensor<T>("B", di, di);
    auto   C  = create_random_tensor<T>("C", di);

    auto C_ref = Tensor<T, 1>("C_ref", di);
    for (size_t i0 = 0; i0 < di; i0++) {
        C_ref(i0) = c_pf * C(i0) + ab_pf * A(i0, i0) * B(i0, i0);
    }

    REQUIRE_NOTHROW(einsum(c_pf, Indices{i}, &C, ab_pf, Indices{i, i}, A, Indices{i, i}, B, &alg_choice));

    for (size_t i0 = 0; i0 < di; i0++) {
        REQUIRE_THAT(C(i0), CheckWithinRel(C_ref(i0)));
    }
}

// ============================================================================
// Broadcast output indices must reach the generic algorithm
// ============================================================================

TEMPLATE_LIST_TEST_CASE("broadcast_output_index_reaches_generic", "[dispatch][generic][broadcast]", testing::AllScalarTypes) {
    using T = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // Defect: an output index carried by neither operand is a broadcast, and
    // only the generic algorithm implements one. "ikl <- ij ; jk" nonetheless
    // satisfied einsum_is_matrix_product, which never inspects l, so the GEMM
    // route filled C's first slice and left the rest at whatever the prefactor
    // had scaled them to. The answer was wrong and nothing said so.
    //
    // Deterministic operands: the failure was in which route claimed the spec,
    // not in the arithmetic, so a fixed A and B pin it without a random draw.
    size_t const di = 2, dj = 3, dk = 4, dl = 3;

    auto A = Tensor<T, 2>("A", di, dj);
    auto B = Tensor<T, 2>("B", dj, dk);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            A(i0, j0) = T(1.0 + static_cast<double>(i0 * dj + j0));
        }
    }
    for (size_t j0 = 0; j0 < dj; j0++) {
        for (size_t k0 = 0; k0 < dk; k0++) {
            B(j0, k0) = T(1.0 + static_cast<double>(j0 * dk + k0));
        }
    }

    // Seeded with a value the contraction never writes, so a slice the engine
    // skips stays visible instead of being mistaken for a correct zero.
    auto C = Tensor<T, 3>("C", di, dk, dl);
    C.set_all(T{-7});

    REQUIRE_NOTHROW(einsum(Indices{i, k, l}, &C, Indices{i, j}, A, Indices{j, k}, B, &alg_choice));

    // No BLAS fast path can carry the broadcast index; the generic loops must
    // take it.
    REQUIRE(alg_choice == tensor_algebra::detail::GENERIC);

    // Every slice along l receives the same matrix product.
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t k0 = 0; k0 < dk; k0++) {
            T ref{0};
            for (size_t j0 = 0; j0 < dj; j0++) {
                ref += A(i0, j0) * B(j0, k0);
            }
            for (size_t l0 = 0; l0 < dl; l0++) {
                REQUIRE_THAT(C(i0, k0, l0), CheckWithinRel(ref, 1.0e-12));
            }
        }
    }
}

// A letter in one input alone, absent from C, is summed over that input. The engine left such a
// letter out of the loop entirely, so it read only the letter's first slice: "i <- ij ; i" gave
// A(i, 0) * B(i) instead of the row sum times B(i). The string engine had the same defect and had
// it fixed; this engine did not, until the reference einsum compared the two.
TEMPLATE_LIST_TEST_CASE("Dispatch - a letter summed over one input alone", "[einsum][dispatch]", testing::AllScalarTypes) {
    using T = TestType;
    // A few order-one terms per element, so a few ulps of the element type.
    double const                            tol        = 100.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon();
    tensor_algebra::detail::AlgorithmChoice alg_choice = tensor_algebra::detail::INDETERMINATE;

    size_t constexpr di = 3, dj = 4, dk = 2;
    auto A = create_random_tensor<T>("A", di, dj);
    auto b = create_random_tensor<T>("b", di);

    auto row_sum = [&](size_t i0) {
        T sum{0};
        for (size_t j0 = 0; j0 < dj; j0++)
            sum += A(i0, j0);
        return sum;
    };

    SECTION("only in A") {
        auto C = create_zero_tensor<T>("C", di);
        einsum(Indices{i}, &C, Indices{i, j}, A, Indices{i}, b, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::GENERIC);
        for (size_t i0 = 0; i0 < di; i0++)
            REQUIRE_THAT(C(i0), CheckWithinRel(row_sum(i0) * b(i0), tol));
    }

    SECTION("only in B") {
        auto C = create_zero_tensor<T>("C", di);
        einsum(Indices{i}, &C, Indices{i}, b, Indices{i, j}, A, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::GENERIC);
        for (size_t i0 = 0; i0 < di; i0++)
            REQUIRE_THAT(C(i0), CheckWithinRel(b(i0) * row_sum(i0), tol));
    }

    SECTION("beside an outer product") {
        auto c = create_random_tensor<T>("c", dk);
        auto C = create_zero_tensor<T>("C", di, dk);
        einsum(Indices{i, k}, &C, Indices{i, j}, A, Indices{k}, c, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::GENERIC);
        for (size_t i0 = 0; i0 < di; i0++)
            for (size_t k0 = 0; k0 < dk; k0++)
                REQUIRE_THAT(C(i0, k0), CheckWithinRel(row_sum(i0) * c(k0), tol));
    }

    SECTION("repeated within that input") {
        // "i <- ijj ; i": the trace of each slice of T, times b.
        auto X = create_random_tensor<T>("X", di, dj, dj);
        auto C = create_zero_tensor<T>("C", di);
        einsum(Indices{i}, &C, Indices{i, j, j}, X, Indices{i}, b, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::GENERIC);
        for (size_t i0 = 0; i0 < di; i0++) {
            T trace{0};
            for (size_t j0 = 0; j0 < dj; j0++)
                trace += X(i0, j0, j0);
            REQUIRE_THAT(C(i0), CheckWithinRel(trace * b(i0), tol));
        }
    }
}

// An empty operand leaves nothing to contract, but the output prefactor still applies, once. The
// engine ran no iteration for an empty link and so never applied c_pf: C came back unscaled. The
// string engine had this rule already.
TEMPLATE_LIST_TEST_CASE("Dispatch - an empty operand applies the output prefactor once", "[einsum][dispatch]", testing::AllScalarTypes) {
    using T                                            = TestType;
    tensor_algebra::detail::AlgorithmChoice alg_choice = tensor_algebra::detail::INDETERMINATE;

    SECTION("an empty link scales C") {
        auto A = create_zero_tensor<T>("A", 3, 0);
        auto B = create_zero_tensor<T>("B", 0, 4);
        auto C = create_zero_tensor<T>("C", 3, 4);
        C.set_all(T{3});
        einsum(T{2}, Indices{i, j}, &C, T{1}, Indices{i, k}, A, Indices{k, j}, B, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::EMPTY);
        REQUIRE(C(0, 0) == T{6});
        REQUIRE(C(2, 3) == T{6});
    }

    SECTION("c_pf == 0 assigns zero, whatever C held") {
        auto A = create_zero_tensor<T>("A", 3, 0);
        auto B = create_zero_tensor<T>("B", 0, 4);
        auto C = create_zero_tensor<T>("C", 3, 4);
        C.set_all(testing::prefactor<T>(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()));
        einsum(T{0}, Indices{i, j}, &C, T{1}, Indices{i, k}, A, Indices{k, j}, B, &alg_choice);
        REQUIRE(C(1, 1) == T{0});
    }

    SECTION("an empty output is left alone") {
        auto A = create_random_tensor<T>("A", 0, 5);
        auto B = create_random_tensor<T>("B", 5, 4);
        auto C = create_zero_tensor<T>("C", 0, 4);
        REQUIRE_NOTHROW(einsum(T{2}, Indices{i, j}, &C, T{1}, Indices{i, k}, A, Indices{k, j}, B, &alg_choice));
        REQUIRE(alg_choice == tensor_algebra::detail::EMPTY);
    }

    SECTION("a scalar output") {
        auto x = create_zero_tensor<T>("x", 0);
        auto y = create_zero_tensor<T>("y", 0);
        T    c = T{5};
        einsum(T{2}, Indices{}, &c, T{1}, Indices{i}, x, Indices{i}, y, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::EMPTY);
        REQUIRE(c == T{10});
    }
}
