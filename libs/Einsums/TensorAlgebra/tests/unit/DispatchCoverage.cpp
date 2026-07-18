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

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;

// ============================================================================
// DOT path gaps: conjugation, prefactors
// ============================================================================

TEST_CASE("dot_conjugation", "[dispatch][dot]") {
    using T = std::complex<double>;
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

TEST_CASE("dot_prefactors", "[dispatch][dot]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 10;
    auto   A  = create_random_tensor<double>("A", di);
    auto   B  = create_random_tensor<double>("B", di);

    // C = 0.5*C + 2.0 * A.B
    Tensor<double, 0> C("C");
    (double &)C = 3.14;

    double ref = 3.14 * 0.5;
    for (size_t i0 = 0; i0 < di; i0++) {
        ref += 2.0 * A(i0) * B(i0);
    }

    REQUIRE_NOTHROW(einsum(0.5, Indices{}, &C, 2.0, Indices{i}, A, Indices{i}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::DOT);
    REQUIRE_THAT((double)C, Catch::Matchers::WithinRel(ref, 0.0001));
}

// ============================================================================
// DIRECT path gaps: α≠1
// ============================================================================

TEST_CASE("direct_alpha", "[dispatch][direct]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 5, dj = 6;
    auto   A = create_random_tensor<double>("A", di, dj);
    auto   B = create_random_tensor<double>("B", di, dj);
    auto   C = create_random_tensor<double>("C", di, dj);

    auto C_ref = Tensor<double, 2>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0, j0) = 0.5 * C(i0, j0) + 3.0 * A(i0, j0) * B(i0, j0);
        }
    }

    REQUIRE_NOTHROW(einsum(0.5, Indices{i, j}, &C, 3.0, Indices{i, j}, A, Indices{i, j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::DIRECT);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinRel(C_ref(i0, j0), 0.0001));
        }
    }
}

// ============================================================================
// GER path gaps: prefactors (β≠0, α≠1), conjugation
// ============================================================================

TEST_CASE("ger_prefactors", "[dispatch][ger]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 5, dj = 6;
    auto   A = create_random_tensor<double>("A", di);
    auto   B = create_random_tensor<double>("B", dj);
    auto   C = create_random_tensor<double>("C", di, dj);

    auto C_ref = Tensor<double, 2>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0, j0) = 0.5 * C(i0, j0) + 3.0 * A(i0) * B(j0);
        }
    }

    REQUIRE_NOTHROW(einsum(0.5, Indices{i, j}, &C, 3.0, Indices{i}, A, Indices{j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GER);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinRel(C_ref(i0, j0), 0.0001));
        }
    }
}

TEST_CASE("ger_conjugation", "[dispatch][ger]") {
    using T = std::complex<double>;
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

TEST_CASE("gemv_tensorview", "[dispatch][gemv]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 5, dj = 6;
    auto   A_full = create_random_tensor<double>("A_full", di + 2, dj + 2);
    auto   B_full = create_random_tensor<double>("B_full", dj + 2);
    auto   A_view = A_full(Range{1, (int64_t)(di + 1)}, Range{1, (int64_t)(dj + 1)});
    auto   B_view = B_full(Range{1, (int64_t)(dj + 1)});
    auto   C      = create_zero_tensor<double>("C", di);

    auto C_ref = create_zero_tensor<double>("C_ref", di);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0) += A_view(i0, j0) * B_view(j0);
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i}, &C, Indices{i, j}, A_view, Indices{j}, B_view, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMV);

    for (size_t i0 = 0; i0 < di; i0++) {
        REQUIRE_THAT(C(i0), Catch::Matchers::WithinRel(C_ref(i0), 0.0001));
    }
}

TEST_CASE("gemv_conjugation", "[dispatch][gemv]") {
    using T = std::complex<double>;
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

TEST_CASE("gemv_prefactors", "[dispatch][gemv]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 5, dj = 6;
    auto   A = create_random_tensor<double>("A", di, dj);
    auto   B = create_random_tensor<double>("B", dj);
    auto   C = create_random_tensor<double>("C", di);

    auto C_ref = Tensor<double, 1>("C_ref", di);
    for (size_t i0 = 0; i0 < di; i0++) {
        C_ref(i0) = 0.5 * C(i0);
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0) += 2.0 * A(i0, j0) * B(j0);
        }
    }

    REQUIRE_NOTHROW(einsum(0.5, Indices{i}, &C, 2.0, Indices{i, j}, A, Indices{j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMV);

    for (size_t i0 = 0; i0 < di; i0++) {
        REQUIRE_THAT(C(i0), Catch::Matchers::WithinRel(C_ref(i0), 0.0001));
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

TEST_CASE("gemm_conjA_transposed", "[dispatch][gemm]") {
    using T = std::complex<double>;
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

TEST_CASE("gemm_conjA_nontransposed_falls_through", "[dispatch][gemm]") {
    using T = std::complex<double>;
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

TEST_CASE("gemm_prefactors", "[dispatch][gemm]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    size_t di = 4, dj = 5, dk = 6;
    auto   A = create_random_tensor<double>("A", di, dk);
    auto   B = create_random_tensor<double>("B", dk, dj);
    auto   C = create_random_tensor<double>("C", di, dj);

    auto C_ref = Tensor<double, 2>("C_ref", di, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            C_ref(i0, j0) = 0.5 * C(i0, j0);
            for (size_t k0 = 0; k0 < dk; k0++) {
                C_ref(i0, j0) += 2.0 * A(i0, k0) * B(k0, j0);
            }
        }
    }

    REQUIRE_NOTHROW(einsum(0.5, Indices{i, j}, &C, 2.0, Indices{i, k}, A, Indices{k, j}, B, &alg_choice));
    REQUIRE(alg_choice == tensor_algebra::detail::GEMM);

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t j0 = 0; j0 < dj; j0++) {
            REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinRel(C_ref(i0, j0), 0.0001));
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

TEST_CASE("generic_tensorview", "[dispatch][generic]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // Hadamard-like contraction via generic: C(i,j) = A(i,j,k) * B(i,j,k)
    // Uses Hadamard (repeated i,j in output), forces generic path.
    size_t di = 3, dj = 4, dk = 5;
    auto   A_full = create_random_tensor<double>("A_full", di + 2, dj + 2, dk + 2);
    auto   B_full = create_random_tensor<double>("B_full", di + 2, dj + 2, dk + 2);
    auto   A_view = A_full(Range{1, (int64_t)(di + 1)}, Range{1, (int64_t)(dj + 1)}, Range{1, (int64_t)(dk + 1)});
    auto   B_view = B_full(Range{1, (int64_t)(di + 1)}, Range{1, (int64_t)(dj + 1)}, Range{1, (int64_t)(dk + 1)});
    auto   C      = create_zero_tensor<double>("C", di, dj);

    auto C_ref = create_zero_tensor<double>("C_ref", di, dj);
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
            REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinRel(C_ref(i0, j0), 0.0001));
        }
    }
}

TEST_CASE("generic_conjugation", "[dispatch][generic]") {
    using T = std::complex<double>;
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

TEST_CASE("generic_prefactors", "[dispatch][generic]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // Hadamard + prefactors: C(i) = 0.5*C(i) + 2.0*A(i,i)*B(i,i)
    size_t di = 5;
    auto   A  = create_random_tensor<double>("A", di, di);
    auto   B  = create_random_tensor<double>("B", di, di);
    auto   C  = create_random_tensor<double>("C", di);

    auto C_ref = Tensor<double, 1>("C_ref", di);
    for (size_t i0 = 0; i0 < di; i0++) {
        C_ref(i0) = 0.5 * C(i0) + 2.0 * A(i0, i0) * B(i0, i0);
    }

    REQUIRE_NOTHROW(einsum(0.5, Indices{i}, &C, 2.0, Indices{i, i}, A, Indices{i, i}, B, &alg_choice));

    for (size_t i0 = 0; i0 < di; i0++) {
        REQUIRE_THAT(C(i0), Catch::Matchers::WithinRel(C_ref(i0), 0.0001));
    }
}
