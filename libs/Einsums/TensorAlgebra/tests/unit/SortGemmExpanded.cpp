//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;

// All sort_gemm tests use rank-3+ tensors with interleaved target/link indices
// so that einsum_is_matrix_product fails and sort+GEMM is selected.

TEST_CASE("sort_gemm_complex", "[sort_gemm]") {
    using T = std::complex<double>;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(i,l,j) = A(j,k,i) * B(l,k)
    // M={i,j} in A at positions {2,0}, non-contiguous → not a matrix product
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
    // On rungs whose tile kernel wins the scatter path (SME), einsum
    // legitimately picks PACKED_GEMM for these shapes instead of Sort+GEMM.
    REQUIRE((alg_choice == tensor_algebra::detail::SORT_GEMM || alg_choice == tensor_algebra::detail::PACKED_GEMM));

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                REQUIRE_THAT(C(i0, l0, j0).real(), Catch::Matchers::WithinRel(C_ref(i0, l0, j0).real(), 0.001));
                REQUIRE_THAT(C(i0, l0, j0).imag(), Catch::Matchers::WithinRel(C_ref(i0, l0, j0).imag(), 0.001));
            }
        }
    }
}

TEST_CASE("sort_gemm_conjugation", "[sort_gemm]") {
    using T = std::complex<double>;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(i,l,j) = conj(A(j,k,i)) * B(l,k), ConjA with scrambled indices
    size_t di = 3, dj = 4, dk = 5, dl = 3;
    auto   A = create_random_tensor<T>("A", dj, dk, di);
    auto   B = create_random_tensor<T>("B", dl, dk);
    auto   C = create_zero_tensor<T>("C", di, dl, dj);

    auto C_ref = create_zero_tensor<T>("C_ref", di, dl, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                for (size_t k0 = 0; k0 < dk; k0++) {
                    C_ref(i0, l0, j0) += std::conj(A(j0, k0, i0)) * B(l0, k0);
                }
            }
        }
    }

    REQUIRE_NOTHROW(einsum<true, false>(Indices{i, l, j}, &C, Indices{j, k, i}, A, Indices{l, k}, B, &alg_choice));
    // On rungs whose tile kernel wins the scatter path (SME), einsum
    // legitimately picks PACKED_GEMM for these shapes instead of Sort+GEMM.
    REQUIRE((alg_choice == tensor_algebra::detail::SORT_GEMM || alg_choice == tensor_algebra::detail::PACKED_GEMM));

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                REQUIRE_THAT(C(i0, l0, j0).real(), Catch::Matchers::WithinRel(C_ref(i0, l0, j0).real(), 0.001));
                REQUIRE_THAT(C(i0, l0, j0).imag(), Catch::Matchers::WithinRel(C_ref(i0, l0, j0).imag(), 0.001));
            }
        }
    }
}

TEST_CASE("sort_gemm_tensorview", "[sort_gemm]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(i,l,j) = A(j,k,i) * B(l,k) using TensorView inputs
    size_t di = 3, dj = 4, dk = 5, dl = 3;
    auto   A_full = create_random_tensor<double>("A_full", dj + 2, dk + 2, di + 2);
    auto   B_full = create_random_tensor<double>("B_full", dl + 2, dk + 2);

    auto A_view = A_full(Range{1, (int64_t)(dj + 1)}, Range{1, (int64_t)(dk + 1)}, Range{1, (int64_t)(di + 1)});
    auto B_view = B_full(Range{1, (int64_t)(dl + 1)}, Range{1, (int64_t)(dk + 1)});
    auto C      = create_zero_tensor<double>("C", di, dl, dj);

    auto C_ref = create_zero_tensor<double>("C_ref", di, dl, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                for (size_t k0 = 0; k0 < dk; k0++) {
                    C_ref(i0, l0, j0) += A_view(j0, k0, i0) * B_view(l0, k0);
                }
            }
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i, l, j}, &C, Indices{j, k, i}, A_view, Indices{l, k}, B_view, &alg_choice));
    // On rungs whose tile kernel wins the scatter path (SME), einsum
    // legitimately picks PACKED_GEMM for these shapes instead of Sort+GEMM.
    REQUIRE((alg_choice == tensor_algebra::detail::SORT_GEMM || alg_choice == tensor_algebra::detail::PACKED_GEMM));

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                REQUIRE_THAT(C(i0, l0, j0), Catch::Matchers::WithinRel(C_ref(i0, l0, j0), 0.0001));
            }
        }
    }
}

TEST_CASE("sort_gemm_batch_scrambled", "[sort_gemm]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(p,i,l,j) = A(p,j,k,i) * B(p,l,k), single batch dim p, scrambled M/N/K
    size_t dp = 3, di = 4, dj = 5, dk = 6, dl = 3;
    auto   A     = create_random_tensor<double>("A", dp, dj, dk, di);
    auto   B     = create_random_tensor<double>("B", dp, dl, dk);
    auto   C     = create_zero_tensor<double>("C", dp, di, dl, dj);
    auto   C_ref = create_zero_tensor<double>("C_ref", dp, di, dl, dj);

    for (size_t p0 = 0; p0 < dp; p0++) {
        for (size_t i0 = 0; i0 < di; i0++) {
            for (size_t l0 = 0; l0 < dl; l0++) {
                for (size_t j0 = 0; j0 < dj; j0++) {
                    for (size_t k0 = 0; k0 < dk; k0++) {
                        C_ref(p0, i0, l0, j0) += A(p0, j0, k0, i0) * B(p0, l0, k0);
                    }
                }
            }
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{p, i, l, j}, &C, Indices{p, j, k, i}, A, Indices{p, l, k}, B, &alg_choice));
    // On rungs whose tile kernel wins the scatter path (SME), einsum
    // legitimately picks PACKED_GEMM for these shapes instead of Sort+GEMM.
    REQUIRE((alg_choice == tensor_algebra::detail::SORT_GEMM || alg_choice == tensor_algebra::detail::PACKED_GEMM));

    for (size_t p0 = 0; p0 < dp; p0++) {
        for (size_t i0 = 0; i0 < di; i0++) {
            for (size_t l0 = 0; l0 < dl; l0++) {
                for (size_t j0 = 0; j0 < dj; j0++) {
                    REQUIRE_THAT(C(p0, i0, l0, j0), Catch::Matchers::WithinRel(C_ref(p0, i0, l0, j0), 0.0001));
                }
            }
        }
    }
}

TEST_CASE("sort_gemm_batch_c_permute", "[sort_gemm]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(i,p,j,l) = A(p,j,k,i) * B(p,l,k), batch dim p not at front of C (c_needs_permute)
    size_t dp = 3, di = 4, dj = 5, dk = 6, dl = 3;
    auto   A     = create_random_tensor<double>("A", dp, dj, dk, di);
    auto   B     = create_random_tensor<double>("B", dp, dl, dk);
    auto   C     = create_zero_tensor<double>("C", di, dp, dj, dl);
    auto   C_ref = create_zero_tensor<double>("C_ref", di, dp, dj, dl);

    for (size_t p0 = 0; p0 < dp; p0++) {
        for (size_t i0 = 0; i0 < di; i0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                for (size_t l0 = 0; l0 < dl; l0++) {
                    for (size_t k0 = 0; k0 < dk; k0++) {
                        C_ref(i0, p0, j0, l0) += A(p0, j0, k0, i0) * B(p0, l0, k0);
                    }
                }
            }
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{i, p, j, l}, &C, Indices{p, j, k, i}, A, Indices{p, l, k}, B, &alg_choice));
    // On rungs whose tile kernel wins the scatter path (SME), einsum
    // legitimately picks PACKED_GEMM for these shapes instead of Sort+GEMM.
    REQUIRE((alg_choice == tensor_algebra::detail::SORT_GEMM || alg_choice == tensor_algebra::detail::PACKED_GEMM));

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t p0 = 0; p0 < dp; p0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                for (size_t l0 = 0; l0 < dl; l0++) {
                    REQUIRE_THAT(C(i0, p0, j0, l0), Catch::Matchers::WithinRel(C_ref(i0, p0, j0, l0), 0.0001));
                }
            }
        }
    }
}

TEST_CASE("sort_gemm_batch_beta", "[sort_gemm]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C = 0.5*C + 2.0*A*B with batch dim p
    // C(p,i,l,j) = 0.5*C(p,i,l,j) + 2.0*A(p,j,k,i)*B(p,l,k)
    size_t dp = 2, di = 3, dj = 4, dk = 5, dl = 3;
    auto   A     = create_random_tensor<double>("A", dp, dj, dk, di);
    auto   B     = create_random_tensor<double>("B", dp, dl, dk);
    auto   C     = create_random_tensor<double>("C", dp, di, dl, dj);
    auto   C_ref = Tensor<double, 4>("C_ref", dp, di, dl, dj);

    // Copy C into C_ref for reference computation.
    for (size_t p0 = 0; p0 < dp; p0++) {
        for (size_t i0 = 0; i0 < di; i0++) {
            for (size_t l0 = 0; l0 < dl; l0++) {
                for (size_t j0 = 0; j0 < dj; j0++) {
                    C_ref(p0, i0, l0, j0) = C(p0, i0, l0, j0);
                }
            }
        }
    }

    for (size_t p0 = 0; p0 < dp; p0++) {
        for (size_t i0 = 0; i0 < di; i0++) {
            for (size_t l0 = 0; l0 < dl; l0++) {
                for (size_t j0 = 0; j0 < dj; j0++) {
                    C_ref(p0, i0, l0, j0) *= 0.5;
                    for (size_t k0 = 0; k0 < dk; k0++) {
                        C_ref(p0, i0, l0, j0) += 2.0 * A(p0, j0, k0, i0) * B(p0, l0, k0);
                    }
                }
            }
        }
    }

    REQUIRE_NOTHROW(einsum(0.5, Indices{p, i, l, j}, &C, 2.0, Indices{p, j, k, i}, A, Indices{p, l, k}, B, &alg_choice));
    // On rungs whose tile kernel wins the scatter path (SME), einsum
    // legitimately picks PACKED_GEMM for these shapes instead of Sort+GEMM.
    REQUIRE((alg_choice == tensor_algebra::detail::SORT_GEMM || alg_choice == tensor_algebra::detail::PACKED_GEMM));

    for (size_t p0 = 0; p0 < dp; p0++) {
        for (size_t i0 = 0; i0 < di; i0++) {
            for (size_t l0 = 0; l0 < dl; l0++) {
                for (size_t j0 = 0; j0 < dj; j0++) {
                    REQUIRE_THAT(C(p0, i0, l0, j0), Catch::Matchers::WithinRel(C_ref(p0, i0, l0, j0), 0.0001));
                }
            }
        }
    }
}

TEST_CASE("sort_gemm_batch_tensorview", "[sort_gemm]") {
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(p,i,l,j) = A(p,j,k,i) * B(p,l,k) using TensorView inputs with batch dim
    size_t dp = 3, di = 4, dj = 5, dk = 6, dl = 3;
    auto   A_full = create_random_tensor<double>("A_full", dp + 2, dj + 2, dk + 2, di + 2);
    auto   B_full = create_random_tensor<double>("B_full", dp + 2, dl + 2, dk + 2);

    auto A_view =
        A_full(Range{1, (int64_t)(dp + 1)}, Range{1, (int64_t)(dj + 1)}, Range{1, (int64_t)(dk + 1)}, Range{1, (int64_t)(di + 1)});
    auto B_view = B_full(Range{1, (int64_t)(dp + 1)}, Range{1, (int64_t)(dl + 1)}, Range{1, (int64_t)(dk + 1)});
    auto C      = create_zero_tensor<double>("C", dp, di, dl, dj);

    auto C_ref = create_zero_tensor<double>("C_ref", dp, di, dl, dj);
    for (size_t p0 = 0; p0 < dp; p0++) {
        for (size_t i0 = 0; i0 < di; i0++) {
            for (size_t l0 = 0; l0 < dl; l0++) {
                for (size_t j0 = 0; j0 < dj; j0++) {
                    for (size_t k0 = 0; k0 < dk; k0++) {
                        C_ref(p0, i0, l0, j0) += A_view(p0, j0, k0, i0) * B_view(p0, l0, k0);
                    }
                }
            }
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{p, i, l, j}, &C, Indices{p, j, k, i}, A_view, Indices{p, l, k}, B_view, &alg_choice));
    // On rungs whose tile kernel wins the scatter path (SME), einsum
    // legitimately picks PACKED_GEMM for these shapes instead of Sort+GEMM.
    REQUIRE((alg_choice == tensor_algebra::detail::SORT_GEMM || alg_choice == tensor_algebra::detail::PACKED_GEMM));

    for (size_t p0 = 0; p0 < dp; p0++) {
        for (size_t i0 = 0; i0 < di; i0++) {
            for (size_t l0 = 0; l0 < dl; l0++) {
                for (size_t j0 = 0; j0 < dj; j0++) {
                    REQUIRE_THAT(C(p0, i0, l0, j0), Catch::Matchers::WithinRel(C_ref(p0, i0, l0, j0), 0.0001));
                }
            }
        }
    }
}

TEST_CASE("sort_gemm_complex_conj_combined", "[sort_gemm]") {
    using T = std::complex<double>;
    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // C(i,l,j) = conj(A(j,k,i)) * conj(B(l,k)), both ConjA and ConjB, scrambled
    size_t di = 3, dj = 4, dk = 5, dl = 3;
    auto   A = create_random_tensor<T>("A", dj, dk, di);
    auto   B = create_random_tensor<T>("B", dl, dk);
    auto   C = create_zero_tensor<T>("C", di, dl, dj);

    auto C_ref = create_zero_tensor<T>("C_ref", di, dl, dj);
    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                for (size_t k0 = 0; k0 < dk; k0++) {
                    C_ref(i0, l0, j0) += std::conj(A(j0, k0, i0)) * std::conj(B(l0, k0));
                }
            }
        }
    }

    REQUIRE_NOTHROW(einsum<true, true>(Indices{i, l, j}, &C, Indices{j, k, i}, A, Indices{l, k}, B, &alg_choice));
    // On rungs whose tile kernel wins the scatter path (SME), einsum
    // legitimately picks PACKED_GEMM for these shapes instead of Sort+GEMM.
    REQUIRE((alg_choice == tensor_algebra::detail::SORT_GEMM || alg_choice == tensor_algebra::detail::PACKED_GEMM));

    for (size_t i0 = 0; i0 < di; i0++) {
        for (size_t l0 = 0; l0 < dl; l0++) {
            for (size_t j0 = 0; j0 < dj; j0++) {
                REQUIRE_THAT(C(i0, l0, j0).real(), Catch::Matchers::WithinRel(C_ref(i0, l0, j0).real(), 0.001));
                REQUIRE_THAT(C(i0, l0, j0).imag(), Catch::Matchers::WithinRel(C_ref(i0, l0, j0).imag(), 0.001));
            }
        }
    }
}
