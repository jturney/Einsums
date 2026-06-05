//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkMultiMN.cpp
/// @brief Exhaustive benchmarks for multi-M, multi-N, batch GEMM, and packing specializations.
///
/// Compares PackedGemm (einsum dispatch) against the generic nested-loop algorithm
/// for various contraction patterns and tensor sizes. Reports speedup ratios.

#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <cstdio>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
using namespace einsums::performance;

namespace {

template <typename... CI, typename... AI, typename... BI, size_t CR, size_t AR, size_t BR>
void run_generic(double beta, std::tuple<CI...> c_idx, Tensor<double, CR> &C, double alpha, std::tuple<AI...> a_idx,
                 Tensor<double, AR> const &A, std::tuple<BI...> b_idx, Tensor<double, BR> const &B) {
    tensor_algebra::detail::einsum<true, false, false, false>(beta, c_idx, &C, alpha, a_idx, A, b_idx, B);
}

template <typename... CI, typename... AI, typename... BI, size_t CR, size_t AR, size_t BR>
void run_einsum(double beta, std::tuple<CI...> c_idx, Tensor<double, CR> &C, double alpha, std::tuple<AI...> a_idx,
                Tensor<double, AR> const &A, std::tuple<BI...> b_idx, Tensor<double, BR> const &B) {
    tensor_algebra::einsum(beta, c_idx, &C, alpha, a_idx, A, b_idx, B);
}

void compare(char const *label, int N, TimingStats const &t_gen, TimingStats const &t_ein) {
    double speedup = (t_ein.avg > 0) ? t_gen.avg / t_ein.avg : 0;
    publish_benchmark_result(label, "t_einsum", N, t_ein);
    std::printf("  [%s] generic: %.2f us  einsum: %.2f us  speedup: %.1fx\n", label, t_gen.avg, t_ein.avg, speedup);
    std::fflush(stdout);
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-M: C[i,j,l] = A[i,j,k] * B[k,l]
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench multi-M: C[i,j,l] = A[i,j,k]*B[k,l] N=8", "[PackedGemm][MultiM][benchmark]") {
    LabeledSection0();
    constexpr int     N = 8;
    Tensor<double, 3> A{"A", N, N, N}, C{"C", N, N, N};
    Tensor<double, 2> B{"B", N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,j,l]=A[i,j,k]*B[k,l]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "multi-M");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k}, A, Indices{k, l}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k}, A, Indices{k, l}, B);
    });
    compare("multiM-N8", N, tg, te);
}

EINSUMS_TEST_CASE("Bench multi-M: C[i,j,l] = A[i,j,k]*B[k,l] N=16", "[PackedGemm][MultiM][benchmark]") {
    LabeledSection0();
    constexpr int     N = 16;
    Tensor<double, 3> A{"A", N, N, N}, C{"C", N, N, N};
    Tensor<double, 2> B{"B", N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,j,l]=A[i,j,k]*B[k,l]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "multi-M");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k}, A, Indices{k, l}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k}, A, Indices{k, l}, B);
    });
    compare("multiM-N16", N, tg, te);
}

EINSUMS_TEST_CASE("Bench multi-M: C[i,j,l] = A[i,j,k]*B[k,l] N=32", "[PackedGemm][MultiM][benchmark]") {
    LabeledSection0();
    constexpr int     N = 32;
    Tensor<double, 3> A{"A", N, N, N}, C{"C", N, N, N};
    Tensor<double, 2> B{"B", N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,j,l]=A[i,j,k]*B[k,l]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "multi-M");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k}, A, Indices{k, l}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k}, A, Indices{k, l}, B);
    });
    compare("multiM-N32", N, tg, te);
}

EINSUMS_TEST_CASE("Bench multi-M: C[i,j,l] = A[i,j,k]*B[k,l] N=64", "[PackedGemm][MultiM][benchmark]") {
    LabeledSection0();
    constexpr int     N = 64;
    Tensor<double, 3> A{"A", N, N, N}, C{"C", N, N, N};
    Tensor<double, 2> B{"B", N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,j,l]=A[i,j,k]*B[k,l]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "multi-M");
    auto tg = time_us(
        "generic",
        [&]() {
            C.zero();
            run_generic(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k}, A, Indices{k, l}, B);
        },
        3);
    auto te = time_us(
        "einsum",
        [&]() {
            C.zero();
            run_einsum(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k}, A, Indices{k, l}, B);
        },
        3);
    compare("multiM-N64", N, tg, te);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-N: C[i,j,l] = A[i,k] * B[k,j,l]
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench multi-N: C[i,j,l] = A[i,k]*B[k,j,l] N=16", "[PackedGemm][MultiN][benchmark]") {
    LabeledSection0();
    constexpr int     N = 16;
    Tensor<double, 2> A{"A", N, N};
    Tensor<double, 3> B{"B", N, N, N}, C{"C", N, N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,j,l]=A[i,k]*B[k,j,l]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "multi-N");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{i, j, l}, C, 1.0, Indices{i, k}, A, Indices{k, j, l}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{i, j, l}, C, 1.0, Indices{i, k}, A, Indices{k, j, l}, B);
    });
    compare("multiN-N16", N, tg, te);
}

EINSUMS_TEST_CASE("Bench multi-N: C[i,j,l] = A[i,k]*B[k,j,l] N=32", "[PackedGemm][MultiN][benchmark]") {
    LabeledSection0();
    constexpr int     N = 32;
    Tensor<double, 2> A{"A", N, N};
    Tensor<double, 3> B{"B", N, N, N}, C{"C", N, N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,j,l]=A[i,k]*B[k,j,l]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "multi-N");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{i, j, l}, C, 1.0, Indices{i, k}, A, Indices{k, j, l}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{i, j, l}, C, 1.0, Indices{i, k}, A, Indices{k, j, l}, B);
    });
    compare("multiN-N32", N, tg, te);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-M + Multi-N: C[i,j,l,m] = A[i,j,k] * B[k,l,m]
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench multi-MN: C[i,j,l,m] = A[i,j,k]*B[k,l,m] N=8", "[PackedGemm][MultiMN][benchmark]") {
    LabeledSection0();
    constexpr int     N = 8;
    Tensor<double, 3> A{"A", N, N, N}, B{"B", N, N, N};
    Tensor<double, 4> C{"C", N, N, N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,j,l,m]=A[i,j,k]*B[k,l,m]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "multi-MN");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{i, j, l, m}, C, 1.0, Indices{i, j, k}, A, Indices{k, l, m}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{i, j, l, m}, C, 1.0, Indices{i, j, k}, A, Indices{k, l, m}, B);
    });
    compare("multiMN-N8", N, tg, te);
}

EINSUMS_TEST_CASE("Bench multi-MN: C[i,j,l,m] = A[i,j,k]*B[k,l,m] N=16", "[PackedGemm][MultiMN][benchmark]") {
    LabeledSection0();
    constexpr int     N = 16;
    Tensor<double, 3> A{"A", N, N, N}, B{"B", N, N, N};
    Tensor<double, 4> C{"C", N, N, N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,j,l,m]=A[i,j,k]*B[k,l,m]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "multi-MN");
    auto tg = time_us(
        "generic",
        [&]() {
            C.zero();
            run_generic(0.0, Indices{i, j, l, m}, C, 1.0, Indices{i, j, k}, A, Indices{k, l, m}, B);
        },
        3);
    auto te = time_us(
        "einsum",
        [&]() {
            C.zero();
            run_einsum(0.0, Indices{i, j, l, m}, C, 1.0, Indices{i, j, k}, A, Indices{k, l, m}, B);
        },
        3);
    compare("multiMN-N16", N, tg, te);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-M + Multi-K: C[i,j,l] = A[i,j,k,m] * B[k,m,l]
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench multi-MK: C[i,j,l] = A[i,j,k,m]*B[k,m,l] N=8", "[PackedGemm][MultiMK][benchmark]") {
    LabeledSection0();
    constexpr int     N = 8;
    Tensor<double, 4> A{"A", N, N, N, N};
    Tensor<double, 3> B{"B", N, N, N}, C{"C", N, N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,j,l]=A[i,j,k,m]*B[k,m,l]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "multi-MK");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k, m}, A, Indices{k, m, l}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k, m}, A, Indices{k, m, l}, B);
    });
    compare("multiMK-N8", N, tg, te);
}

EINSUMS_TEST_CASE("Bench multi-MK: C[i,j,l] = A[i,j,k,m]*B[k,m,l] N=16", "[PackedGemm][MultiMK][benchmark]") {
    LabeledSection0();
    constexpr int     N = 16;
    Tensor<double, 4> A{"A", N, N, N, N};
    Tensor<double, 3> B{"B", N, N, N}, C{"C", N, N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,j,l]=A[i,j,k,m]*B[k,m,l]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "multi-MK");
    auto tg = time_us(
        "generic",
        [&]() {
            C.zero();
            run_generic(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k, m}, A, Indices{k, m, l}, B);
        },
        3);
    auto te = time_us(
        "einsum",
        [&]() {
            C.zero();
            run_einsum(0.0, Indices{i, j, l}, C, 1.0, Indices{i, j, k, m}, A, Indices{k, m, l}, B);
        },
        3);
    compare("multiMK-N16", N, tg, te);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch GEMM: C[b,i,j] = A[b,i,k] * B[b,k,j]
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench batch GEMM: C[b,i,j] = A[b,i,k]*B[b,k,j] b=4 N=32", "[PackedGemm][Batch][benchmark]") {
    LabeledSection0();
    constexpr int     B_DIM = 4, N = 32;
    Tensor<double, 3> A{"A", B_DIM, N, N}, B{"B", B_DIM, N, N}, C{"C", B_DIM, N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[b,i,j]=A[b,i,k]*B[b,k,j]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "batch");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{b, i, j}, C, 1.0, Indices{b, i, k}, A, Indices{b, k, j}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{b, i, j}, C, 1.0, Indices{b, i, k}, A, Indices{b, k, j}, B);
    });
    compare("batch-b4-N32", N, tg, te);
}

EINSUMS_TEST_CASE("Bench batch GEMM: C[b,i,j] = A[b,i,k]*B[b,k,j] b=16 N=16", "[PackedGemm][Batch][benchmark]") {
    LabeledSection0();
    constexpr int     B_DIM = 16, N = 16;
    Tensor<double, 3> A{"A", B_DIM, N, N}, B{"B", B_DIM, N, N}, C{"C", B_DIM, N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[b,i,j]=A[b,i,k]*B[b,k,j]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "batch");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{b, i, j}, C, 1.0, Indices{b, i, k}, A, Indices{b, k, j}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{b, i, j}, C, 1.0, Indices{b, i, k}, A, Indices{b, k, j}, B);
    });
    compare("batch-b16-N16", N, tg, te);
}

EINSUMS_TEST_CASE("Bench batch GEMM: C[b,i,j] = A[b,i,k]*B[b,k,j] b=64 N=8", "[PackedGemm][Batch][benchmark]") {
    LabeledSection0();
    constexpr int     B_DIM = 64, N = 8;
    Tensor<double, 3> A{"A", B_DIM, N, N}, B{"B", B_DIM, N, N}, C{"C", B_DIM, N, N};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[b,i,j]=A[b,i,k]*B[b,k,j]");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "batch");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{b, i, j}, C, 1.0, Indices{b, i, k}, A, Indices{b, k, j}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{b, i, j}, C, 1.0, Indices{b, i, k}, A, Indices{b, k, j}, B);
    });
    compare("batch-b64-N8", N, tg, te);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Asymmetric sizes (realistic QC-like dimensions)
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench QC-like: C[i,a] = A[i,j]*B[j,a] nocc=10 nvirt=50", "[PackedGemm][QC][benchmark]") {
    LabeledSection0();
    constexpr int     nocc = 10, nvirt = 50;
    Tensor<double, 2> A{"A", nocc, nocc}, B{"B", nocc, nvirt}, C{"C", nocc, nvirt};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[i,a]=A[i,j]*B[j,a]");
    ProfileAnnotate("N", int64_t(nocc));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "QC");
    auto tg = time_us("generic", [&]() {
        C.zero();
        run_generic(0.0, Indices{i, a}, C, 1.0, Indices{i, j}, A, Indices{j, a}, B);
    });
    auto te = time_us("einsum", [&]() {
        C.zero();
        run_einsum(0.0, Indices{i, a}, C, 1.0, Indices{i, j}, A, Indices{j, a}, B);
    });
    compare("QC-ov-10x50", nocc, tg, te);
}

EINSUMS_TEST_CASE("Bench QC-like: C[i,j,a,b] = g[i,j,k,l]*t2[k,l,a,b] no=5 nv=15", "[PackedGemm][QC][benchmark]") {
    LabeledSection0();
    constexpr int     no = 5, nv = 15;
    Tensor<double, 4> g{"g", no, no, no, no}, t2{"t2", no, no, nv, nv};
    Tensor<double, 4> C{"C", no, no, nv, nv};
    fill(g);
    fill(t2);

    ProfileAnnotate("pattern", "C[i,j,a,b]=g[i,j,k,l]*t2[k,l,a,b]");
    ProfileAnnotate("N", int64_t(no));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "QC");
    auto tg = time_us(
        "generic",
        [&]() {
            C.zero();
            run_generic(0.0, Indices{i, j, a, b}, C, 1.0, Indices{i, j, k, l}, g, Indices{k, l, a, b}, t2);
        },
        3);
    auto te = time_us(
        "einsum",
        [&]() {
            C.zero();
            run_einsum(0.0, Indices{i, j, a, b}, C, 1.0, Indices{i, j, k, l}, g, Indices{k, l, a, b}, t2);
        },
        3);
    compare("QC-g*t2-5x15", no, tg, te);
}

EINSUMS_TEST_CASE("Bench QC-like: integral transform step C[mu,q,r,s] = A[mu,nu,r,s]*B[nu,q] nao=10 nmo=8", "[PackedGemm][QC][benchmark]") {
    LabeledSection0();
    constexpr int     nao = 10, nmo = 8;
    Tensor<double, 4> A{"A", nao, nao, nmo, nmo};
    Tensor<double, 2> B{"B", nao, nmo};
    Tensor<double, 4> C{"C", nao, nmo, nmo, nmo};
    fill(A);
    fill(B);

    ProfileAnnotate("pattern", "C[mu,q,r,s]=A[mu,nu,r,s]*B[nu,q]");
    ProfileAnnotate("N", int64_t(nao));
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("category", "QC");
    auto tg = time_us(
        "generic",
        [&]() {
            C.zero();
            run_generic(0.0, Indices{i, a, k, l}, C, 1.0, Indices{i, j, k, l}, A, Indices{j, a}, B);
        },
        3);
    auto te = time_us(
        "einsum",
        [&]() {
            C.zero();
            run_einsum(0.0, Indices{i, a, k, l}, C, 1.0, Indices{i, j, k, l}, A, Indices{j, a}, B);
        },
        3);
    compare("QC-transform-10x8", nao, tg, te);
}
