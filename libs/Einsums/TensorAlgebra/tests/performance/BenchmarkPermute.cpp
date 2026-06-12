//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Performance benchmarks for tensor_algebra::permute.
// Permute is called twice per sort+GEMM contraction and can dominate cost
// for small tensors. This benchmark measures permute across ranks 2–6 and
// various sizes, with different permutation patterns, data types, rectangular
// shapes, and plan caching effects.

#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <complex>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
using namespace einsums::performance;

namespace {

// =====================================================================
// Rank-2
// =====================================================================

void bench_permute_rank2_transpose(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N), C("C", N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("pattern", "ji<-ij");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N);
    auto t = time_us("permute-r2-transpose", [&] { tensor_algebra::permute(Indices{j, i}, &C, Indices{i, j}, A); });
    publish_benchmark_result("permute-r2-transpose", "t_permute", N, t);
}

void bench_permute_rank2_identity(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N), C("C", N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("pattern", "ij<-ij");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N);
    auto t = time_us("permute-r2-identity", [&] { tensor_algebra::permute(Indices{i, j}, &C, Indices{i, j}, A); });
    publish_benchmark_result("permute-r2-identity", "t_permute", N, t);
}

// Rectangular: tall-skinny and short-wide
void bench_permute_rank2_rect(int M, int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", M, N), C("C", N, M);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("pattern", "ji<-ij");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("M", int64_t(M));
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("elements", int64_t(M) * N);
    auto t = time_us("permute-r2-rect", [&] { tensor_algebra::permute(Indices{j, i}, &C, Indices{i, j}, A); });
    publish_benchmark_result("permute-r2-rect", "t_permute", M * 1000 + N, t);
}

// Float type
void bench_permute_rank2_float(int N) {
    LabeledSection0();
    Tensor<float, 2> A("A", N, N), C("C", N, N);
    for (size_t n = 0; n < A.size(); ++n)
        A.data()[n] = static_cast<float>(n + 1) / static_cast<float>(A.size());
    C.zero();
    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("pattern", "ji<-ij");
    ProfileAnnotate("dtype", "float");
    ProfileAnnotate("elements", int64_t(N) * N);
    auto t = time_us("permute-r2-float", [&] { tensor_algebra::permute(Indices{j, i}, &C, Indices{i, j}, A); });
    publish_benchmark_result("permute-r2-float", "t_permute", N, t);
}

// Complex<double>
void bench_permute_rank2_complex(int N) {
    LabeledSection0();
    using T = std::complex<double>;
    Tensor<T, 2> A("A", N, N), C("C", N, N);
    for (size_t n = 0; n < A.size(); ++n)
        A.data()[n] = T(static_cast<double>(n + 1) / static_cast<double>(A.size()), 0.5);
    C.zero();
    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("pattern", "ji<-ij");
    ProfileAnnotate("dtype", "complex<double>");
    ProfileAnnotate("elements", int64_t(N) * N);
    auto t = time_us("permute-r2-cmplx", [&] { tensor_algebra::permute(Indices{j, i}, &C, Indices{i, j}, A); });
    publish_benchmark_result("permute-r2-cmplx", "t_permute", N, t);
}

// =====================================================================
// Rank-3
// =====================================================================

void bench_permute_rank3_cyclic(int N) {
    LabeledSection0();
    Tensor<double, 3> A("A", N, N, N), C("C", N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "kij<-ijk");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N);
    auto t = time_us("permute-r3-cyclic", [&] { tensor_algebra::permute(Indices{k, i, j}, &C, Indices{i, j, k}, A); });
    publish_benchmark_result("permute-r3-cyclic", "t_permute", N, t);
}

void bench_permute_rank3_swap01(int N) {
    LabeledSection0();
    Tensor<double, 3> A("A", N, N, N), C("C", N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "jik<-ijk");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N);
    auto t = time_us("permute-r3-swap01", [&] { tensor_algebra::permute(Indices{j, i, k}, &C, Indices{i, j, k}, A); });
    publish_benchmark_result("permute-r3-swap01", "t_permute", N, t);
}

// Swap last two: C(i,k,j) = A(i,j,k)
void bench_permute_rank3_swap12(int N) {
    LabeledSection0();
    Tensor<double, 3> A("A", N, N, N), C("C", N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "ikj<-ijk");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N);
    auto t = time_us("permute-r3-swap12", [&] { tensor_algebra::permute(Indices{i, k, j}, &C, Indices{i, j, k}, A); });
    publish_benchmark_result("permute-r3-swap12", "t_permute", N, t);
}

// Full reversal: C(k,j,i) = A(i,j,k)
void bench_permute_rank3_reverse(int N) {
    LabeledSection0();
    Tensor<double, 3> A("A", N, N, N), C("C", N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "kji<-ijk");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N);
    auto t = time_us("permute-r3-reverse", [&] { tensor_algebra::permute(Indices{k, j, i}, &C, Indices{i, j, k}, A); });
    publish_benchmark_result("permute-r3-reverse", "t_permute", N, t);
}

// With beta scaling  C = 0.5*C + 1.0*permute(A)
void bench_permute_rank3_beta(int N) {
    LabeledSection0();
    Tensor<double, 3> A("A", N, N, N), C("C", N, N, N);
    fill(A);
    fill(C);
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "kij<-ijk");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("beta", 0.5);
    ProfileAnnotate("elements", int64_t(N) * N * N);
    auto t = time_us("permute-r3-beta", [&] { tensor_algebra::permute(0.5, Indices{k, i, j}, &C, 1.0, Indices{i, j, k}, A); });
    publish_benchmark_result("permute-r3-beta", "t_permute", N, t);
}

// Rectangular rank-3: C(k,j,i) = A(i,j,k) with different dim sizes
void bench_permute_rank3_rect(int di, int dj, int dk) {
    LabeledSection0();
    Tensor<double, 3> A("A", di, dj, dk), C("C", dk, dj, di);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "kji<-ijk");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("di", int64_t(di));
    ProfileAnnotate("dj", int64_t(dj));
    ProfileAnnotate("dk", int64_t(dk));
    int total = di * dj * dk;
    ProfileAnnotate("elements", int64_t(total));
    auto t = time_us("permute-r3-rect", [&] { tensor_algebra::permute(Indices{k, j, i}, &C, Indices{i, j, k}, A); });
    publish_benchmark_result("permute-r3-rect", "t_permute", total, t);
}

// =====================================================================
// Rank-4
// =====================================================================

void bench_permute_rank4_reverse(int N) {
    LabeledSection0();
    Tensor<double, 4> A("A", N, N, N, N), C("C", N, N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(4));
    ProfileAnnotate("pattern", "lkji<-ijkl");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N * N);
    auto t = time_us("permute-r4-reverse", [&] { tensor_algebra::permute(Indices{l, k, j, i}, &C, Indices{i, j, k, l}, A); });
    publish_benchmark_result("permute-r4-reverse", "t_permute", N, t);
}

void bench_permute_rank4_inner(int N) {
    LabeledSection0();
    Tensor<double, 4> A("A", N, N, N, N), C("C", N, N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(4));
    ProfileAnnotate("pattern", "ikjl<-ijkl");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N * N);
    auto t = time_us("permute-r4-inner", [&] { tensor_algebra::permute(Indices{i, k, j, l}, &C, Indices{i, j, k, l}, A); });
    publish_benchmark_result("permute-r4-inner", "t_permute", N, t);
}

// Cyclic: C(l,i,j,k) = A(i,j,k,l)
void bench_permute_rank4_cyclic(int N) {
    LabeledSection0();
    Tensor<double, 4> A("A", N, N, N, N), C("C", N, N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(4));
    ProfileAnnotate("pattern", "lijk<-ijkl");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N * N);
    auto t = time_us("permute-r4-cyclic", [&] { tensor_algebra::permute(Indices{l, i, j, k}, &C, Indices{i, j, k, l}, A); });
    publish_benchmark_result("permute-r4-cyclic", "t_permute", N, t);
}

// Swap outer pair: C(j,i,l,k) = A(i,j,k,l)
void bench_permute_rank4_outerswap(int N) {
    LabeledSection0();
    Tensor<double, 4> A("A", N, N, N, N), C("C", N, N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(4));
    ProfileAnnotate("pattern", "jilk<-ijkl");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N * N);
    auto t = time_us("permute-r4-outerswap", [&] { tensor_algebra::permute(Indices{j, i, l, k}, &C, Indices{i, j, k, l}, A); });
    publish_benchmark_result("permute-r4-outerswap", "t_permute", N, t);
}

// Rectangular: chemistry MO integrals C(a,i,b,j) = A(i,j,a,b)
void bench_permute_rank4_rect(int nocc, int nvir) {
    LabeledSection0();
    Tensor<double, 4> A("A", nocc, nocc, nvir, nvir), C("C", nvir, nocc, nvir, nocc);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(4));
    ProfileAnnotate("pattern", "aibj<-ijab");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("nocc", int64_t(nocc));
    ProfileAnnotate("nvir", int64_t(nvir));
    ProfileAnnotate("elements", int64_t(nocc) * nocc * nvir * nvir);
    auto t = time_us("permute-r4-rect", [&] { tensor_algebra::permute(Indices{a, i, b, j}, &C, Indices{i, j, a, b}, A); });
    publish_benchmark_result("permute-r4-rect", "t_permute", nocc * 1000 + nvir, t);
}

// Rectangular: 3-index integrals C(Q,a,i) = A(Q,i,a)  (Q large, i,a medium)
void bench_permute_rank3_3idx(int naux, int nocc, int nvir) {
    LabeledSection0();
    Tensor<double, 3> A("A", naux, nocc, nvir), C("C", naux, nvir, nocc);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "Qai<-Qia");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("naux", int64_t(naux));
    ProfileAnnotate("nocc", int64_t(nocc));
    ProfileAnnotate("nvir", int64_t(nvir));
    ProfileAnnotate("elements", int64_t(naux) * nocc * nvir);
    auto t = time_us("permute-r3-3idx", [&] { tensor_algebra::permute(Indices{k, j, i}, &C, Indices{k, i, j}, A); });
    publish_benchmark_result("permute-r3-3idx", "t_permute", naux * 1000 + nocc * 10 + nvir, t);
}

// =====================================================================
// Rank-5 and Rank-6
// =====================================================================

void bench_permute_rank5_cyclic(int N) {
    LabeledSection0();
    Tensor<double, 5> A("A", N, N, N, N, N), C("C", N, N, N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(5));
    ProfileAnnotate("pattern", "eabcd<-abcde");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N * N * N);
    auto t = time_us("permute-r5-cyclic", [&] { tensor_algebra::permute(Indices{e, a, b, c, d}, &C, Indices{a, b, c, d, e}, A); });
    publish_benchmark_result("permute-r5-cyclic", "t_permute", N, t);
}

// Rank-5: swap adjacent pair C(a,c,b,d,e) = A(a,b,c,d,e)
void bench_permute_rank5_swap12(int N) {
    LabeledSection0();
    Tensor<double, 5> A("A", N, N, N, N, N), C("C", N, N, N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(5));
    ProfileAnnotate("pattern", "acbde<-abcde");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N * N * N);
    auto t = time_us("permute-r5-swap12", [&] { tensor_algebra::permute(Indices{a, c, b, d, e}, &C, Indices{a, b, c, d, e}, A); });
    publish_benchmark_result("permute-r5-swap12", "t_permute", N, t);
}

// Rank-5: reverse C(e,d,c,b,a) = A(a,b,c,d,e)
void bench_permute_rank5_reverse(int N) {
    LabeledSection0();
    Tensor<double, 5> A("A", N, N, N, N, N), C("C", N, N, N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(5));
    ProfileAnnotate("pattern", "edcba<-abcde");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N * N * N);
    auto t = time_us("permute-r5-reverse", [&] { tensor_algebra::permute(Indices{e, d, c, b, a}, &C, Indices{a, b, c, d, e}, A); });
    publish_benchmark_result("permute-r5-reverse", "t_permute", N, t);
}

void bench_permute_rank6_pairswap(int N) {
    LabeledSection0();
    Tensor<double, 6> A("A", N, N, N, N, N, N), C("C", N, N, N, N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(6));
    ProfileAnnotate("pattern", "badcfe<-abcdef");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N * N * N * N);
    auto t = time_us("permute-r6-pairswap", [&] { tensor_algebra::permute(Indices{b, a, d, c, f, e}, &C, Indices{a, b, c, d, e, f}, A); });
    publish_benchmark_result("permute-r6-pairswap", "t_permute", N, t);
}

// Rank-6: cyclic C(f,a,b,c,d,e) = A(a,b,c,d,e,f)
void bench_permute_rank6_cyclic(int N) {
    LabeledSection0();
    Tensor<double, 6> A("A", N, N, N, N, N, N), C("C", N, N, N, N, N, N);
    fill(A);
    C.zero();
    ProfileAnnotate("rank", int64_t(6));
    ProfileAnnotate("pattern", "fabcde<-abcdef");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("elements", int64_t(N) * N * N * N * N * N);
    auto t = time_us("permute-r6-cyclic", [&] { tensor_algebra::permute(Indices{f, a, b, c, d, e}, &C, Indices{a, b, c, d, e, f}, A); });
    publish_benchmark_result("permute-r6-cyclic", "t_permute", N, t);
}

// =====================================================================
// Plan caching effect: measures first call (cold) vs subsequent (cached)
// Uses sort+GEMM which internally calls cached_permute
// =====================================================================

void bench_sort_gemm_caching(int N) {
    LabeledSection0();
    Tensor<double, 3> A("A", N, N, N), C("C", N, N, N);
    Tensor<double, 2> B("B", N, N);
    fill(A);
    fill(B);

    // First call (cold, compiles HPTT plan)
    C.zero();
    ProfileAnnotate("cache_state", "cold");
    auto t_cold = time_us(
        "sort-gemm-cold",
        [&] {
            tensor_algebra::einsum(Indices{i, l, j}, &C, Indices{j, k, i}, A, Indices{l, k}, B);
        },
        /*reps=*/1);
    publish_benchmark_result("sort-gemm-cold", "t_sort_gemm", N, t_cold);

    // Subsequent calls (cached plans)
    C.zero();
    ProfileAnnotate("cache_state", "warm");
    auto t_warm = time_us("sort-gemm-cached", [&] { tensor_algebra::einsum(Indices{i, l, j}, &C, Indices{j, k, i}, A, Indices{l, k}, B); });
    publish_benchmark_result("sort-gemm-cached", "t_sort_gemm", N, t_warm);
}

} // namespace

// =====================================================================
// Test cases
// =====================================================================

EINSUMS_TEST_CASE("Permute Rank-2", "[performance][permute]") {
    progress_init(28);
    for (int N : {16, 32, 64, 128, 256, 512, 1024}) {
        bench_permute_rank2_transpose(N);
        bench_permute_rank2_identity(N);
    }
    // Rectangular
    for (auto [M, N] : std::initializer_list<std::pair<int, int>>{{16, 1024}, {1024, 16}, {64, 512}, {512, 64}, {128, 256}}) {
        bench_permute_rank2_rect(M, N);
    }
    // Float and complex at representative sizes
    for (int N : {128, 512}) {
        bench_permute_rank2_float(N);
        bench_permute_rank2_complex(N);
    }
}

EINSUMS_TEST_CASE("Permute Rank-3", "[performance][permute]") {
    progress_init(40);
    for (int N : {8, 16, 32, 64, 128, 192, 256}) {
        bench_permute_rank3_cyclic(N);
        bench_permute_rank3_swap01(N);
        bench_permute_rank3_swap12(N);
        bench_permute_rank3_reverse(N);
    }
    // Beta scaling
    for (int N : {32, 64, 128, 256}) {
        bench_permute_rank3_beta(N);
    }
    // Rectangular rank-3
    for (auto [di, dj, dk] :
         std::initializer_list<std::tuple<int, int, int>>{{10, 10, 100}, {100, 10, 10}, {10, 100, 100}, {50, 50, 200}, {200, 50, 50}}) {
        bench_permute_rank3_rect(di, dj, dk);
    }
    // 3-index integrals (Q, occ, vir)
    for (auto [naux, nocc, nvir] :
         std::initializer_list<std::tuple<int, int, int>>{{100, 10, 50}, {200, 10, 50}, {200, 20, 100}, {400, 10, 50}}) {
        bench_permute_rank3_3idx(naux, nocc, nvir);
    }
}

EINSUMS_TEST_CASE("Permute Rank-4", "[performance][permute]") {
    progress_init(30);
    for (int N : {4, 8, 12, 16, 24, 32, 48, 64}) {
        bench_permute_rank4_reverse(N);
        bench_permute_rank4_inner(N);
        if (N <= 32) {
            bench_permute_rank4_cyclic(N);
            bench_permute_rank4_outerswap(N);
        }
    }
    // Rectangular: chemistry shapes (nocc × nvir)
    for (auto [nocc, nvir] :
         std::initializer_list<std::pair<int, int>>{{5, 20}, {5, 50}, {10, 50}, {10, 100}, {15, 50}, {15, 100}, {20, 80}, {20, 150}}) {
        bench_permute_rank4_rect(nocc, nvir);
    }
}

EINSUMS_TEST_CASE("Permute Rank-5 and Rank-6", "[performance][permute]") {
    progress_init(21);
    for (int N : {4, 6, 8, 10, 12}) {
        bench_permute_rank5_cyclic(N);
        bench_permute_rank5_swap12(N);
        bench_permute_rank5_reverse(N);
    }
    for (int N : {3, 4, 5, 6}) {
        bench_permute_rank6_pairswap(N);
        if (N <= 5) {
            bench_permute_rank6_cyclic(N);
        }
    }
}

EINSUMS_TEST_CASE("Sort+GEMM Plan Caching", "[performance][permute][caching]") {
    progress_init(10);
    for (int N : {16, 32, 64, 96, 128}) {
        bench_sort_gemm_caching(N);
    }
}
