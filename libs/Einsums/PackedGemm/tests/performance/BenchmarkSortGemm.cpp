//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Performance benchmark: Sort (permute) + BLAS GEMM vs. MLIR JIT vs. generic.
//
// Many tensor contractions in quantum chemistry have scrambled index orderings
// that prevent einsum() from dispatching to BLAS GEMM directly.  Two strategies
// exist to accelerate these:
//
//   (a) MLIR JIT backend: compiles a custom kernel for the contraction topology.
//   (b) Sort + GEMM: permute (sort) the input tensors so that target indices
//       are contiguous and link indices are contiguous in the same order,
//       enabling einsum() to dispatch to BLAS GEMM.
//
// This benchmark compares three (or four) paths for each contraction:
//
//   1. t_generic: OpenMP generic algorithm (OnlyUseGenericAlgorithm=true).
//   2. t_packed: MLIR JIT backend called directly.
//   3. t_sort_gemm: permute A and B, then einsum() → BLAS GEMM.
//   4. t_einsum: full einsum() dispatch (shows which path it chose).
//
// Speedup ratios are reported relative to t_generic.

#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
using namespace einsums::performance;

// ---------------------------------------------------------------------------
// Progress bar: Catch2 listener that prints progress to stderr after each test
// ---------------------------------------------------------------------------
namespace {
struct ProgressListener : Catch::EventListenerBase {
    using EventListenerBase::EventListenerBase;

    int total_ = 0;
    int done_  = 0;

    void testRunStarting(Catch::TestRunInfo const &info) override {
        auto const &tests = Catch::getAllTestCasesSorted(*m_config);
        total_            = static_cast<int>(tests.size());
        done_             = 0;
    }

    void testCaseEnded(Catch::TestCaseStats const &stats) override {
        ++done_;
        if (total_ > 0) {
            int pct    = done_ * 100 / total_;
            int filled = pct / 2;
            int empty  = 50 - filled;
            std::fprintf(stderr, "\r  [%.*s%.*s] %3d%% (%d/%d)", filled, "##################################################", empty,
                         "                                                  ", pct, done_, total_);
            std::fflush(stderr);
            if (done_ == total_) {
                std::fprintf(stderr, "\n");
            }
        }
    }
};
} // namespace
CATCH_REGISTER_LISTENER(ProgressListener)

namespace {

/// Call the MLIR JIT backend directly.
template <typename... CI, typename... AI, typename... BI, size_t CR, size_t AR, size_t BR>
bool run_packed_gemm(double beta, std::tuple<CI...> c_idx, einsums::Tensor<double, CR> &C, double alpha, std::tuple<AI...> a_idx,
                     einsums::Tensor<double, AR> const &A, std::tuple<BI...> b_idx, einsums::Tensor<double, BR> const &B) {
    LabeledSection0();
    return einsums::packed_gemm::try_packed_gemm<false, false>(beta, c_idx, &C, alpha, a_idx, A, b_idx, B);
}

/// Call full einsum() and return the algorithm it chose.
template <typename... CI, typename... AI, typename... BI, size_t CR, size_t AR, size_t BR>
einsums::tensor_algebra::detail::AlgorithmChoice run_einsum(double beta, std::tuple<CI...> c_idx, einsums::Tensor<double, CR> &C,
                                                            double alpha, std::tuple<AI...> a_idx, einsums::Tensor<double, AR> const &A,
                                                            std::tuple<BI...> b_idx, einsums::Tensor<double, BR> const &B) {
    LabeledSection0();
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(beta, c_idx, &C, alpha, a_idx, A, b_idx, B, &alg);
    return alg;
}

/// Force the generic OpenMP algorithm, bypassing both BLAS and MLIR.
template <typename... CI, typename... AI, typename... BI, size_t CR, size_t AR, size_t BR>
void run_generic(double beta, std::tuple<CI...> c_idx, einsums::Tensor<double, CR> &C, double alpha, std::tuple<AI...> a_idx,
                 einsums::Tensor<double, AR> const &A, std::tuple<BI...> b_idx, einsums::Tensor<double, BR> const &B) {
    LabeledSection0();
    einsums::tensor_algebra::detail::einsum<true, false, false, false>(beta, c_idx, &C, alpha, a_idx, A, b_idx, B);
}

/// Report all four timing paths using the standard report() format.
void report_paths(char const *label, int N, TimingStats const &s_generic, TimingStats const &s_packed_gemm, TimingStats const &s_sort_gemm,
                  TimingStats const &s_einsum) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "generic-%s", label);
    publish_benchmark_result(buf, "t_generic", N, s_generic);
    std::snprintf(buf, sizeof(buf), "packed-%s", label);
    publish_benchmark_result(buf, "t_packed_gemm", N, s_packed_gemm);
    std::snprintf(buf, sizeof(buf), "sortgemm-%s", label);
    publish_benchmark_result(buf, "t_sortgemm", N, s_sort_gemm);
    std::snprintf(buf, sizeof(buf), "einsum-%s", label);
    publish_benchmark_result(buf, "t_einsum", N, s_einsum);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Rank-3 contraction: C[i,l] += A[i,k,j] * B[l,j,k]
//
// The link indices (j, k) are NOT contiguous in both A and B simultaneously
// (in A: k at pos 1, j at pos 2; in B: j at pos 1, k at pos 2).
// einsum() cannot reshape this to GEMM.
//
// Sort strategy:
//   A_s[i,j,k] = permute A[i,k,j]  (swap j,k)
//   B_s[l,j,k] = B[l,j,k]          (already sorted, j,k contiguous)
//   C[i,l] += A_s[i,j,k] * B_s[l,j,k]  →  BLAS GEMM
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-3 C[i,l]+=A[i,k,j]*B[l,j,k] N=32", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "C[i,l]+=A[i,k,j]*B[l,j,k]");
    ProfileAnnotate("N", int64_t(32));
    ProfileAnnotate("dtype", "double");

    constexpr size_t N = 32;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 3> A{"A", N, N, N}; // dims: i, k, j
    Tensor<double, 3> B{"B", N, N, N}; // dims: l, j, k

    fill(A);
    fill(B);

    // Reference: explicit loops.
    Tensor<double, 2> ref{"ref", N, N};
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t kk = 0; kk < N; ++kk)
                    ref(ii, ll) += A(ii, kk, jj) * B(ll, jj, kk);

    // --- Correctness: MLIR path ---
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B);
    REQUIRE(ok);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            REQUIRE(std::abs(C(ii, ll) - ref(ii, ll)) < 1e-10);

    // --- Correctness: Sort + GEMM path ---
    // Permute A[i,k,j] → A_s[i,j,k] so link indices (j,k) are contiguous.
    Tensor<double, 3> A_s{"A_s", N, N, N};
    tensor_algebra::permute(Indices{i, j, k}, &A_s, Indices{i, k, j}, A);
    // B[l,j,k] already has j,k contiguous; no permute needed.
    C.zero();
    auto sort_alg = run_einsum(0.0, Indices{i, l}, C, 1.0, Indices{i, j, k}, A_s, Indices{l, j, k}, B);
    REQUIRE(sort_alg == tensor_algebra::detail::GEMM);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            REQUIRE(std::abs(C(ii, ll) - ref(ii, ll)) < 1e-10);

    // --- Timing ---
    int reps = 10;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B); }, reps);

    // Sort + GEMM: includes the permute cost.
    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{i, j, k}, &A_s, Indices{i, k, j}, A);
            tensor_algebra::einsum(0.0, Indices{i, l}, &C, 1.0, Indices{i, j, k}, A_s, Indices{l, j, k}, B);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B); }, reps);

    report_paths("Rank-3 C[i,l]+=A[i,k,j]*B[l,j,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

TEST_CASE("Sort+GEMM: rank-3 C[i,l]+=A[i,k,j]*B[l,j,k] N=64", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "C[i,l]+=A[i,k,j]*B[l,j,k]");
    ProfileAnnotate("N", int64_t(64));
    ProfileAnnotate("dtype", "double");

    constexpr size_t N = 64;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 3> A{"A", N, N, N};
    Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    Tensor<double, 3> A_s{"A_s", N, N, N};

    // Warmup MLIR JIT.
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B);
    REQUIRE(ok);

    int reps = 5;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B); }, reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{i, j, k}, &A_s, Indices{i, k, j}, A);
            tensor_algebra::einsum(0.0, Indices{i, l}, &C, 1.0, Indices{i, j, k}, A_s, Indices{l, j, k}, B);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B); }, reps);

    report_paths("Rank-3 C[i,l]+=A[i,k,j]*B[l,j,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-4 contraction: C[i,j] += A[i,l,k,m] * B[m,l,j,k]
//
// Link indices (l, k, m) are non-contiguous in B (j sits at pos 2).
// einsum() cannot reshape to GEMM.
//
// Sort strategy:
//   A_s[i,k,l,m] = permute A[i,l,k,m]  (target i first, link k,l,m contiguous)
//   B_s[j,k,l,m] = permute B[m,l,j,k]  (target j first, link k,l,m contiguous)
//   C[i,j] += A_s[i,k,l,m] * B_s[j,k,l,m]  →  BLAS GEMM
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k] N=8", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(4));
    ProfileAnnotate("pattern", "C[i,j]+=A[i,l,k,m]*B[m,l,j,k]");
    ProfileAnnotate("N", int64_t(8));
    ProfileAnnotate("dtype", "double");

    constexpr size_t N = 8;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 4> A{"A", N, N, N, N}; // i, l, k, m
    Tensor<double, 4> B{"B", N, N, N, N}; // m, l, j, k

    fill(A);
    fill(B);

    // Reference.
    Tensor<double, 2> ref{"ref", N, N};
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t ll = 0; ll < N; ++ll)
                for (size_t kk = 0; kk < N; ++kk)
                    for (size_t mm = 0; mm < N; ++mm)
                        ref(ii, jj) += A(ii, ll, kk, mm) * B(mm, ll, jj, kk);

    // --- Correctness: MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B);
    REQUIRE(ok);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Correctness: Sort + GEMM ---
    Tensor<double, 4> A_s{"A_s", N, N, N, N}; // i, k, l, m
    Tensor<double, 4> B_s{"B_s", N, N, N, N}; // j, k, l, m
    tensor_algebra::permute(Indices{i, k, l, m}, &A_s, Indices{i, l, k, m}, A);
    tensor_algebra::permute(Indices{j, k, l, m}, &B_s, Indices{m, l, j, k}, B);
    C.zero();
    auto sort_alg = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A_s, Indices{j, k, l, m}, B_s);
    REQUIRE(sort_alg == tensor_algebra::detail::GEMM);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Timing ---
    int reps = 10;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B); }, reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{i, k, l, m}, &A_s, Indices{i, l, k, m}, A);
            tensor_algebra::permute(Indices{j, k, l, m}, &B_s, Indices{m, l, j, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m}, A_s, Indices{j, k, l, m}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B); }, reps);

    report_paths("Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

TEST_CASE("Sort+GEMM: rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k] N=16", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(4));
    ProfileAnnotate("pattern", "C[i,j]+=A[i,l,k,m]*B[m,l,j,k]");
    ProfileAnnotate("N", int64_t(16));
    ProfileAnnotate("dtype", "double");

    constexpr size_t N = 16;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 4> A{"A", N, N, N, N};
    Tensor<double, 4> B{"B", N, N, N, N};

    fill(A);
    fill(B);

    Tensor<double, 4> A_s{"A_s", N, N, N, N};
    Tensor<double, 4> B_s{"B_s", N, N, N, N};

    // Warmup MLIR JIT.
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B);
    REQUIRE(ok);

    int reps = 5;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B); }, reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{i, k, l, m}, &A_s, Indices{i, l, k, m}, A);
            tensor_algebra::permute(Indices{j, k, l, m}, &B_s, Indices{m, l, j, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m}, A_s, Indices{j, k, l, m}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B); }, reps);

    report_paths("Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-4 contraction with rectangular dimensions:
// C[i,j] += A[i,k,l,m] * B[j,m,k,l]
//
// Dimensions: i=32, j=32, k=16, l=16, m=8
// Link indices are scrambled in B.
//
// Sort strategy:
//   A_s[i,k,l,m] = A (already sorted: target i first, link k,l,m contiguous)
//   B_s[j,k,l,m] = permute B[j,m,k,l]
//   C[i,j] += A_s[i,k,l,m] * B_s[j,k,l,m]  →  BLAS GEMM
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-4 rectangular C[i,j]+=A[i,k,l,m]*B[j,m,k,l]", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(4));
    ProfileAnnotate("pattern", "C[i,j]+=A[i,k,l,m]*B[j,m,k,l]");
    ProfileAnnotate("N", int64_t(0));
    ProfileAnnotate("dtype", "double");

    constexpr size_t Ni = 32, Nj = 32, Nk = 16, Nl = 16, Nm = 8;

    Tensor<double, 2> C{"C", Ni, Nj};
    Tensor<double, 4> A{"A", Ni, Nk, Nl, Nm}; // i, k, l, m
    Tensor<double, 4> B{"B", Nj, Nm, Nk, Nl}; // j, m, k, l

    fill(A);
    fill(B);

    // Reference.
    Tensor<double, 2> ref{"ref", Ni, Nj};
    ref.zero();
    for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
            for (size_t kk = 0; kk < Nk; ++kk)
                for (size_t ll = 0; ll < Nl; ++ll)
                    for (size_t mm = 0; mm < Nm; ++mm)
                        ref(ii, jj) += A(ii, kk, ll, mm) * B(jj, mm, kk, ll);

    // --- Correctness: MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, m, k, l}, B);
    REQUIRE(ok);
    for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Correctness: Sort + GEMM ---
    // A[i,k,l,m] already has target i first and link k,l,m contiguous; no permute needed.
    // Permute B[j,m,k,l] → B_s[j,k,l,m].
    Tensor<double, 4> B_s{"B_s", Nj, Nk, Nl, Nm};
    tensor_algebra::permute(Indices{j, k, l, m}, &B_s, Indices{j, m, k, l}, B);
    C.zero();
    auto sort_alg = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, k, l, m}, B_s);
    REQUIRE(sort_alg == tensor_algebra::detail::GEMM);
    for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Timing ---
    int reps = 10;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, m, k, l}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, m, k, l}, B); }, reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{j, k, l, m}, &B_s, Indices{j, m, k, l}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m}, A, Indices{j, k, l, m}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, m, k, l}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, m, k, l}, B); }, reps);

    report_paths("Rank-4 rect C[i,j]+=A[i,k,l,m]*B[j,m,k,l] (32x32x16x16x8)", 0, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-3 with all indices scrambled: C[j,i] += A[k,i,l] * B[l,j,k]
//
// Both output and link indices are scrambled.
//
// Sort strategy:
//   A_s[i,k,l] = permute A[k,i,l]  (target i first, link k,l contiguous)
//   B_s[j,k,l] = permute B[l,j,k]  (target j first, link k,l contiguous, same order)
//   C_s[i,j]   (transposed output)
//   C_s[i,j] += A_s[i,k,l] * B_s[j,k,l]  →  BLAS GEMM
//   permute C_s[i,j] → C[j,i]
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-3 scrambled C[j,i]+=A[k,i,l]*B[l,j,k] N=32", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "C[j,i]+=A[k,i,l]*B[l,j,k]");
    ProfileAnnotate("N", int64_t(32));
    ProfileAnnotate("dtype", "double");

    constexpr size_t N = 32;

    Tensor<double, 2> C{"C", N, N};    // j, i
    Tensor<double, 3> A{"A", N, N, N}; // k, i, l
    Tensor<double, 3> B{"B", N, N, N}; // l, j, k

    fill(A);
    fill(B);

    // Reference.
    Tensor<double, 2> ref{"ref", N, N};
    ref.zero();
    for (size_t jj = 0; jj < N; ++jj)
        for (size_t ii = 0; ii < N; ++ii)
            for (size_t kk = 0; kk < N; ++kk)
                for (size_t ll = 0; ll < N; ++ll)
                    ref(jj, ii) += A(kk, ii, ll) * B(ll, jj, kk);

    // --- Correctness: MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B);
    REQUIRE(ok);
    for (size_t jj = 0; jj < N; ++jj)
        for (size_t ii = 0; ii < N; ++ii)
            REQUIRE(std::abs(C(jj, ii) - ref(jj, ii)) < 1e-10);

    // --- Correctness: Sort + GEMM ---
    Tensor<double, 3> A_s{"A_s", N, N, N}; // i, k, l
    Tensor<double, 3> B_s{"B_s", N, N, N}; // j, k, l
    Tensor<double, 2> C_s{"C_s", N, N};    // i, j
    tensor_algebra::permute(Indices{i, k, l}, &A_s, Indices{k, i, l}, A);
    tensor_algebra::permute(Indices{j, k, l}, &B_s, Indices{l, j, k}, B);
    C_s.zero();
    auto sort_alg = run_einsum(0.0, Indices{i, j}, C_s, 1.0, Indices{i, k, l}, A_s, Indices{j, k, l}, B_s);
    REQUIRE(sort_alg == tensor_algebra::detail::GEMM);
    // Permute C_s[i,j] → C[j,i].
    tensor_algebra::permute(Indices{j, i}, &C, Indices{i, j}, C_s);
    for (size_t jj = 0; jj < N; ++jj)
        for (size_t ii = 0; ii < N; ++ii)
            REQUIRE(std::abs(C(jj, ii) - ref(jj, ii)) < 1e-10);

    // --- Timing ---
    int reps = 10;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B); }, reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{i, k, l}, &A_s, Indices{k, i, l}, A);
            tensor_algebra::permute(Indices{j, k, l}, &B_s, Indices{l, j, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C_s, 1.0, Indices{i, k, l}, A_s, Indices{j, k, l}, B_s);
            tensor_algebra::permute(Indices{j, i}, &C, Indices{i, j}, C_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B); }, reps);

    report_paths("Rank-3 scrambled C[j,i]+=A[k,i,l]*B[l,j,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

TEST_CASE("Sort+GEMM: rank-3 scrambled C[j,i]+=A[k,i,l]*B[l,j,k] N=64", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "C[j,i]+=A[k,i,l]*B[l,j,k]");
    ProfileAnnotate("N", int64_t(64));
    ProfileAnnotate("dtype", "double");

    constexpr size_t N = 64;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 3> A{"A", N, N, N};
    Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    Tensor<double, 3> A_s{"A_s", N, N, N};
    Tensor<double, 3> B_s{"B_s", N, N, N};
    Tensor<double, 2> C_s{"C_s", N, N};

    // Warmup MLIR JIT.
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B);
    REQUIRE(ok);

    int reps = 5;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B); }, reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{i, k, l}, &A_s, Indices{k, i, l}, A);
            tensor_algebra::permute(Indices{j, k, l}, &B_s, Indices{l, j, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C_s, 1.0, Indices{i, k, l}, A_s, Indices{j, k, l}, B_s);
            tensor_algebra::permute(Indices{j, i}, &C, Indices{i, j}, C_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B); }, reps);

    report_paths("Rank-3 scrambled C[j,i]+=A[k,i,l]*B[l,j,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Larger N sizes for rank-3 and rank-4
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-3 C[i,l]+=A[i,k,j]*B[l,j,k] N=128", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "C[i,l]+=A[i,k,j]*B[l,j,k]");
    ProfileAnnotate("N", int64_t(128));
    ProfileAnnotate("dtype", "double");

    constexpr size_t N = 128;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 3> A{"A", N, N, N};
    Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    Tensor<double, 3> A_s{"A_s", N, N, N};

    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B);
    REQUIRE(ok);

    int reps = 3;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B); }, reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{i, j, k}, &A_s, Indices{i, k, j}, A);
            tensor_algebra::einsum(0.0, Indices{i, l}, &C, 1.0, Indices{i, j, k}, A_s, Indices{l, j, k}, B);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, l}, C, 1.0, Indices{i, k, j}, A, Indices{l, j, k}, B); }, reps);

    report_paths("Rank-3 C[i,l]+=A[i,k,j]*B[l,j,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

TEST_CASE("Sort+GEMM: rank-3 scrambled C[j,i]+=A[k,i,l]*B[l,j,k] N=128", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "C[j,i]+=A[k,i,l]*B[l,j,k]");
    ProfileAnnotate("N", int64_t(128));
    ProfileAnnotate("dtype", "double");

    constexpr size_t N = 128;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 3> A{"A", N, N, N};
    Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    Tensor<double, 3> A_s{"A_s", N, N, N};
    Tensor<double, 3> B_s{"B_s", N, N, N};
    Tensor<double, 2> C_s{"C_s", N, N};

    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B);
    REQUIRE(ok);

    int reps = 3;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B); }, reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{i, k, l}, &A_s, Indices{k, i, l}, A);
            tensor_algebra::permute(Indices{j, k, l}, &B_s, Indices{l, j, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C_s, 1.0, Indices{i, k, l}, A_s, Indices{j, k, l}, B_s);
            tensor_algebra::permute(Indices{j, i}, &C, Indices{i, j}, C_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{j, i}, C, 1.0, Indices{k, i, l}, A, Indices{l, j, k}, B); }, reps);

    report_paths("Rank-3 scrambled C[j,i]+=A[k,i,l]*B[l,j,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

TEST_CASE("Sort+GEMM: rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k] N=32", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(4));
    ProfileAnnotate("pattern", "C[i,j]+=A[i,l,k,m]*B[m,l,j,k]");
    ProfileAnnotate("N", int64_t(32));
    ProfileAnnotate("dtype", "double");

    constexpr size_t N = 32;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 4> A{"A", N, N, N, N};
    Tensor<double, 4> B{"B", N, N, N, N};

    fill(A);
    fill(B);

    Tensor<double, 4> A_s{"A_s", N, N, N, N};
    Tensor<double, 4> B_s{"B_s", N, N, N, N};

    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B);
    REQUIRE(ok);

    int reps = 3;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B); }, reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{i, k, l, m}, &A_s, Indices{i, l, k, m}, A);
            tensor_algebra::permute(Indices{j, k, l, m}, &B_s, Indices{m, l, j, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m}, A_s, Indices{j, k, l, m}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, l, k, m}, A, Indices{m, l, j, k}, B); }, reps);

    report_paths("Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-4 rectangular with larger dimensions
// C[i,j] += A[i,k,l,m] * B[j,m,k,l]
// Dimensions: i=64, j=64, k=32, l=16, m=8
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-4 rect C[i,j]+=A[i,k,l,m]*B[j,m,k,l] (64x64x32x16x8)", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(4));
    ProfileAnnotate("pattern", "C[i,j]+=A[i,k,l,m]*B[j,m,k,l]");
    ProfileAnnotate("N", int64_t(0));
    ProfileAnnotate("dtype", "double");

    constexpr size_t Ni = 64, Nj = 64, Nk = 32, Nl = 16, Nm = 8;

    Tensor<double, 2> C{"C", Ni, Nj};
    Tensor<double, 4> A{"A", Ni, Nk, Nl, Nm};
    Tensor<double, 4> B{"B", Nj, Nm, Nk, Nl};

    fill(A);
    fill(B);

    Tensor<double, 4> B_s{"B_s", Nj, Nk, Nl, Nm};

    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, m, k, l}, B);
    REQUIRE(ok);

    int reps = 3;

    auto t_generic = time_us([&]() { run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, m, k, l}, B); }, reps);

    auto t_packed = time_us([&]() { run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, m, k, l}, B); }, reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{j, k, l, m}, &B_s, Indices{j, m, k, l}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m}, A, Indices{j, k, l, m}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, m, k, l}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m}, A, Indices{j, m, k, l}, B); }, reps);

    report_paths("Rank-4 rect C[i,j]+=A[i,k,l,m]*B[j,m,k,l] (64x64x32x16x8)", 0, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-5 contraction: C[i,j] += A[i,k,l,m,n] * B[j,n,m,l,k]
//
// 4 link indices (k,l,m,n) fully scrambled in B.
//
// Sort strategy:
//   A[i,k,l,m,n] already has target i first, link contiguous; no sort needed.
//   B_s[j,k,l,m,n] = permute B[j,n,m,l,k]
//   C[i,j] += A[i,k,l,m,n] * B_s[j,k,l,m,n]  →  BLAS GEMM
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-5 C[i,j]+=A[i,k,l,m,n]*B[j,n,m,l,k] N=8", "[mlir][benchmark][sort]") {
    LabeledSection0();
    ProfileAnnotate("rank", int64_t(5));
    ProfileAnnotate("pattern", "C[i,j]+=A[i,k,l,m,n]*B[j,n,m,l,k]");
    ProfileAnnotate("N", int64_t(8));
    ProfileAnnotate("dtype", "double");

    constexpr size_t N = 8;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 5> A{"A", N, N, N, N, N}; // i, k, l, m, n
    Tensor<double, 5> B{"B", N, N, N, N, N}; // j, n, m, l, k

    fill(A);
    fill(B);

    // Reference.
    Tensor<double, 2> ref{"ref", N, N};
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t kk = 0; kk < N; ++kk)
                for (size_t ll = 0; ll < N; ++ll)
                    for (size_t mm = 0; mm < N; ++mm)
                        for (size_t nn = 0; nn < N; ++nn)
                            ref(ii, jj) += A(ii, kk, ll, mm, nn) * B(jj, nn, mm, ll, kk);

    // --- Correctness: MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
    REQUIRE(ok);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Correctness: Sort + GEMM ---
    Tensor<double, 5> B_s{"B_s", N, N, N, N, N}; // j, k, l, m, n
    tensor_algebra::permute(Indices{j, k, l, m, n}, &B_s, Indices{j, n, m, l, k}, B);
    C.zero();
    auto sort_alg = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, k, l, m, n}, B_s);
    REQUIRE(sort_alg == tensor_algebra::detail::GEMM);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Timing ---
    int reps = 10;

    auto t_generic = time_us(
        [&]() {
            run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
        },
        reps);

    auto t_packed = time_us(
        [&]() {
            run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
        },
        reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{j, k, l, m, n}, &B_s, Indices{j, n, m, l, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, k, l, m, n}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B); }, reps);

    report_paths("Rank-5 C[i,j]+=A[i,k,l,m,n]*B[j,n,m,l,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

TEST_CASE("Sort+GEMM: rank-5 C[i,j]+=A[i,k,l,m,n]*B[j,n,m,l,k] N=16", "[mlir][benchmark][sort]") {
    LabeledSection0();

    constexpr size_t N = 16;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 5> A{"A", N, N, N, N, N};
    Tensor<double, 5> B{"B", N, N, N, N, N};

    fill(A);
    fill(B);

    Tensor<double, 5> B_s{"B_s", N, N, N, N, N};

    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
    REQUIRE(ok);

    int reps = 5;

    auto t_generic = time_us(
        [&]() {
            run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
        },
        reps);

    auto t_packed = time_us(
        [&]() {
            run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
        },
        reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{j, k, l, m, n}, &B_s, Indices{j, n, m, l, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, k, l, m, n}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B); }, reps);

    report_paths("Rank-5 C[i,j]+=A[i,k,l,m,n]*B[j,n,m,l,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-5 rectangular: C[i,j] += A[i,k,l,m,n] * B[j,n,m,l,k]
// Dimensions: i=32, j=32, k=16, l=8, m=8, n=4
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-5 rect C[i,j]+=A[i,k,l,m,n]*B[j,n,m,l,k] (32x32x16x8x8x4)", "[mlir][benchmark][sort]") {
    LabeledSection0();

    constexpr size_t Ni = 32, Nj = 32, Nk = 16, Nl = 8, Nm = 8, Nn = 4;

    Tensor<double, 2> C{"C", Ni, Nj};
    Tensor<double, 5> A{"A", Ni, Nk, Nl, Nm, Nn}; // i, k, l, m, n
    Tensor<double, 5> B{"B", Nj, Nn, Nm, Nl, Nk}; // j, n, m, l, k

    fill(A);
    fill(B);

    // Reference.
    Tensor<double, 2> ref{"ref", Ni, Nj};
    ref.zero();
    for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
            for (size_t kk = 0; kk < Nk; ++kk)
                for (size_t ll = 0; ll < Nl; ++ll)
                    for (size_t mm = 0; mm < Nm; ++mm)
                        for (size_t nn = 0; nn < Nn; ++nn)
                            ref(ii, jj) += A(ii, kk, ll, mm, nn) * B(jj, nn, mm, ll, kk);

    // --- Correctness: MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
    REQUIRE(ok);
    for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Correctness: Sort + GEMM ---
    Tensor<double, 5> B_s{"B_s", Nj, Nk, Nl, Nm, Nn};
    tensor_algebra::permute(Indices{j, k, l, m, n}, &B_s, Indices{j, n, m, l, k}, B);
    C.zero();
    auto sort_alg = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, k, l, m, n}, B_s);
    REQUIRE(sort_alg == tensor_algebra::detail::GEMM);
    for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Timing ---
    int reps = 5;

    auto t_generic = time_us(
        [&]() {
            run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
        },
        reps);

    auto t_packed = time_us(
        [&]() {
            run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
        },
        reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{j, k, l, m, n}, &B_s, Indices{j, n, m, l, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, k, l, m, n}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n}, A, Indices{j, n, m, l, k}, B); }, reps);

    report_paths("Rank-5 rect C[i,j]+=A[i,k,l,m,n]*B[j,n,m,l,k] (32x32x16x8x8x4)", 0, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-5 scrambled output: C[j,i] += A[l,i,k,m,n] * B[n,j,m,l,k]
//
// Both output and link indices are scrambled.
//
// Sort strategy:
//   A_s[i,k,l,m,n] = permute A[l,i,k,m,n]
//   B_s[j,k,l,m,n] = permute B[n,j,m,l,k]
//   C_s[i,j] += A_s[i,k,l,m,n] * B_s[j,k,l,m,n]  →  BLAS GEMM
//   permute C_s[i,j] → C[j,i]
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-5 scrambled C[j,i]+=A[l,i,k,m,n]*B[n,j,m,l,k] N=8", "[mlir][benchmark][sort]") {
    LabeledSection0();

    constexpr size_t N = 8;

    Tensor<double, 2> C{"C", N, N};          // j, i
    Tensor<double, 5> A{"A", N, N, N, N, N}; // l, i, k, m, n
    Tensor<double, 5> B{"B", N, N, N, N, N}; // n, j, m, l, k

    fill(A);
    fill(B);

    // Reference.
    Tensor<double, 2> ref{"ref", N, N};
    ref.zero();
    for (size_t jj = 0; jj < N; ++jj)
        for (size_t ii = 0; ii < N; ++ii)
            for (size_t kk = 0; kk < N; ++kk)
                for (size_t ll = 0; ll < N; ++ll)
                    for (size_t mm = 0; mm < N; ++mm)
                        for (size_t nn = 0; nn < N; ++nn)
                            ref(jj, ii) += A(ll, ii, kk, mm, nn) * B(nn, jj, mm, ll, kk);

    // --- Correctness: MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{j, i}, C, 1.0, Indices{l, i, k, m, n}, A, Indices{n, j, m, l, k}, B);
    REQUIRE(ok);
    for (size_t jj = 0; jj < N; ++jj)
        for (size_t ii = 0; ii < N; ++ii)
            REQUIRE(std::abs(C(jj, ii) - ref(jj, ii)) < 1e-10);

    // --- Timing ---
    int reps = 10;

    auto t_generic = time_us(
        [&]() {
            run_generic(0.0, Indices{j, i}, C, 1.0, Indices{l, i, k, m, n}, A, Indices{n, j, m, l, k}, B);
        },
        reps);

    auto t_packed = time_us(
        [&]() {
            run_packed_gemm(0.0, Indices{j, i}, C, 1.0, Indices{l, i, k, m, n}, A, Indices{n, j, m, l, k}, B);
        },
        reps);

    // Sort + GEMM.
    Tensor<double, 5> A_s{"A_s", N, N, N, N, N}; // i, k, l, m, n
    Tensor<double, 5> B_s{"B_s", N, N, N, N, N}; // j, k, l, m, n
    Tensor<double, 2> C_s{"C_s", N, N};          // i, j
    auto              t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{i, k, l, m, n}, &A_s, Indices{l, i, k, m, n}, A);
            tensor_algebra::permute(Indices{j, k, l, m, n}, &B_s, Indices{n, j, m, l, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C_s, 1.0, Indices{i, k, l, m, n}, A_s, Indices{j, k, l, m, n}, B_s);
            tensor_algebra::permute(Indices{j, i}, &C, Indices{i, j}, C_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{j, i}, C, 1.0, Indices{l, i, k, m, n}, A, Indices{n, j, m, l, k}, B);
    auto t_einsum = time_us([&]() { run_einsum(0.0, Indices{j, i}, C, 1.0, Indices{l, i, k, m, n}, A, Indices{n, j, m, l, k}, B); }, reps);

    report_paths("Rank-5 scrambled C[j,i]+=A[l,i,k,m,n]*B[n,j,m,l,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-6 contraction: C[i,j] += A[i,k,l,m,n,o] * B[j,o,n,m,l,k]
//
// 5 link indices fully scrambled in B.
//
// Sort strategy:
//   A already sorted (target i first, link contiguous).
//   B_s[j,k,l,m,n,o] = permute B[j,o,n,m,l,k]
//   C[i,j] += A[i,k,l,m,n,o] * B_s[j,k,l,m,n,o]  →  BLAS GEMM
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-6 C[i,j]+=A[i,k,l,m,n,o]*B[j,o,n,m,l,k] N=6", "[mlir][benchmark][sort]") {
    LabeledSection0();

    constexpr size_t N = 6;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 6> A{"A", N, N, N, N, N, N}; // i, k, l, m, n, o
    Tensor<double, 6> B{"B", N, N, N, N, N, N}; // j, o, n, m, l, k

    fill(A);
    fill(B);

    // Reference.
    Tensor<double, 2> ref{"ref", N, N};
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t kk = 0; kk < N; ++kk)
                for (size_t ll = 0; ll < N; ++ll)
                    for (size_t mm = 0; mm < N; ++mm)
                        for (size_t nn = 0; nn < N; ++nn)
                            for (size_t oo = 0; oo < N; ++oo)
                                ref(ii, jj) += A(ii, kk, ll, mm, nn, oo) * B(jj, oo, nn, mm, ll, kk);

    // --- Correctness: MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
    REQUIRE(ok);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Correctness: Sort + GEMM ---
    Tensor<double, 6> B_s{"B_s", N, N, N, N, N, N}; // j, k, l, m, n, o
    tensor_algebra::permute(Indices{j, k, l, m, n, o}, &B_s, Indices{j, o, n, m, l, k}, B);
    C.zero();
    auto sort_alg = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, k, l, m, n, o}, B_s);
    REQUIRE(sort_alg == tensor_algebra::detail::GEMM);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Timing ---
    int reps = 10;

    auto t_generic = time_us(
        [&]() {
            run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
        },
        reps);

    auto t_packed = time_us(
        [&]() {
            run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
        },
        reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{j, k, l, m, n, o}, &B_s, Indices{j, o, n, m, l, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, k, l, m, n, o}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
    auto t_einsum = time_us(
        [&]() {
            run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
        },
        reps);

    report_paths("Rank-6 C[i,j]+=A[i,k,l,m,n,o]*B[j,o,n,m,l,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

TEST_CASE("Sort+GEMM: rank-6 C[i,j]+=A[i,k,l,m,n,o]*B[j,o,n,m,l,k] N=8", "[mlir][benchmark][sort]") {
    LabeledSection0();

    constexpr size_t N = 8;

    Tensor<double, 2> C{"C", N, N};
    Tensor<double, 6> A{"A", N, N, N, N, N, N};
    Tensor<double, 6> B{"B", N, N, N, N, N, N};

    fill(A);
    fill(B);

    Tensor<double, 6> B_s{"B_s", N, N, N, N, N, N};

    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
    REQUIRE(ok);

    int reps = 5;

    auto t_generic = time_us(
        [&]() {
            run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
        },
        reps);

    auto t_packed = time_us(
        [&]() {
            run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
        },
        reps);

    auto t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{j, k, l, m, n, o}, &B_s, Indices{j, o, n, m, l, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, k, l, m, n, o}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
    auto t_einsum = time_us(
        [&]() {
            run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
        },
        reps);

    report_paths("Rank-6 C[i,j]+=A[i,k,l,m,n,o]*B[j,o,n,m,l,k]", N, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-6 rectangular: C[i,j] += A[i,k,l,m,n,o] * B[j,o,n,m,l,k]
// Dimensions: i=16, j=16, k=8, l=8, m=4, n=4, o=4
// ---------------------------------------------------------------------------

TEST_CASE("Sort+GEMM: rank-6 rect C[i,j]+=A[i,k,l,m,n,o]*B[j,o,n,m,l,k] (16x16x8x8x4x4x4)", "[mlir][benchmark][sort]") {
    LabeledSection0();

    constexpr size_t Ni = 16, Nj = 16, Nk = 8, Nl = 8, Nm = 4, Nn = 4, No = 4;

    Tensor<double, 2> C{"C", Ni, Nj};
    Tensor<double, 6> A{"A", Ni, Nk, Nl, Nm, Nn, No}; // i, k, l, m, n, o
    Tensor<double, 6> B{"B", Nj, No, Nn, Nm, Nl, Nk}; // j, o, n, m, l, k

    fill(A);
    fill(B);

    // Reference.
    Tensor<double, 2> ref{"ref", Ni, Nj};
    ref.zero();
    for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
            for (size_t kk = 0; kk < Nk; ++kk)
                for (size_t ll = 0; ll < Nl; ++ll)
                    for (size_t mm = 0; mm < Nm; ++mm)
                        for (size_t nn = 0; nn < Nn; ++nn)
                            for (size_t oo = 0; oo < No; ++oo)
                                ref(ii, jj) += A(ii, kk, ll, mm, nn, oo) * B(jj, oo, nn, mm, ll, kk);

    // --- Correctness: MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
    REQUIRE(ok);
    for (size_t ii = 0; ii < Ni; ++ii)
        for (size_t jj = 0; jj < Nj; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Timing ---
    int reps = 5;

    auto t_generic = time_us(
        [&]() {
            run_generic(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
        },
        reps);

    auto t_packed = time_us(
        [&]() {
            run_packed_gemm(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
        },
        reps);

    // Sort + GEMM.
    Tensor<double, 6> B_s{"B_s", Nj, Nk, Nl, Nm, Nn, No};
    auto              t_sort_gemm = time_us(
        [&]() {
            tensor_algebra::permute(Indices{j, k, l, m, n, o}, &B_s, Indices{j, o, n, m, l, k}, B);
            tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, k, l, m, n, o}, B_s);
        },
        reps);

    auto alg      = run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
    auto t_einsum = time_us(
        [&]() {
            run_einsum(0.0, Indices{i, j}, C, 1.0, Indices{i, k, l, m, n, o}, A, Indices{j, o, n, m, l, k}, B);
        },
        reps);

    report_paths("Rank-6 rect C[i,j]+=A[i,k,l,m,n,o]*B[j,o,n,m,l,k] (16x16x8x8x4x4x4)", 0, t_generic, t_packed, t_sort_gemm, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}
