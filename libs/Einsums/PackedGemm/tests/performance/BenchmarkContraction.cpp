//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Performance benchmark: MLIR JIT kernels for tensor contractions.
//
// Each test reports three timings:
//   1. t_generic     — OpenMP generic algorithm only (pre-MLIR einsum behavior for
//                      non-BLAS contractions; for rank-2 this bypasses BLAS).
//   2. t_packed        — MLIR JIT backend called directly (Pack-A/Pack-B + JIT macro-kernel).
//   3. t_einsum      — Full einsum() dispatch; the algorithm label in the output
//                      shows which path einsum() actually chose (BLAS-GEMM, BLAS-GEMV,
//                      generic/MLIR, etc.).
//
// Speedup ratios reported relative to t_generic:
//   MLIR/gen  : t_generic / t_packed        (>1 means MLIR is faster than generic)
//   einsum/gen: t_generic / t_einsum      (>1 means einsum is faster than generic)

// TensorAlgebra.hpp must come first — it defines the einsums::index namespace
// containing the index tag types (i, j, k, l, ...) used below.
#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
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
        // Count total test cases at the start
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

/// Call the MLIR JIT backend directly for: C = beta*C + alpha * A[...] * B[...].
/// Returns true if the MLIR kernel compiled and ran; false on fallback.
template <typename... CI, typename... AI, typename... BI, size_t CR, size_t AR, size_t BR>
bool run_packed_gemm(double beta, std::tuple<CI...> c_idx, einsums::Tensor<double, CR> &C, double alpha, std::tuple<AI...> a_idx,
                     einsums::Tensor<double, AR> const &A, std::tuple<BI...> b_idx, einsums::Tensor<double, BR> const &B) {
    return einsums::packed_gemm::try_packed_gemm<false, false>(beta, c_idx, &C, alpha, a_idx, A, b_idx, B);
}

/// Call the full einsum() dispatcher and return the algorithm it chose.
/// For rank-2 contractions einsum() typically uses BLAS (GEMM/GEMV/etc.).
/// For rank-3+ contractions that cannot be reshaped to GEMM it uses
/// the MLIR JIT backend (via einsum_generic_default<false,...>).
template <typename... CI, typename... AI, typename... BI, size_t CR, size_t AR, size_t BR>
einsums::tensor_algebra::detail::AlgorithmChoice run_einsum(double beta, std::tuple<CI...> c_idx, einsums::Tensor<double, CR> &C,
                                                            double alpha, std::tuple<AI...> a_idx, einsums::Tensor<double, AR> const &A,
                                                            std::tuple<BI...> b_idx, einsums::Tensor<double, BR> const &B) {
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(beta, c_idx, &C, alpha, a_idx, A, b_idx, B, &alg);
    return alg;
}

/// Force the generic OpenMP algorithm, bypassing both BLAS and MLIR.
/// For rank-3+ contractions this matches the pre-MLIR einsum behavior.
/// For rank-2 contractions this bypasses BLAS and is slower than einsum().
template <typename... CI, typename... AI, typename... BI, size_t CR, size_t AR, size_t BR>
void run_generic(double beta, std::tuple<CI...> c_idx, einsums::Tensor<double, CR> &C, double alpha, std::tuple<AI...> a_idx,
                 einsums::Tensor<double, AR> const &A, std::tuple<BI...> b_idx, einsums::Tensor<double, BR> const &B) {
    einsums::tensor_algebra::detail::einsum<true, false, false, false>(beta, c_idx, &C, alpha, a_idx, A, b_idx, B);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Rank-2 contraction: C[i,j] = A[i,k] * B[j,k]
//
// In matrix notation this is C = A * B^T.  einsum() routes this through BLAS
// GEMM (TRANSB='T').  We call MLIR directly via try_packed_gemm() to show
// what MLIR would produce even though einsum() bypasses it here.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: rank-2 contraction N=64 (MLIR direct)", "[mlir][benchmark]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,j]+=A[i,k]*B[j,k]");
    ProfileAnnotate("N", int64_t(64));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 64;
    einsums::Tensor<double, 2> A{"A", N, N};
    einsums::Tensor<double, 2> B{"B", N, N};
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 2> ref{"ref", N, N};

    fill(A);
    fill(B);

    // --- Reference: explicit nested loops (correctness only, not timed) ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t kk = 0; kk < N; ++kk)
                ref(ii, jj) += A(ii, kk) * B(jj, kk);

    // --- MLIR: try_packed_gemm correctly declines rank-2 GEMM (defers to BLAS) ---
    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A,
                              std::tuple<struct j, struct k>{}, B);
    // Rank-2 single-M/single-N/single-K is a standard GEMM — MLIR correctly defers to BLAS dispatch.
    REQUIRE_FALSE(ok);

    // --- Timing: generic vs einsum (MLIR not applicable for this shape) ---
    auto t_generic = time_us("generic", [&]() {
        run_generic(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A, std::tuple<struct j, struct k>{},
                    B);
    });

    auto alg =
        run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A, std::tuple<struct j, struct k>{}, B);
    auto t_einsum = time_us("einsum", [&]() {
        run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A, std::tuple<struct j, struct k>{}, B);
    });

    // --- Correctness: verify einsum produces correct result ---
    C.zero();
    run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A, std::tuple<struct j, struct k>{}, B);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    publish_benchmark_result("generic-Rank-2", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Rank-2", "t_einsum", N, t_einsum);
}

// ---------------------------------------------------------------------------
// Rank-3 contraction: C[i,l] += A[i,j,k] * B[j,k,l]
//
// einsum() may route this through BLAS GEMM by reshaping A to [i, j*k] and
// B to [j*k, l] (contiguous link indices allow this).  We call MLIR directly
// to show the MLIR JIT performance for this contraction.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: rank-3 contraction N=32 (MLIR direct)", "[mlir][benchmark]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,l]+=A[i,j,k]*B[j,k,l]");
    ProfileAnnotate("N", int64_t(32));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 32;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};
    einsums::Tensor<double, 2> ref{"ref", N, N};

    fill(A);
    fill(B);

    // --- Reference (correctness only, not timed) ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t kk = 0; kk < N; ++kk)
                    ref(ii, ll) += A(ii, jj, kk) * B(jj, kk, ll);

    // --- MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                              std::tuple<struct j, struct k, struct l>{}, B);
    REQUIRE(ok);

    // --- Correctness check ---
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            REQUIRE(std::abs(C(ii, ll) - ref(ii, ll)) < 1e-10);

    // --- Timing ---
    auto t_generic = time_us("generic", [&]() {
        run_generic(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                    std::tuple<struct j, struct k, struct l>{}, B);
    });

    auto t_packed = time_us("packed_gemm", [&]() {
        run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                        std::tuple<struct j, struct k, struct l>{}, B);
    });

    auto alg      = run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                               std::tuple<struct j, struct k, struct l>{}, B);
    auto t_einsum = time_us("einsum", [&]() {
        run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                   std::tuple<struct j, struct k, struct l>{}, B);
    });

    publish_benchmark_result("generic-Rank-3", "t_generic", N, t_generic);
    publish_benchmark_result("packed-Rank-3", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-Rank-3", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-3 contraction N=64 — larger problem size
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: rank-3 contraction N=64 (MLIR direct)", "[mlir][benchmark]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,l]+=A[i,j,k]*B[j,k,l]");
    ProfileAnnotate("N", int64_t(64));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 64;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};
    einsums::Tensor<double, 2> ref{"ref", N, N};

    fill(A);
    fill(B);

    // --- Reference (correctness only, not timed) ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t kk = 0; kk < N; ++kk)
                    ref(ii, ll) += A(ii, jj, kk) * B(jj, kk, ll);

    // --- MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                              std::tuple<struct j, struct k, struct l>{}, B);
    REQUIRE(ok);

    // --- Correctness check ---
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            REQUIRE(std::abs(C(ii, ll) - ref(ii, ll)) < 1e-10);

    // --- Timing (fewer reps for the larger sizes) ---
    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                        std::tuple<struct j, struct k, struct l>{}, B);
        },
        5);

    auto t_packed = time_us(
        "packed_gemm",
        [&]() {
            run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                            std::tuple<struct j, struct k, struct l>{}, B);
        },
        5);

    auto alg      = run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                               std::tuple<struct j, struct k, struct l>{}, B);
    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                       std::tuple<struct j, struct k, struct l>{}, B);
        },
        5);

    publish_benchmark_result("generic-Rank-3", "t_generic", N, t_generic);
    publish_benchmark_result("packed-Rank-3", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-Rank-3", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-4 contraction: C[i,j] += A[i,l,k,m] * B[m,l,j,k]
//
// Link indices: l, k, m  (three contracted dimensions).
// The link indices are non-contiguous in B (m at pos 0, l at pos 1, k at pos 3
// but j at pos 2), so einsum() cannot reshape this to GEMM.  It falls through
// to einsum_generic_default which tries the MLIR JIT backend first.
//
// N=8 gives 8^5 = 32768 FLOPs — fast enough for correctness checking.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: rank-4 contraction C[i,j]+=A[i,l,k,m]*B[m,l,j,k] N=8 (MLIR direct)", "[mlir][benchmark]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,j]+=A[i,l,k,m]*B[m,l,j,k]");
    ProfileAnnotate("N", int64_t(8));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 8;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 4> A{"A", N, N, N, N}; // i, l, k, m
    einsums::Tensor<double, 4> B{"B", N, N, N, N}; // m, l, j, k
    einsums::Tensor<double, 2> ref{"ref", N, N};

    fill(A);
    fill(B);

    // --- Reference (correctness only, not timed) ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t ll = 0; ll < N; ++ll)
                for (size_t kk = 0; kk < N; ++kk)
                    for (size_t mm = 0; mm < N; ++mm)
                        ref(ii, jj) += A(ii, ll, kk, mm) * B(mm, ll, jj, kk);

    // --- MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct l, struct k, struct m>{}, A,
                              std::tuple<struct m, struct l, struct j, struct k>{}, B);
    REQUIRE(ok);

    // --- Correctness check ---
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Timing (beta=0.0 resets C each call; no separate C.zero() needed) ---
    auto t_generic = time_us("generic", [&]() {
        run_generic(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct l, struct k, struct m>{}, A,
                    std::tuple<struct m, struct l, struct j, struct k>{}, B);
    });

    auto t_packed = time_us("packed_gemm", [&]() {
        run_packed_gemm(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct l, struct k, struct m>{}, A,
                        std::tuple<struct m, struct l, struct j, struct k>{}, B);
    });

    auto alg      = run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct l, struct k, struct m>{}, A,
                               std::tuple<struct m, struct l, struct j, struct k>{}, B);
    auto t_einsum = time_us("einsum", [&]() {
        run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct l, struct k, struct m>{}, A,
                   std::tuple<struct m, struct l, struct j, struct k>{}, B);
    });

    publish_benchmark_result("generic-Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k]", "t_generic", N, t_generic);
    publish_benchmark_result("packed-Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k]", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k]", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-2 N=256 — GEMM-equivalent large-N (vs BLAS target: MLIR ≤ 2× BLAS)
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: rank-2 contraction N=256 (MLIR direct)", "[mlir][benchmark][large]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,j]+=A[i,k]*B[j,k]");
    ProfileAnnotate("N", int64_t(256));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 256;
    einsums::Tensor<double, 2> A{"A", N, N};
    einsums::Tensor<double, 2> B{"B", N, N};
    einsums::Tensor<double, 2> C{"C", N, N};

    fill(A);
    fill(B);

    // Rank-2 single-M/single-N/single-K: MLIR correctly defers to BLAS dispatch.
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A,
                              std::tuple<struct j, struct k>{}, B);
    REQUIRE_FALSE(ok);

    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A,
                        std::tuple<struct j, struct k>{}, B);
        },
        5);

    auto alg =
        run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A, std::tuple<struct j, struct k>{}, B);
    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A, std::tuple<struct j, struct k>{},
                       B);
        },
        5);

    publish_benchmark_result("generic-Rank-2", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Rank-2", "t_einsum", N, t_einsum);
}

// ---------------------------------------------------------------------------
// Rank-3 N=128 — first tiling-benefiting size (link_dim=128 > tile_size=64)
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: rank-3 contraction N=128 (MLIR direct)", "[mlir][benchmark][large]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,l]+=A[i,j,k]*B[j,k,l]");
    ProfileAnnotate("N", int64_t(128));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 128;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                              std::tuple<struct j, struct k, struct l>{}, B);
    REQUIRE(ok);

    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                        std::tuple<struct j, struct k, struct l>{}, B);
        },
        3);

    auto t_packed = time_us(
        "packed_gemm",
        [&]() {
            run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                            std::tuple<struct j, struct k, struct l>{}, B);
        },
        3);

    auto alg      = run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                               std::tuple<struct j, struct k, struct l>{}, B);
    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                       std::tuple<struct j, struct k, struct l>{}, B);
        },
        3);

    publish_benchmark_result("generic-Rank-3", "t_generic", N, t_generic);
    publish_benchmark_result("packed-Rank-3", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-Rank-3", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-3 N=256 — large-N tiling benefit (link_dims 256 >> tile_size 64)
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: rank-3 contraction N=256 (MLIR direct)", "[mlir][benchmark][large]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,l]+=A[i,j,k]*B[j,k,l]");
    ProfileAnnotate("N", int64_t(256));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 256;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                              std::tuple<struct j, struct k, struct l>{}, B);
    REQUIRE(ok);

    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                        std::tuple<struct j, struct k, struct l>{}, B);
        },
        3);

    auto t_packed = time_us(
        "packed_gemm",
        [&]() {
            run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                            std::tuple<struct j, struct k, struct l>{}, B);
        },
        3);

    auto alg      = run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                               std::tuple<struct j, struct k, struct l>{}, B);
    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                       std::tuple<struct j, struct k, struct l>{}, B);
        },
        3);

    publish_benchmark_result("generic-Rank-3", "t_generic", N, t_generic);
    publish_benchmark_result("packed-Rank-3", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-Rank-3", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Cache-hit overhead: verify repeated calls reuse the compiled kernel.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: kernel cache hit overhead", "[mlir][benchmark]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,l]+=A[i,j,k]*B[j,k,l]");
    ProfileAnnotate("N", int64_t(32));
    ProfileAnnotate("dtype", "double");
    // Use a rank-3 contraction C[i,l] += A[i,j,k] * B[j,k,l] that MLIR handles
    // (rank-2 GEMM shapes are correctly deferred to BLAS dispatch).
    constexpr size_t           N = 32;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    // First call: compiles the kernel.
    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                              std::tuple<struct j, struct k, struct l>{}, B);
    REQUIRE(ok);

    // Subsequent calls: cache warm — measure per-call overhead.
    auto t_packed = time_us(
        "packed_gemm",
        [&]() {
            run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                            std::tuple<struct j, struct k, struct l>{}, B);
        },
        50);

    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                        std::tuple<struct j, struct k, struct l>{}, B);
        },
        50);

    auto alg      = run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                               std::tuple<struct j, struct k, struct l>{}, B);
    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                       std::tuple<struct j, struct k, struct l>{}, B);
        },
        50);

    publish_benchmark_result("cache-generic", "t_generic", N, t_generic);
    publish_benchmark_result("cache-packed", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("cache-einsum", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ===========================================================================
// NON-GEMM CONTRACTIONS
//
// The following benchmarks test contraction patterns that cannot be mapped
// to a single BLAS GEMM call.  einsum() routes these through the MLIR JIT
// backend (or falls back to the generic OpenMP algorithm).
// ===========================================================================

// ---------------------------------------------------------------------------
// Hadamard: C[i] += A[i,i] * B[i]  (diagonal extraction)
//
// The repeated index 'i' in A prevents the packing path (no valid
// M/N/K decomposition).  Falls through to the per-topology MLIR kernel.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: Hadamard C[i]+=A[i,i]*B[i] N=256", "[mlir][benchmark][nongemm]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i]+=A[i,i]*B[i]");
    ProfileAnnotate("N", int64_t(256));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 256;
    einsums::Tensor<double, 1> C{"C", N};
    einsums::Tensor<double, 2> A{"A", N, N};
    einsums::Tensor<double, 1> B{"B", N};
    einsums::Tensor<double, 1> ref{"ref", N};

    fill(A);
    fill(B);

    // --- Reference ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        ref(ii) += A(ii, ii) * B(ii);

    // --- einsum (should use MLIR generic path) ---
    C.zero();
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(Indices{i}, &C, Indices{i, i}, A, Indices{i}, B, &alg);
    for (size_t ii = 0; ii < N; ++ii)
        REQUIRE(std::abs(C(ii) - ref(ii)) < 1e-10);

    // --- Timing ---
    auto t_generic = time_us("generic", [&]() {
        run_generic(0.0, std::tuple<struct i>{}, C, 1.0, std::tuple<struct i, struct i>{}, A, std::tuple<struct i>{}, B);
    });

    auto t_einsum = time_us("einsum", [&]() {
        run_einsum(0.0, std::tuple<struct i>{}, C, 1.0, std::tuple<struct i, struct i>{}, A, std::tuple<struct i>{}, B);
    });

    publish_benchmark_result("generic-Hadamard C[i]+=A[i,i]*B[i]", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Hadamard C[i]+=A[i,i]*B[i]", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Hadamard rank-3: C[i,j] += A[i,i,j] * B[j]  (repeated index + free dims)
//
// Same Hadamard blocking — repeated 'i' in A means no packing path.
// Larger working set than the rank-1 Hadamard above.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: Hadamard C[i,j]+=A[i,i,j]*B[j] N=64", "[mlir][benchmark][nongemm]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,j]+=A[i,i,j]*B[j]");
    ProfileAnnotate("N", int64_t(64));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 64;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 1> B{"B", N};
    einsums::Tensor<double, 2> ref{"ref", N, N};

    fill(A);
    fill(B);

    // --- Reference ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            ref(ii, jj) += A(ii, ii, jj) * B(jj);

    // --- einsum ---
    C.zero();
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(Indices{i, j}, &C, Indices{i, i, j}, A, Indices{j}, B, &alg);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Timing ---
    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct i, struct j>{}, A,
                        std::tuple<struct j>{}, B);
        },
        5);

    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct i, struct j>{}, A, std::tuple<struct j>{},
                       B);
        },
        5);

    publish_benchmark_result("generic-Hadamard C[i,j]+=A[i,i,j]*B[j]", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Hadamard C[i,j]+=A[i,i,j]*B[j]", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Scalar output: s += A[i,j] * B[i,j]  (Frobenius inner product)
//
// All indices are link (reduction) dims, C is a scalar (rank-0 memref).
// No packing path — scalar output is always per-topology MLIR.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: scalar output s+=A[i,j]*B[i,j] N=128", "[mlir][benchmark][nongemm]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "s+=A[i,j]*B[i,j]");
    ProfileAnnotate("N", int64_t(128));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 128;
    einsums::Tensor<double, 2> A{"A", N, N};
    einsums::Tensor<double, 2> B{"B", N, N};

    fill(A);
    fill(B);

    // --- Reference ---
    double ref = 0.0;
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            ref += A(ii, jj) * B(ii, jj);

    // --- einsum ---
    double                                           result = 0.0;
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(0.0, Indices{}, &result, 1.0, Indices{i, j}, A, Indices{i, j}, B, &alg);
    REQUIRE(std::abs(result - ref) < 1e-6);

    // --- Timing ---
    auto t_generic = time_us("generic", [&]() {
        einsums::tensor_algebra::detail::einsum<true, false, false, false>(0.0, Indices{}, &result, 1.0, Indices{i, j}, A, Indices{i, j},
                                                                           B);
    });

    auto t_einsum = time_us("einsum", [&]() {
        einsums::tensor_algebra::einsum(0.0, Indices{}, &result, 1.0, Indices{i, j}, A, Indices{i, j}, B);
    });

    publish_benchmark_result("generic-Scalar s+=A[i,j]*B[i,j]", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Scalar s+=A[i,j]*B[i,j]", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Scalar output rank-3: s += A[i,j,k] * B[i,j,k]
//
// Higher-rank all-reduction contraction.  Tests MLIR scalar output with
// 3 nested reduction loops.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: scalar output s+=A[i,j,k]*B[i,j,k] N=32", "[mlir][benchmark][nongemm]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "s+=A[i,j,k]*B[i,j,k]");
    ProfileAnnotate("N", int64_t(32));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 32;
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    // --- Reference ---
    double ref = 0.0;
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t kk = 0; kk < N; ++kk)
                ref += A(ii, jj, kk) * B(ii, jj, kk);

    // --- einsum ---
    double                                           result = 0.0;
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(0.0, Indices{}, &result, 1.0, Indices{i, j, k}, A, Indices{i, j, k}, B, &alg);
    REQUIRE(std::abs(result - ref) / std::max(std::abs(ref), 1.0) < 1e-10);

    // --- Timing ---
    auto t_generic = time_us("generic", [&]() {
        einsums::tensor_algebra::detail::einsum<true, false, false, false>(0.0, Indices{}, &result, 1.0, Indices{i, j, k}, A,
                                                                           Indices{i, j, k}, B);
    });

    auto t_einsum = time_us("einsum", [&]() {
        einsums::tensor_algebra::einsum(0.0, Indices{}, &result, 1.0, Indices{i, j, k}, A, Indices{i, j, k}, B);
    });

    publish_benchmark_result("generic-Scalar s+=A[i,j,k]*B[i,j,k]", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Scalar s+=A[i,j,k]*B[i,j,k]", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Asymmetric rank: C[i,j,k] += A[i,l] * B[l,j,k]  (rank-2 x rank-3 -> rank-3)
//
// One M dim (i), two N dims (j,k), one K dim (l).  Multi-dim N means the
// packing path is invalid — falls through to per-topology MLIR.
// ---------------------------------------------------------------------------
// FIXME: This test crashes with SIGABRT during the timing phase (pre-existing).
// The generic OMP algorithm with nested parallelism on rank-3+ contractions
// causes stack overflow / memory corruption on OMP worker threads.
EINSUMS_TEST_CASE("Benchmark: asymmetric C[i,j,k]+=A[i,l]*B[l,j,k] N=32", "[mlir][benchmark][nongemm][!mayfail]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,j,k]+=A[i,l]*B[l,j,k]");
    ProfileAnnotate("N", int64_t(32));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 32;
    einsums::Tensor<double, 3> C{"C", N, N, N};
    einsums::Tensor<double, 2> A{"A", N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};
    einsums::Tensor<double, 3> ref{"ref", N, N, N};

    fill(A);
    fill(B);

    // --- Reference ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t kk = 0; kk < N; ++kk)
                for (size_t ll = 0; ll < N; ++ll)
                    ref(ii, jj, kk) += A(ii, ll) * B(ll, jj, kk);

    // --- einsum ---
    C.zero();
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(Indices{i, j, k}, &C, Indices{i, l}, A, Indices{l, j, k}, B, &alg);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t kk = 0; kk < N; ++kk)
                REQUIRE(std::abs(C(ii, jj, kk) - ref(ii, jj, kk)) < 1e-10);

    // --- Timing ---
    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct j, struct k>{}, C, 1.0, std::tuple<struct i, struct l>{}, A,
                        std::tuple<struct l, struct j, struct k>{}, B);
        },
        5);

    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct j, struct k>{}, C, 1.0, std::tuple<struct i, struct l>{}, A,
                       std::tuple<struct l, struct j, struct k>{}, B);
        },
        5);

    publish_benchmark_result("generic-Asymmetric C[i,j,k]+=A[i,l]*B[l,j,k]", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Asymmetric C[i,j,k]+=A[i,l]*B[l,j,k]", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Batch contraction: C[b,i,j] += A[b,i,k] * B[b,k,j]  (batched GEMM)
//
// Index 'b' is a target index that appears in both A and B — a batch dim.
// The packing path rejects batch dims, so this falls through to MLIR.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: batch C[b,i,j]+=A[b,i,k]*B[b,k,j] B=16 N=32", "[mlir][benchmark][nongemm]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[b,i,j]+=A[b,i,k]*B[b,k,j]");
    ProfileAnnotate("N", int64_t(32));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           Nb = 16; // batch size
    constexpr size_t           N  = 32; // spatial dims
    einsums::Tensor<double, 3> C{"C", Nb, N, N};
    einsums::Tensor<double, 3> A{"A", Nb, N, N};
    einsums::Tensor<double, 3> B{"B", Nb, N, N};
    einsums::Tensor<double, 3> ref{"ref", Nb, N, N};

    fill(A);
    fill(B);

    // --- Reference ---
    ref.zero();
    for (size_t bb = 0; bb < Nb; ++bb)
        for (size_t ii = 0; ii < N; ++ii)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t kk = 0; kk < N; ++kk)
                    ref(bb, ii, jj) += A(bb, ii, kk) * B(bb, kk, jj);

    // --- einsum ---
    C.zero();
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(Indices{b, i, j}, &C, Indices{b, i, k}, A, Indices{b, k, j}, B, &alg);
    for (size_t bb = 0; bb < Nb; ++bb)
        for (size_t ii = 0; ii < N; ++ii)
            for (size_t jj = 0; jj < N; ++jj)
                REQUIRE(std::abs(C(bb, ii, jj) - ref(bb, ii, jj)) < 1e-10);

    // --- Timing ---
    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct b, struct i, struct j>{}, C, 1.0, std::tuple<struct b, struct i, struct k>{}, A,
                        std::tuple<struct b, struct k, struct j>{}, B);
        },
        5);

    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct b, struct i, struct j>{}, C, 1.0, std::tuple<struct b, struct i, struct k>{}, A,
                       std::tuple<struct b, struct k, struct j>{}, B);
        },
        5);

    publish_benchmark_result("generic-Batch C[b,i,j]+=A[b,i,k]*B[b,k,j] B={}", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Batch C[b,i,j]+=A[b,i,k]*B[b,k,j] B={}", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Batch contraction with larger spatial dims: C[b,i,j] += A[b,i,k] * B[b,k,j]
//
// B=4 N=128: each batch slice is a 128x128 GEMM (2*128^3 = 4.2M FLOPs).
// BLAS should be 10-50x faster than the generic algorithm per slice.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: batch C[b,i,j]+=A[b,i,k]*B[b,k,j] B=4 N=128", "[mlir][benchmark][batch]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[b,i,j]+=A[b,i,k]*B[b,k,j]");
    ProfileAnnotate("N", int64_t(128));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           Nb = 4;
    constexpr size_t           N  = 128;
    einsums::Tensor<double, 3> C{"C", Nb, N, N};
    einsums::Tensor<double, 3> A{"A", Nb, N, N};
    einsums::Tensor<double, 3> B{"B", Nb, N, N};

    fill(A);
    fill(B);

    // --- Correctness check (first run only) ---
    {
        einsums::Tensor<double, 3> ref{"ref", Nb, N, N};
        ref.zero();
        for (size_t bb = 0; bb < Nb; ++bb)
            for (size_t ii = 0; ii < N; ++ii)
                for (size_t jj = 0; jj < N; ++jj)
                    for (size_t kk = 0; kk < N; ++kk)
                        ref(bb, ii, jj) += A(bb, ii, kk) * B(bb, kk, jj);

        C.zero();
        einsums::tensor_algebra::einsum(Indices{b, i, j}, &C, Indices{b, i, k}, A, Indices{b, k, j}, B);
        for (size_t bb = 0; bb < Nb; ++bb)
            for (size_t ii = 0; ii < N; ++ii)
                for (size_t jj = 0; jj < N; ++jj)
                    REQUIRE(std::abs(C(bb, ii, jj) - ref(bb, ii, jj)) < 1e-6);
    }

    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct b, struct i, struct j>{}, C, 1.0, std::tuple<struct b, struct i, struct k>{}, A,
                        std::tuple<struct b, struct k, struct j>{}, B);
        },
        5);

    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct b, struct i, struct j>{}, C, 1.0, std::tuple<struct b, struct i, struct k>{}, A,
                       std::tuple<struct b, struct k, struct j>{}, B);
        },
        5);

    publish_benchmark_result("generic-Batch C[b,i,j]+=A[b,i,k]*B[b,k,j] B=4 N={}", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Batch C[b,i,j]+=A[b,i,k]*B[b,k,j] B=4 N={}", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Batch contraction with batch as inner dim: C[i,j,b] += A[i,k,b] * B[k,j,b]
//
// When b is last (innermost for row-major, outermost stride for col-major),
// each batch slice has contiguous M*K / K*N layout — ideal for direct BLAS GEMM.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: batch C[i,j,b]+=A[i,k,b]*B[k,j,b] B=4 N=128", "[mlir][benchmark][batch]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,j,b]+=A[i,k,b]*B[k,j,b]");
    ProfileAnnotate("N", int64_t(128));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           Nb = 4;
    constexpr size_t           N  = 128;
    einsums::Tensor<double, 3> C{"C", N, N, Nb};
    einsums::Tensor<double, 3> A{"A", N, N, Nb};
    einsums::Tensor<double, 3> B{"B", N, N, Nb};

    fill(A);
    fill(B);

    // --- Correctness check ---
    {
        einsums::Tensor<double, 3> ref{"ref", N, N, Nb};
        ref.zero();
        for (size_t ii = 0; ii < N; ++ii)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t bb = 0; bb < Nb; ++bb)
                    for (size_t kk = 0; kk < N; ++kk)
                        ref(ii, jj, bb) += A(ii, kk, bb) * B(kk, jj, bb);

        C.zero();
        einsums::tensor_algebra::einsum(Indices{i, j, b}, &C, Indices{i, k, b}, A, Indices{k, j, b}, B);
        for (size_t ii = 0; ii < N; ++ii)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t bb = 0; bb < Nb; ++bb)
                    REQUIRE(std::abs(C(ii, jj, bb) - ref(ii, jj, bb)) < 1e-6);
    }

    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct j, struct b>{}, C, 1.0, std::tuple<struct i, struct k, struct b>{}, A,
                        std::tuple<struct k, struct j, struct b>{}, B);
        },
        5);

    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct j, struct b>{}, C, 1.0, std::tuple<struct i, struct k, struct b>{}, A,
                       std::tuple<struct k, struct j, struct b>{}, B);
        },
        5);

    publish_benchmark_result("generic-Batch C[i,j,b]+=A[i,k,b]*B[k,j,b] B=4 N={}", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Batch C[i,j,b]+=A[i,k,b]*B[k,j,b] B=4 N={}", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Batch contraction with multi-K: C[b,i,l] += A[b,i,j,k] * B[b,j,k,l]
//
// Batch dim + 2 link dims.  Tests the batch loop wrapping multi-K flatten+GEMM.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: batch+multiK C[b,i,l]+=A[b,i,j,k]*B[b,j,k,l] B=4 N=32", "[mlir][benchmark][batch]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[b,i,l]+=A[b,i,j,k]*B[b,j,k,l]");
    ProfileAnnotate("N", int64_t(32));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           Nb = 4;
    constexpr size_t           N  = 32;
    einsums::Tensor<double, 3> C{"C", Nb, N, N};
    einsums::Tensor<double, 4> A{"A", Nb, N, N, N};
    einsums::Tensor<double, 4> B{"B", Nb, N, N, N};

    fill(A);
    fill(B);

    // --- Correctness check ---
    {
        einsums::Tensor<double, 3> ref{"ref", Nb, N, N};
        ref.zero();
        for (size_t bb = 0; bb < Nb; ++bb)
            for (size_t ii = 0; ii < N; ++ii)
                for (size_t ll = 0; ll < N; ++ll)
                    for (size_t jj = 0; jj < N; ++jj)
                        for (size_t kk = 0; kk < N; ++kk)
                            ref(bb, ii, ll) += A(bb, ii, jj, kk) * B(bb, jj, kk, ll);

        C.zero();
        einsums::tensor_algebra::einsum(Indices{b, i, l}, &C, Indices{b, i, j, k}, A, Indices{b, j, k, l}, B);
        for (size_t bb = 0; bb < Nb; ++bb)
            for (size_t ii = 0; ii < N; ++ii)
                for (size_t ll = 0; ll < N; ++ll)
                    REQUIRE(std::abs(C(bb, ii, ll) - ref(bb, ii, ll)) < 1e-6);
    }

    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct b, struct i, struct l>{}, C, 1.0, std::tuple<struct b, struct i, struct j, struct k>{}, A,
                        std::tuple<struct b, struct j, struct k, struct l>{}, B);
        },
        3);

    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct b, struct i, struct l>{}, C, 1.0, std::tuple<struct b, struct i, struct j, struct k>{}, A,
                       std::tuple<struct b, struct j, struct k, struct l>{}, B);
        },
        3);

    publish_benchmark_result("generic-Batch+multiK C[b,i,l]+=A[b,i,j,k]*B[b,j,k,l] B=4 N={}", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Batch+multiK C[b,i,l]+=A[b,i,j,k]*B[b,j,k,l] B=4 N={}", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Multi-link asymmetric: C[i,k] += A[i,j,k,l] * B[j,l]  (rank-4 x rank-2)
//
// Two link dims (j,l) with different positions in A and B.  The output
// index k appears in both C and A but NOT B — one M dim (i) and one N dim
// (k) in the standard sense, but k is embedded in a rank-4 A.  einsum
// may reshape this as GEMM or fall through to MLIR depending on stride
// contiguity.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: C[i,k]+=A[i,j,k,l]*B[j,l] N=16", "[mlir][benchmark][nongemm]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,k]+=A[i,j,k,l]*B[j,l]");
    ProfileAnnotate("N", int64_t(16));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 16;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 4> A{"A", N, N, N, N};
    einsums::Tensor<double, 2> B{"B", N, N};
    einsums::Tensor<double, 2> ref{"ref", N, N};

    fill(A);
    fill(B);

    // --- Reference ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t kk = 0; kk < N; ++kk)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t ll = 0; ll < N; ++ll)
                    ref(ii, kk) += A(ii, jj, kk, ll) * B(jj, ll);

    // --- einsum ---
    C.zero();
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(Indices{i, k}, &C, Indices{i, j, k, l}, A, Indices{j, l}, B, &alg);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t kk = 0; kk < N; ++kk)
            REQUIRE(std::abs(C(ii, kk) - ref(ii, kk)) < 1e-10);

    // --- Timing ---
    auto t_generic = time_us("generic", [&]() {
        run_generic(0.0, std::tuple<struct i, struct k>{}, C, 1.0, std::tuple<struct i, struct j, struct k, struct l>{}, A,
                    std::tuple<struct j, struct l>{}, B);
    });

    auto t_einsum = time_us("einsum", [&]() {
        run_einsum(0.0, std::tuple<struct i, struct k>{}, C, 1.0, std::tuple<struct i, struct j, struct k, struct l>{}, A,
                   std::tuple<struct j, struct l>{}, B);
    });

    publish_benchmark_result("generic-C[i,k]+=A[i,j,k,l]*B[j,l]", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-C[i,k]+=A[i,j,k,l]*B[j,l]", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Outer product: C[i,j] += A[i] * B[j]  (no link/reduction dims)
//
// Zero contraction dims — this is a pure outer product.  Not a GEMM
// (no summation).  einsum() uses BLAS GER for this, but the MLIR backend
// should also be able to handle it via a 2D parallel linalg.generic with
// no reduction iterator.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: outer product C[i,j]+=A[i]*B[j] N=256", "[mlir][benchmark][nongemm]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,j]+=A[i]*B[j]");
    ProfileAnnotate("N", int64_t(256));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 256;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 1> A{"A", N};
    einsums::Tensor<double, 1> B{"B", N};
    einsums::Tensor<double, 2> ref{"ref", N, N};

    fill(A);
    fill(B);

    // --- Reference ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            ref(ii, jj) += A(ii) * B(jj);

    // --- einsum ---
    C.zero();
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(Indices{i, j}, &C, Indices{i}, A, Indices{j}, B, &alg);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) < 1e-10);

    // --- Timing ---
    auto t_generic = time_us(
        "generic",
        [&]() { run_generic(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i>{}, A, std::tuple<struct j>{}, B); }, 5);

    auto t_einsum = time_us(
        "einsum",
        [&]() { run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i>{}, A, std::tuple<struct j>{}, B); }, 5);

    publish_benchmark_result("generic-Outer product C[i,j]+=A[i]*B[j]", "t_generic", N, t_generic);
    publish_benchmark_result("einsum-Outer product C[i,j]+=A[i]*B[j]", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Tensor dot: C[i,l] += A[i,j,k] * B[k,j,l]  (transposed link dims)
//
// Link dims j,k appear in reversed order in B (k at pos 0, j at pos 1).
// einsum() cannot reshape to GEMM because the link indices are not
// contiguous in the same order in both A and B.  This forces the
// generic/MLIR path.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: tensor dot C[i,l]+=A[i,j,k]*B[k,j,l] N=32", "[mlir][benchmark][nongemm]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,l]+=A[i,j,k]*B[k,j,l]");
    ProfileAnnotate("N", int64_t(32));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 32;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};
    einsums::Tensor<double, 2> ref{"ref", N, N};

    fill(A);
    fill(B);

    // --- Reference ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t kk = 0; kk < N; ++kk)
                    ref(ii, ll) += A(ii, jj, kk) * B(kk, jj, ll);

    // --- einsum ---
    C.zero();
    einsums::tensor_algebra::detail::AlgorithmChoice alg;
    einsums::tensor_algebra::einsum(Indices{i, l}, &C, Indices{i, j, k}, A, Indices{k, j, l}, B, &alg);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            REQUIRE(std::abs(C(ii, ll) - ref(ii, ll)) < 1e-10);

    // --- Timing ---
    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                        std::tuple<struct k, struct j, struct l>{}, B);
        },
        5);

    // Also try the MLIR packing path directly (transposed link dims may still be packable).
    auto t_packed = time_us(
        "packed_gemm",
        [&]() {
            run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                            std::tuple<struct k, struct j, struct l>{}, B);
        },
        5);

    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                       std::tuple<struct k, struct j, struct l>{}, B);
        },
        5);

    publish_benchmark_result("generic-Tensor dot C[i,l]+=A[i,j,k]*B[k,j,l]", "t_generic", N, t_generic);
    publish_benchmark_result("packed-Tensor dot C[i,l]+=A[i,j,k]*B[k,j,l]", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-Tensor dot C[i,l]+=A[i,j,k]*B[k,j,l]", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Large-K contraction: C[i,j] += A[i,k] * B[k,j]  with M=N=64, K=2048
//
// This IS a GEMM shape, but with K >> M,N.  einsum() routes to BLAS GEMM,
// but we call MLIR directly to stress-test the packing tiling path with
// many KC tiles: K/BLIS_KC = 2048/256 = 8 tiles.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: large-K C[i,j]+=A[i,k]*B[k,j] M=N=64 K=2048", "[mlir][benchmark][nongemm][large]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,j]+=A[i,k]*B[k,j]");
    ProfileAnnotate("N", int64_t(64));
    ProfileAnnotate("K", int64_t(2048));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           M = 64;
    constexpr size_t           N = 64;
    constexpr size_t           K = 2048;
    einsums::Tensor<double, 2> C{"C", M, N};
    einsums::Tensor<double, 2> A{"A", M, K};
    einsums::Tensor<double, 2> B{"B", K, N};

    fill(A);
    fill(B);

    // Warmup + compile
    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A,
                              std::tuple<struct k, struct j>{}, B);
    // May or may not be valid for packing (k,j index order in B is standard)
    (void)ok;

    // --- Timing ---
    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A,
                        std::tuple<struct k, struct j>{}, B);
        },
        5);

    auto t_packed = time_us(
        "packed_gemm",
        [&]() {
            run_packed_gemm(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A,
                            std::tuple<struct k, struct j>{}, B);
        },
        5);

    auto alg =
        run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A, std::tuple<struct k, struct j>{}, B);
    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct k>{}, A, std::tuple<struct k, struct j>{},
                       B);
        },
        5);

    publish_benchmark_result("generic-Large-K C[i,j]+=A[i,k]*B[k,j] M={}", "t_generic", N, t_generic);
    publish_benchmark_result("packed-Large-K C[i,j]+=A[i,k]*B[k,j] M={}", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-Large-K C[i,j]+=A[i,k]*B[k,j] M={}", "t_einsum", N, t_einsum);
    REQUIRE(t_einsum > 0.0);
}

// ---------------------------------------------------------------------------
// Rank-4 contraction with 3 link dims at larger N:
// C[i,j] += A[i,l,k,m] * B[m,l,j,k]  (same as existing but N=16)
//
// More FLOPs than the N=8 version (16^5 = 1M vs 8^5 = 32K) to give a
// better picture of MLIR packing overhead at meaningful sizes.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k] N=16", "[mlir][benchmark][nongemm]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,j]+=A[i,l,k,m]*B[m,l,j,k]");
    ProfileAnnotate("N", int64_t(16));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 16;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 4> A{"A", N, N, N, N};
    einsums::Tensor<double, 4> B{"B", N, N, N, N};
    einsums::Tensor<double, 2> ref{"ref", N, N};

    fill(A);
    fill(B);

    // --- Reference ---
    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t ll = 0; ll < N; ++ll)
                for (size_t kk = 0; kk < N; ++kk)
                    for (size_t mm = 0; mm < N; ++mm)
                        ref(ii, jj) += A(ii, ll, kk, mm) * B(mm, ll, jj, kk);

    // --- MLIR ---
    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct l, struct k, struct m>{}, A,
                              std::tuple<struct m, struct l, struct j, struct k>{}, B);
    REQUIRE(ok);
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            REQUIRE(std::abs(C(ii, jj) - ref(ii, jj)) / std::max(std::abs(ref(ii, jj)), 1.0) < 1e-8);

    // --- Timing ---
    auto t_generic = time_us(
        "generic",
        [&]() {
            run_generic(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct l, struct k, struct m>{}, A,
                        std::tuple<struct m, struct l, struct j, struct k>{}, B);
        },
        5);

    auto t_packed = time_us(
        "packed_gemm",
        [&]() {
            run_packed_gemm(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct l, struct k, struct m>{}, A,
                            std::tuple<struct m, struct l, struct j, struct k>{}, B);
        },
        5);

    auto alg      = run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct l, struct k, struct m>{}, A,
                               std::tuple<struct m, struct l, struct j, struct k>{}, B);
    auto t_einsum = time_us(
        "einsum",
        [&]() {
            run_einsum(0.0, std::tuple<struct i, struct j>{}, C, 1.0, std::tuple<struct i, struct l, struct k, struct m>{}, A,
                       std::tuple<struct m, struct l, struct j, struct k>{}, B);
        },
        5);

    publish_benchmark_result("generic-Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k]", "t_generic", N, t_generic);
    publish_benchmark_result("packed-Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k]", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-Rank-4 C[i,j]+=A[i,l,k,m]*B[m,l,j,k]", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ===========================================================================
// ZERO-COPY LAYOUT BENCHMARKS
//
// Compare flatten+GEMM performance across four zero-copy combinations.
// All use rank-3 contractions with N=64 (K=4096, 16 KC tiles).
// ===========================================================================

// ---------------------------------------------------------------------------
// Both A and B zero-copy: C[i,l] += A[i,j,k] * B[l,j,k]
// No data copying at all — single GEMM call on raw tensor pointers.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: zero-copy both N=64", "[mlir][benchmark][zerocopy]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,l]+=A[i,j,k]*B[l,j,k]");
    ProfileAnnotate("N", int64_t(64));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 64;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                              std::tuple<struct l, struct j, struct k>{}, B);
    REQUIRE(ok);

    auto t_packed = time_us("packed_gemm", [&]() {
        run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                        std::tuple<struct l, struct j, struct k>{}, B);
    });

    auto t_einsum = time_us("einsum", [&]() {
        run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                   std::tuple<struct l, struct j, struct k>{}, B);
    });

    publish_benchmark_result("packed-ZC-both C[i,l]+=A[i,j,k]*B[l,j,k]", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-ZC-both C[i,l]+=A[i,j,k]*B[l,j,k]", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Only A zero-copy: C[i,l] += A[i,j,k] * B[j,k,l]  (the original benchmark)
// A passed directly, B copied per KC tile.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: zero-copy A-only N=64", "[mlir][benchmark][zerocopy]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,l]+=A[i,j,k]*B[j,k,l]");
    ProfileAnnotate("N", int64_t(64));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 64;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                              std::tuple<struct j, struct k, struct l>{}, B);
    REQUIRE(ok);

    auto t_packed = time_us("packed_gemm", [&]() {
        run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                        std::tuple<struct j, struct k, struct l>{}, B);
    });

    auto t_einsum = time_us("einsum", [&]() {
        run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct i, struct j, struct k>{}, A,
                   std::tuple<struct j, struct k, struct l>{}, B);
    });

    publish_benchmark_result("packed-ZC-A C[i,l]+=A[i,j,k]*B[j,k,l]", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-ZC-A C[i,l]+=A[i,j,k]*B[j,k,l]", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Only B zero-copy: C[i,l] += A[j,i,k] * B[l,j,k]
// B passed directly, A copied per KC tile.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: zero-copy B-only N=64", "[mlir][benchmark][zerocopy]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,l]+=A[j,i,k]*B[l,j,k]");
    ProfileAnnotate("N", int64_t(64));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 64;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct j, struct i, struct k>{}, A,
                              std::tuple<struct l, struct j, struct k>{}, B);
    REQUIRE(ok);

    auto t_packed = time_us("packed_gemm", [&]() {
        run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct j, struct i, struct k>{}, A,
                        std::tuple<struct l, struct j, struct k>{}, B);
    });

    auto t_einsum = time_us("einsum", [&]() {
        run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct j, struct i, struct k>{}, A,
                   std::tuple<struct l, struct j, struct k>{}, B);
    });

    publish_benchmark_result("packed-ZC-B C[i,l]+=A[j,i,k]*B[l,j,k]", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-ZC-B C[i,l]+=A[j,i,k]*B[l,j,k]", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}

// ---------------------------------------------------------------------------
// Neither zero-copy: C[i,l] += A[j,i,k] * B[j,k,l]
// Both A and B copied per KC tile.
// ---------------------------------------------------------------------------
EINSUMS_TEST_CASE("Benchmark: zero-copy neither N=64", "[mlir][benchmark][zerocopy]") {
    LabeledSection0();
    ProfileAnnotate("pattern", "C[i,l]+=A[j,i,k]*B[j,k,l]");
    ProfileAnnotate("N", int64_t(64));
    ProfileAnnotate("dtype", "double");
    constexpr size_t           N = 64;
    einsums::Tensor<double, 2> C{"C", N, N};
    einsums::Tensor<double, 3> A{"A", N, N, N};
    einsums::Tensor<double, 3> B{"B", N, N, N};

    fill(A);
    fill(B);

    C.zero();
    bool ok = run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct j, struct i, struct k>{}, A,
                              std::tuple<struct j, struct k, struct l>{}, B);
    REQUIRE(ok);

    auto t_packed = time_us("packed_gemm", [&]() {
        run_packed_gemm(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct j, struct i, struct k>{}, A,
                        std::tuple<struct j, struct k, struct l>{}, B);
    });

    auto t_einsum = time_us("einsum", [&]() {
        run_einsum(0.0, std::tuple<struct i, struct l>{}, C, 1.0, std::tuple<struct j, struct i, struct k>{}, A,
                   std::tuple<struct j, struct k, struct l>{}, B);
    });

    publish_benchmark_result("packed-ZC-none C[i,l]+=A[j,i,k]*B[j,k,l]", "t_packed_gemm", N, t_packed);
    publish_benchmark_result("einsum-ZC-none C[i,l]+=A[j,i,k]*B[j,k,l]", "t_einsum", N, t_einsum);
    REQUIRE(t_packed.avg > 0.0);
}
