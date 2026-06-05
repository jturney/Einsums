//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Performance benchmark: thread scaling.
//
// Runs GEMM at various thread counts (1, 2, 4, max) to detect parallelism
// regressions. Reports timing and parallel efficiency.

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <Einsums/Profile/Profile.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::performance;

namespace {

/// Run GEMM at a specific thread count and report.
void bench_gemm_threads(int N, int nthreads) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N), B("B", N, N), C("C", N, N);
    fill(A);
    fill(B);
    C.zero();

#ifdef _OPENMP
    int old_threads = omp_get_max_threads();
    omp_set_num_threads(nthreads);
#endif

    char label[64];
    std::snprintf(label, sizeof(label), "gemm-%dt", nthreads);

    ProfileAnnotate("operation", std::string("gemm"));
    ProfileAnnotate("dtype", std::string("double"));
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("threads", int64_t(nthreads));
    auto s = time_us(label, [&] { blas::gemm('n', 'n', N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C.data(), N); });
    publish_benchmark_result(label, "t_blas", N, s);

#ifdef _OPENMP
    omp_set_num_threads(old_threads);
#endif
}

/// Run the LA gemm wrapper at a specific thread count.
void bench_la_gemm_threads(int N, int nthreads) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N), B("B", N, N), C("C", N, N);
    fill(A);
    fill(B);
    C.zero();

#ifdef _OPENMP
    int old_threads = omp_get_max_threads();
    omp_set_num_threads(nthreads);
#endif

    char label[64];
    std::snprintf(label, sizeof(label), "la-gemm-%dt", nthreads);

    ProfileAnnotate("operation", std::string("la-gemm"));
    ProfileAnnotate("dtype", std::string("double"));
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("threads", int64_t(nthreads));
    auto s = time_us(label, [&] { linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C); });
    publish_benchmark_result(label, "t_la", N, s);

#ifdef _OPENMP
    omp_set_num_threads(old_threads);
#endif
}

} // namespace

// -----------------------------------------------------------------------
// Test cases
// -----------------------------------------------------------------------

EINSUMS_TEST_CASE("Thread Scaling GEMM N=1024", "[performance][scaling]") {
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif

    std::vector<int> thread_counts;
    for (int t = 1; t <= max_threads; t *= 2) {
        thread_counts.push_back(t);
    }
    if (thread_counts.back() != max_threads) {
        thread_counts.push_back(max_threads);
    }

    progress_init(static_cast<int>(thread_counts.size()) * 2);

    for (int t : thread_counts) {
        bench_gemm_threads(1024, t);
        bench_la_gemm_threads(1024, t);
    }
}

EINSUMS_TEST_CASE("Thread Scaling GEMM N=4096", "[performance][scaling]") {
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif

    std::vector<int> thread_counts;
    for (int t = 1; t <= max_threads; t *= 2) {
        thread_counts.push_back(t);
    }
    if (thread_counts.back() != max_threads) {
        thread_counts.push_back(max_threads);
    }

    progress_init(static_cast<int>(thread_counts.size()) * 2);

    for (int t : thread_counts) {
        bench_gemm_threads(4096, t);
        bench_la_gemm_threads(4096, t);
    }
}
