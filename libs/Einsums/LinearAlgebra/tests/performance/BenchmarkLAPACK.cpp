//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Performance benchmarks for LAPACK operations.

#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::performance;

namespace {

// -----------------------------------------------------------------------
// Symmetric Eigendecomposition (syev)
// -----------------------------------------------------------------------

void bench_syev(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N);
    fill_spd(A);

    ProfileAnnotate("operation", "syev");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("description", "eigendecomposition");
    auto t_blas = time_us("blas-syev", [&] {
        Tensor<double, 2>   work_A = A;
        std::vector<double> w(N);
        std::vector<double> work(static_cast<size_t>(3) * N);
        blas::syev('V', 'U', N, work_A.data(), N, w.data(), work.data(), 3 * N);
    });
    publish_benchmark_result("blas-syev", "t_blas", N, t_blas);

    auto t_la = time_us("la-syev", [&] { auto [evecs, evals] = linear_algebra::syev(A); });
    publish_benchmark_result("la-syev", "t_la", N, t_la);
}

// -----------------------------------------------------------------------
// LU Factorization (getrf)
// -----------------------------------------------------------------------

void bench_getrf(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N);
    fill_spd(A);

    ProfileAnnotate("operation", "getrf");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("description", "LU factorization");
    auto t_blas = time_us("blas-getrf", [&] {
        Tensor<double, 2>        work_A = A;
        std::vector<blas::int_t> ipiv(N);
        blas::getrf(N, N, work_A.data(), N, ipiv.data());
    });
    publish_benchmark_result("blas-getrf", "t_blas", N, t_blas);

    auto t_la = time_us("la-getrf", [&] {
        Tensor<double, 2>         work_A = A;
        BufferVector<blas::int_t> pivs;
        (void)linear_algebra::getrf(&work_A, &pivs);
    });
    publish_benchmark_result("la-getrf", "t_la", N, t_la);
}

// -----------------------------------------------------------------------
// SVD (gesvd)
// -----------------------------------------------------------------------

void bench_gesvd(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N);
    fill(A);

    ProfileAnnotate("operation", "gesvd");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("description", "SVD");
    auto t_blas = time_us("blas-gesvd", [&] {
        Tensor<double, 2>   work_A = A;
        auto                NN     = static_cast<size_t>(N);
        std::vector<double> s(NN), u(NN * NN), vt(NN * NN), superb(NN);
        blas::gesvd('A', 'A', N, N, work_A.data(), N, s.data(), u.data(), N, vt.data(), N, superb.data());
    });
    publish_benchmark_result("blas-gesvd", "t_blas", N, t_blas);

    auto t_la = time_us("la-svd", [&] { auto [U, S, Vt] = linear_algebra::svd(A); });
    publish_benchmark_result("la-svd", "t_la", N, t_la);
}

// -----------------------------------------------------------------------
// QR Factorization (geqrf)
// -----------------------------------------------------------------------

void bench_geqrf(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N);
    fill(A);

    ProfileAnnotate("operation", "geqrf");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("description", "QR factorization");
    auto t_blas = time_us("blas-geqrf", [&] {
        Tensor<double, 2>   work_A = A;
        std::vector<double> tau(N);
        blas::geqrf(N, N, work_A.data(), N, tau.data());
    });
    publish_benchmark_result("blas-geqrf", "t_blas", N, t_blas);

    auto t_la = time_us("la-qr", [&] { auto [Q, R] = linear_algebra::qr(A); });
    publish_benchmark_result("la-qr", "t_la", N, t_la);
}

} // namespace

// -----------------------------------------------------------------------
// Test cases
// -----------------------------------------------------------------------

EINSUMS_TEST_CASE("LAPACK Eigendecomposition", "[performance][lapack]") {
    progress_init(6); // 3 sizes × 2 levels
    for (int N : {64, 256, 1024}) {
        bench_syev(N);
    }
}

EINSUMS_TEST_CASE("LAPACK LU Factorization", "[performance][lapack]") {
    progress_init(8);
    for (int N : {64, 256, 1024, 2048}) {
        bench_getrf(N);
    }
}

EINSUMS_TEST_CASE("LAPACK SVD", "[performance][lapack]") {
    progress_init(6); // 3 sizes × 2 levels — SVD at N=2048 exceeds timeout
    for (int N : {64, 256, 512}) {
        bench_gesvd(N);
    }
}

EINSUMS_TEST_CASE("LAPACK QR", "[performance][lapack]") {
    progress_init(8);
    for (int N : {64, 256, 1024, 2048}) {
        bench_geqrf(N);
    }
}
