//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Performance benchmarks for BLAS operations.
//
// Benchmarks both raw blas:: calls and linear_algebra:: tensor wrappers
// to measure BLAS vendor performance and Einsums overhead.

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <tuple>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::performance;

namespace {

// -----------------------------------------------------------------------
// BLAS Level 1
// -----------------------------------------------------------------------

void bench_dot(int N) {
    LabeledSection0();
    Tensor<double, 1> x("x", N), y("y", N);
    fill(x);
    fill(y);

    ProfileAnnotate("level", "L1");
    ProfileAnnotate("operation", "dot");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("elements", int64_t(N));
    auto t_blas = time_us("blas-dot", [&] { blas::dot(N, x.data(), 1, y.data(), 1); });
    publish_benchmark_result("blas-dot", "t_blas", N, t_blas);

    auto t_la = time_us("la-dot", [&] { std::ignore = linear_algebra::dot(x, y); });
    publish_benchmark_result("la-dot", "t_la", N, t_la);
}

void bench_axpy(int N) {
    LabeledSection0();
    Tensor<double, 1> x("x", N), y("y", N), y_backup("y_backup", N);
    fill(x);
    fill(y);
    y_backup = y;

    ProfileAnnotate("level", "L1");
    ProfileAnnotate("operation", "axpy");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("elements", int64_t(N));
    auto t_blas = time_us("blas-axpy", [&] {
        y = y_backup;
        blas::axpy(N, 2.0, x.data(), 1, y.data(), 1);
    });
    publish_benchmark_result("blas-axpy", "t_blas", N, t_blas);

    auto t_la = time_us("la-axpy", [&] {
        y = y_backup;
        linear_algebra::axpy(2.0, x, &y);
    });
    publish_benchmark_result("la-axpy", "t_la", N, t_la);
}

void bench_scal(int N) {
    LabeledSection0();
    Tensor<double, 1> x("x", N), x_backup("x_backup", N);
    fill(x);
    x_backup = x;

    ProfileAnnotate("level", "L1");
    ProfileAnnotate("operation", "scal");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("elements", int64_t(N));
    auto t_blas = time_us("blas-scal", [&] {
        x = x_backup;
        blas::scal(N, 2.0, x.data(), 1);
    });
    publish_benchmark_result("blas-scal", "t_blas", N, t_blas);

    auto t_la = time_us("la-scal", [&] {
        x = x_backup;
        linear_algebra::scale(2.0, &x);
    });
    publish_benchmark_result("la-scal", "t_la", N, t_la);
}

// -----------------------------------------------------------------------
// BLAS Level 2
// -----------------------------------------------------------------------

void bench_gemv(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N);
    Tensor<double, 1> x("x", N), y("y", N), y_backup("y_backup", N);
    fill(A);
    fill(x);
    fill(y);
    y_backup = y;

    ProfileAnnotate("level", "L2");
    ProfileAnnotate("operation", "gemv");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("elements", int64_t(N) * int64_t(N));
    auto t_blas = time_us("blas-gemv", [&] {
        y = y_backup;
        blas::gemv('n', N, N, 1.0, A.data(), N, x.data(), 1, 0.0, y.data(), 1);
    });
    publish_benchmark_result("blas-gemv", "t_blas", N, t_blas);

    auto t_la = time_us("la-gemv", [&] {
        y = y_backup;
        linear_algebra::gemv<false>(1.0, A, x, 0.0, &y);
    });
    publish_benchmark_result("la-gemv", "t_la", N, t_la);
}

void bench_ger(int N) {
    LabeledSection0();
    Tensor<double, 1> x("x", N), y("y", N);
    Tensor<double, 2> A("A", N, N), A_backup("A_backup", N, N);
    fill(x);
    fill(y);
    A.zero();
    A_backup = A;

    ProfileAnnotate("level", "L2");
    ProfileAnnotate("operation", "ger");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("elements", int64_t(N) * int64_t(N));
    auto t_blas = time_us("blas-ger", [&] {
        A = A_backup;
        blas::ger(N, N, 1.0, x.data(), 1, y.data(), 1, A.data(), N);
    });
    publish_benchmark_result("blas-ger", "t_blas", N, t_blas);

    auto t_la = time_us("la-ger", [&] {
        A = A_backup;
        linear_algebra::ger(1.0, x, y, &A);
    });
    publish_benchmark_result("la-ger", "t_la", N, t_la);
}

// -----------------------------------------------------------------------
// BLAS Level 3
// -----------------------------------------------------------------------

void bench_gemm(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N), B("B", N, N), C("C", N, N);
    fill(A);
    fill(B);
    C.zero();

    ProfileAnnotate("level", "L3");
    ProfileAnnotate("operation", "gemm");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("elements", int64_t(N) * int64_t(N));
    auto t_blas = time_us("blas-gemm", [&] { blas::gemm('n', 'n', N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C.data(), N); });
    publish_benchmark_result("blas-gemm", "t_blas", N, t_blas);

    auto t_la = time_us("la-gemm", [&] { linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C); });
    publish_benchmark_result("la-gemm", "t_la", N, t_la);
}

} // namespace

// -----------------------------------------------------------------------
// Test cases
// -----------------------------------------------------------------------

EINSUMS_TEST_CASE("BLAS Level 1", "[performance][blas]") {
    // 6 sizes × 3 ops × 2 levels = 36
    progress_init(36);
    for (int N : {64, 256, 1024, 4096, 16384, 65536}) {
        bench_dot(N);
        bench_axpy(N);
        bench_scal(N);
    }
}

EINSUMS_TEST_CASE("BLAS Level 2", "[performance][blas]") {
    // 5 sizes × 2 ops × 2 levels = 20
    progress_init(20);
    for (int N : {64, 256, 1024, 4096, 8192}) {
        bench_gemv(N);
        bench_ger(N);
    }
}

EINSUMS_TEST_CASE("BLAS Level 3", "[performance][blas]") {
    // 5 sizes × 1 op × 2 levels = 10
    progress_init(10);
    for (int N : {64, 256, 1024, 4096, 8192}) {
        bench_gemm(N);
    }
}
