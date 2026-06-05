//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Performance benchmarks for einsum dispatch path.

#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
using namespace einsums::performance;

namespace {

// -----------------------------------------------------------------------
// Rank-2 contraction (GEMM equivalent): C(i,j) = A(i,k) * B(k,j)
// -----------------------------------------------------------------------

void bench_rank2_gemm(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N), B("B", N, N), C("C", N, N);
    fill(A);
    fill(B);
    C.zero();

    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("pattern", "ij=ik,kj");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("description", "GEMM-equivalent");
    auto t = time_us("einsum-gemm", [&] { tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k}, A, Indices{k, j}, B); });
    publish_benchmark_result("einsum-gemm", "t_einsum", N, t);
}

// -----------------------------------------------------------------------
// Rank-2 with transpose: C(i,j) = A(k,i) * B(k,j)
// -----------------------------------------------------------------------

void bench_rank2_transpose(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N), B("B", N, N), C("C", N, N);
    fill(A);
    fill(B);
    C.zero();

    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("pattern", "ij=ki,kj");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("description", "transposed GEMM");
    auto t = time_us("einsum-gemm-t", [&] { tensor_algebra::einsum(0.0, Indices{i, j}, &C, 1.0, Indices{k, i}, A, Indices{k, j}, B); });
    publish_benchmark_result("einsum-gemm-t", "t_einsum", N, t);
}

// -----------------------------------------------------------------------
// Rank-3 contraction: C(i,j,l) = A(i,j,k) * B(l,k)
// -----------------------------------------------------------------------

void bench_rank3(int N) {
    LabeledSection0();
    Tensor<double, 3> A("A", N, N, N), C("C", N, N, N);
    Tensor<double, 2> B("B", N, N);
    fill(A);
    fill(B);
    C.zero();

    ProfileAnnotate("rank", int64_t(3));
    ProfileAnnotate("pattern", "ijl=ijk,lk");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("elements", int64_t(N) * int64_t(N) * int64_t(N));
    auto t = time_us("einsum-rank3", [&] {
        tensor_algebra::einsum(0.0, Indices{i, j, l}, &C, 1.0, Indices{i, j, k}, A, Indices{l, k}, B);
    });
    publish_benchmark_result("einsum-rank3", "t_einsum", N, t);
}

// -----------------------------------------------------------------------
// Hadamard product: C(i) += A(i,j) * B(j,i)
// -----------------------------------------------------------------------

void bench_hadamard(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N), B("B", N, N);
    Tensor<double, 1> C("C", N);
    fill(A);
    fill(B);
    C.zero();

    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("pattern", "i=ij,ji");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("description", "Hadamard");
    auto t = time_us("einsum-hadamard", [&] { tensor_algebra::einsum(0.0, Indices{i}, &C, 1.0, Indices{i, j}, A, Indices{j, i}, B); });
    publish_benchmark_result("einsum-hadamard", "t_einsum", N, t);
}

// -----------------------------------------------------------------------
// Trace contraction: scalar = A(i,j) * B(j,i)
// -----------------------------------------------------------------------

void bench_trace(int N) {
    LabeledSection0();
    Tensor<double, 2> A("A", N, N), B("B", N, N);
    fill(A);
    fill(B);
    double result = 0.0;

    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("pattern", "=ij,ji");
    ProfileAnnotate("dtype", "double");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("description", "trace");
    auto t = time_us("einsum-trace", [&] { tensor_algebra::einsum(0.0, Indices{}, &result, 1.0, Indices{i, j}, A, Indices{j, i}, B); });
    publish_benchmark_result("einsum-trace", "t_einsum", N, t);
}

} // namespace

// -----------------------------------------------------------------------
// Test cases
// -----------------------------------------------------------------------

EINSUMS_TEST_CASE("Einsum GEMM-equivalent", "[performance][einsum]") {
    progress_init(12); // 6 sizes × 2 ops
    for (int N : {8, 32, 64, 128, 256, 512}) {
        bench_rank2_gemm(N);
        bench_rank2_transpose(N);
    }
}

EINSUMS_TEST_CASE("Einsum Rank-3", "[performance][einsum]") {
    progress_init(4); // 4 sizes × 1 op
    for (int N : {8, 32, 64, 128}) {
        bench_rank3(N);
    }
}

EINSUMS_TEST_CASE("Einsum Hadamard and Trace", "[performance][einsum]") {
    progress_init(12); // 6 sizes × 2 ops
    for (int N : {8, 32, 64, 128, 256, 512}) {
        bench_hadamard(N);
        bench_trace(N);
    }
}
