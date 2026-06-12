//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase D.1: runtime-tensor coverage for BLAS-1 ops + invert + syev/heev
// void forms. These take simple inputs and either return scalars or modify
// in place; one file batches them since each test is short.

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <complex>
#include <cstring>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

template <typename T>
void fill_runtime(RuntimeTensor<T> &t, std::vector<T> const &flat) {
    REQUIRE(t.size() == flat.size());
    std::memcpy(t.data(), flat.data(), flat.size() * sizeof(T));
}

template <typename T>
auto runtime_data(RuntimeTensor<T> const &t) -> std::vector<T> {
    std::vector<T> out(t.size());
    std::memcpy(out.data(), t.data(), t.size() * sizeof(T));
    return out;
}

} // namespace

// ──────────────────────────────────────────────────────────────────────────
// axpy: Y := alpha * X + Y
// ──────────────────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("RuntimeTensor axpy — accumulates onto Y", "[linear-algebra][runtime]", float, double) {
    RuntimeTensor<TestType> X("X", {4});
    RuntimeTensor<TestType> Y("Y", {4});
    fill_runtime(X, {TestType(1), TestType(2), TestType(3), TestType(4)});
    fill_runtime(Y, {TestType(10), TestType(20), TestType(30), TestType(40)});

    linear_algebra::axpy(TestType{2}, X, &Y); // Y += 2*X
    CHECK_THAT(runtime_data(Y), Catch::Matchers::Equals(std::vector<TestType>{TestType(12), TestType(24), TestType(36), TestType(48)}));
}

// ──────────────────────────────────────────────────────────────────────────
// axpby: Y := alpha * X + beta * Y
// ──────────────────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("RuntimeTensor axpby — full BLAS-1 update", "[linear-algebra][runtime]", float, double) {
    RuntimeTensor<TestType> X("X", {3});
    RuntimeTensor<TestType> Y("Y", {3});
    fill_runtime(X, {TestType(1), TestType(2), TestType(3)});
    fill_runtime(Y, {TestType(100), TestType(200), TestType(300)});

    // Y = 3*X + 0.5*Y
    linear_algebra::axpby(TestType{3}, X, TestType{0.5}, &Y);
    CHECK_THAT(runtime_data(Y), Catch::Matchers::Equals(std::vector<TestType>{TestType(53), TestType(106), TestType(159)}));
}

// ──────────────────────────────────────────────────────────────────────────
// dot: scalar = X · Y
// ──────────────────────────────────────────────────────────────────────────

TEMPLATE_TEST_CASE("RuntimeTensor dot — scalar matches manual sum", "[linear-algebra][runtime]", float, double) {
    RuntimeTensor<TestType> X("X", {4});
    RuntimeTensor<TestType> Y("Y", {4});
    fill_runtime(X, {TestType(1), TestType(2), TestType(3), TestType(4)});
    fill_runtime(Y, {TestType(5), TestType(6), TestType(7), TestType(8)});

    auto const result = linear_algebra::dot(X, Y);
    CHECK(result == TestType(70)); // 1*5 + 2*6 + 3*7 + 4*8
}

// ──────────────────────────────────────────────────────────────────────────
// Complex dtypes: exercises the dispatcher path on the BLAS-1 runtime ops
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("RuntimeTensor axpy — complex<double>", "[linear-algebra][runtime]") {
    using C = std::complex<double>;
    RuntimeTensor<C> X("X", {3});
    RuntimeTensor<C> Y("Y", {3});
    fill_runtime(X, {C{1, 0}, C{0, 1}, C{2, 2}});
    fill_runtime(Y, {C{10, 0}, C{20, 0}, C{30, 0}});

    linear_algebra::axpy(C{2, 0}, X, &Y); // Y += 2*X
    CHECK(Y(0) == C{12, 0});
    CHECK(Y(1) == C{20, 2});
    CHECK(Y(2) == C{34, 4});
}

TEST_CASE("RuntimeTensor dot — complex<double> conjugate-aware", "[linear-algebra][runtime]") {
    // linear_algebra::dot computes X^T * Y (not the Hermitian-conjugating
    // version; that's true_dot). Verify with hand-computed values.
    using C = std::complex<double>;
    RuntimeTensor<C> X("X", {3});
    RuntimeTensor<C> Y("Y", {3});
    fill_runtime(X, {C{1, 1}, C{2, 0}, C{0, 1}});
    fill_runtime(Y, {C{1, 0}, C{0, 1}, C{2, 2}});

    auto const result = linear_algebra::dot(X, Y);
    // (1+i)(1) + (2)(i) + (i)(2+2i) = (1+i) + (2i) + (-2 + 2i) = -1 + 5i
    CHECK_THAT(result.real(), Catch::Matchers::WithinAbs(-1.0, 1e-12));
    CHECK_THAT(result.imag(), Catch::Matchers::WithinAbs(5.0, 1e-12));
}

// ──────────────────────────────────────────────────────────────────────────
// RuntimeTensorView inputs: non-owning views must dispatch through the
// runtime overloads exactly like owned RuntimeTensors do.
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("RuntimeTensor gemm — RuntimeTensorView inputs work", "[linear-algebra][runtime][view]") {
    // Owned tensors with an extra row to exercise the view's stride.
    RuntimeTensor<double> A_owned("A", {3, 3});
    RuntimeTensor<double> B_owned("B", {3, 3});
    RuntimeTensor<double> C_owned("C", {3, 3});
    fill_runtime(A_owned, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    fill_runtime(B_owned, {1, 0, 0, 0, 1, 0, 0, 0, 1}); // identity
    fill_runtime(C_owned, std::vector<double>(9, 0.0));

    // Non-owning view over the entire owned A. Dispatch should still go
    // through the dynamic-rank gemm overload.
    RuntimeTensorView<double> Aview(A_owned.impl());
    RuntimeTensorView<double> Bview(B_owned.impl());

    linear_algebra::gemm<false, false>(1.0, Aview, Bview, 0.0, &C_owned);

    // A * I == A.
    CHECK_THAT(runtime_data(C_owned), Catch::Matchers::Equals(runtime_data(A_owned)));
}

TEST_CASE("RuntimeTensor axpy — view + view", "[linear-algebra][runtime][view]") {
    RuntimeTensor<double> X_owned("X", {4});
    RuntimeTensor<double> Y_owned("Y", {4});
    fill_runtime(X_owned, {1.0, 2.0, 3.0, 4.0});
    fill_runtime(Y_owned, {10.0, 20.0, 30.0, 40.0});

    RuntimeTensorView<double> Xview(X_owned.impl());

    linear_algebra::axpy(2.0, Xview, &Y_owned);
    CHECK_THAT(runtime_data(Y_owned), Catch::Matchers::Equals(std::vector<double>{12.0, 24.0, 36.0, 48.0}));
}

// ──────────────────────────────────────────────────────────────────────────
// invert: A := A^{-1}  (in place)
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("RuntimeTensor invert — A * A^{-1} == identity", "[linear-algebra][runtime]") {
    constexpr int N       = 4;
    auto          Astatic = create_random_tensor<double>("A", N, N);

    RuntimeTensor<double> A("A", {N, N}, /*row_major=*/false);
    std::memcpy(A.data(), Astatic.data(), A.size() * sizeof(double));

    RuntimeTensor<double> Aoriginal = A;

    linear_algebra::invert(&A);

    // A_original @ A == I
    RuntimeTensor<double> I("I", {N, N}, /*row_major=*/false);
    linear_algebra::gemm<false, false>(1.0, Aoriginal, A, 0.0, &I);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double const expected = (i == j) ? 1.0 : 0.0;
            CHECK_THAT(I(i, j), Catch::Matchers::WithinAbs(expected, 1e-9));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// syev (void form): A := eigenvectors, W := eigenvalues
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("RuntimeTensor syev (void) — eigenvalues match static path", "[linear-algebra][runtime]") {
    constexpr int N       = 4;
    auto          Astatic = create_random_tensor<double>("A", N, N);
    // Symmetrize.
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double const m = 0.5 * (Astatic(i, j) + Astatic(j, i));
            Astatic(i, j)  = m;
            Astatic(j, i)  = m;
        }
    }

    RuntimeTensor<double> A("A", {N, N}, /*row_major=*/false);
    RuntimeTensor<double> W("W", {N}, /*row_major=*/false);
    std::memcpy(A.data(), Astatic.data(), A.size() * sizeof(double));

    linear_algebra::syev(&A, &W);

    auto              Astatic_ref = Astatic;
    Tensor<double, 1> Wref{"Wref", N};
    linear_algebra::syev(&Astatic_ref, &Wref);

    for (int i = 0; i < N; ++i) {
        CHECK_THAT(W(i), Catch::Matchers::WithinAbs(Wref(i), 1e-9));
    }
}

// ──────────────────────────────────────────────────────────────────────────
// heev (void form): complex Hermitian eigendecomp
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("RuntimeTensor heev — complex Hermitian eigenvalues are real", "[linear-algebra][runtime]") {
    using C               = std::complex<double>;
    constexpr int N       = 4;
    auto          Astatic = create_random_tensor<C>("A", N, N);
    // Hermitize: A = (A + A^H) / 2
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            C const m     = C{0.5, 0.0} * (Astatic(i, j) + std::conj(Astatic(j, i)));
            Astatic(i, j) = m;
            if (i != j) {
                Astatic(j, i) = std::conj(m);
            } else {
                Astatic(i, j) = C{m.real(), 0.0};
            }
        }
    }

    RuntimeTensor<C>      A("A", {N, N}, /*row_major=*/false);
    RuntimeTensor<double> W("W", {N}, /*row_major=*/false);
    std::memcpy(A.data(), Astatic.data(), A.size() * sizeof(C));

    linear_algebra::heev(&A, &W);

    // All eigenvalues should be finite and real.
    for (int i = 0; i < N; ++i) {
        CHECK(std::isfinite(W(i)));
    }
    // And sorted ascending.
    for (int i = 1; i < N; ++i) {
        CHECK(W(i) >= W(i - 1));
    }
}
