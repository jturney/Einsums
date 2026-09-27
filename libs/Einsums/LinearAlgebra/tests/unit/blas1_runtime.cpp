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

TEMPLATE_TEST_CASE("RuntimeTensor axpy — complex alpha onto its own input", "[linear-algebra][runtime]", std::complex<float>,
                   std::complex<double>) {
    // Y += alpha*Y is (1 + alpha)*Y. It used to reach ?axpy with x == y, which
    // BLAS forbids: the complex kernel stored Re(y) before reading Re(x) for
    // Im(y), so (1+2i) + (0.5-0.75i)(1+2i) came back 3+0.75i. A real or
    // pure-real alpha has no cross term, which kept it hidden.
    using C = TestType;
    RuntimeTensor<C> Y("Y", {2, 2});
    fill_runtime(Y, {C{1, 2}, C{-2, 1}, C{0, -4}, C{4, 0}});

    linear_algebra::axpy(C{0.5, -0.75}, Y, &Y);
    CHECK_THAT(runtime_data(Y), Catch::Matchers::Equals(std::vector<C>{C{3, 2.25}, C{-2.25, 3}, C{-3, -6}, C{6, -3}}));
}

// ──────────────────────────────────────────────────────────────────────────
// Input and output are overlapping views of ONE parent
// ──────────────────────────────────────────────────────────────────────────
//
// Two views that share elements without being the same view used to reach the
// vendor BLAS-1 routine aliased, which BLAS forbids: each output element read a
// neighbour already updated, so real prefactors went wrong as well as complex
// ones. Only an input identical to its output (same start, same strides) was
// guarded. Each case builds its views over one parent buffer and compares the
// whole parent, element by element, against the parent as it was before the
// call; parent elements outside the output must come back untouched.

namespace {

/// A view into @p parent: @p offset elements from its start, with its own dims and strides.
struct ViewShape {
    size_t              offset;
    std::vector<size_t> dims;
    std::vector<size_t> strides;
};

struct OverlapCase {
    char const *name;
    size_t      parent_size;
    ViewShape   in;
    ViewShape   out;
};

// Parent layouts are column-major: a 4x3 parent has leading dimension 4.
std::vector<OverlapCase> const overlap_cases{
    {"vector, output one ahead of input", 4, {0, {3}, {1}}, {1, {3}, {1}}},
    {"vector, output one behind input", 4, {1, {3}, {1}}, {0, {3}, {1}}},
    {"matrix, row-shifted", 12, {0, {3, 3}, {1, 4}}, {1, {3, 3}, {1, 4}}},
    {"matrix, column-shifted", 12, {0, {3, 3}, {1, 3}}, {3, {3, 3}, {1, 3}}},
    {"matrix, transposed view onto its parent", 9, {0, {3, 3}, {3, 1}}, {0, {3, 3}, {1, 3}}},
    {"vector, disjoint halves (control)", 6, {0, {3}, {1}}, {3, {3}, {1}}},
};

/// The parent-relative offset of every element of @p v, in column-major logical order.
auto element_offsets(ViewShape const &v) -> std::vector<size_t> {
    size_t n = 1;
    for (auto d : v.dims) {
        n *= d;
    }
    std::vector<size_t> out;
    std::vector<size_t> idx(v.dims.size(), 0);
    for (size_t e = 0; e < n; e++) {
        size_t off = v.offset;
        for (size_t d = 0; d < idx.size(); d++) {
            off += idx[d] * v.strides[d];
        }
        out.push_back(off);
        for (size_t d = 0; d < idx.size() && ++idx[d] == v.dims[d]; d++) {
            idx[d] = 0;
        }
    }
    return out;
}

/// Small exact values: every sum and product below is exact in single precision.
template <typename T>
auto overlap_parent(size_t n) -> std::vector<T> {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; i++) {
        if constexpr (IsComplexV<T>) {
            v[i] = T{RemoveComplexT<T>(i + 1), RemoveComplexT<T>(static_cast<int>(i % 3) - 1)};
        } else {
            v[i] = T(i + 1);
        }
    }
    return v;
}

/// A prefactor whose imaginary part, on a complex type, gives every product a cross term.
template <typename T>
auto overlap_alpha() -> T {
    if constexpr (IsComplexV<T>) {
        return T{0.5, -0.75};
    } else {
        return T{2};
    }
}

template <typename T>
auto view_over(std::vector<T> &parent, ViewShape const &v) -> RuntimeTensorView<T> {
    return RuntimeTensorView<T>(detail::TensorImpl<T>(parent.data() + v.offset, v.dims, v.strides));
}

/// Runs @p op on views of one parent and checks it against @p reference applied to the parent as
/// it was: reference(x, y) gives the new value of an output element from its old input and output.
template <typename T, typename Op, typename Reference>
void check_overlapping(Op &&op, Reference &&reference) {
    for (auto const &c : overlap_cases) {
        DYNAMIC_SECTION(c.name) {
            auto parent = overlap_parent<T>(c.parent_size);
            auto before = parent;
            auto X      = view_over(parent, c.in);
            auto Y      = view_over(parent, c.out);

            auto expected = before;
            auto in_off   = element_offsets(c.in);
            auto out_off  = element_offsets(c.out);
            for (size_t e = 0; e < out_off.size(); e++) {
                expected[out_off[e]] = reference(before[in_off[e]], before[out_off[e]]);
            }

            op(X, Y);
            CHECK_THAT(parent, Catch::Matchers::Equals(expected));
        }
    }
}

} // namespace

TEMPLATE_LIST_TEST_CASE("RuntimeTensor axpy — input overlapping the output", "[linear-algebra][runtime][view]", testing::AllScalarTypes) {
    using T       = TestType;
    T const alpha = overlap_alpha<T>();
    check_overlapping<T>([&](auto const &X, auto &Y) { linear_algebra::axpy(alpha, X, &Y); }, [&](T x, T y) { return y + alpha * x; });
}

TEMPLATE_LIST_TEST_CASE("RuntimeTensor axpby — input overlapping the output", "[linear-algebra][runtime][view]", testing::AllScalarTypes) {
    using T       = TestType;
    T const alpha = overlap_alpha<T>();
    for (T const beta : {T{0}, T{0.5}}) {
        CAPTURE(beta);
        check_overlapping<T>([&](auto const &X, auto &Y) { linear_algebra::axpby(alpha, X, beta, &Y); },
                             [&](T x, T y) { return alpha * x + beta * y; });
    }
}

TEMPLATE_LIST_TEST_CASE("RuntimeTensor copy — input overlapping the output", "[linear-algebra][runtime][view]", testing::AllScalarTypes) {
    using T = TestType;
    check_overlapping<T>([](auto const &X, auto &Y) { detail::impl_copy(X.impl(), Y.impl()); }, [](T x, T) { return x; });
}

TEMPLATE_LIST_TEST_CASE("RuntimeTensor direct_product — input overlapping the output", "[linear-algebra][runtime][view]",
                        testing::AllScalarTypes) {
    using T       = TestType;
    T const alpha = overlap_alpha<T>();
    for (T const beta : {T{0}, T{0.5}}) {
        for (bool const shared_is_a : {true, false}) {
            CAPTURE(beta, shared_is_a);
            // The operand not shared with C is a separate tensor of constant value, so the
            // reference needs only the shared operand's old value.
            T const other = T{3};
            check_overlapping<T>(
                [&](auto const &X, auto &Y) {
                    RuntimeTensor<T> B("B", std::vector<size_t>(X.impl().dims().begin(), X.impl().dims().end()), /*row_major=*/false);
                    B = other;
                    if (shared_is_a) {
                        linear_algebra::direct_product(alpha, X, B, beta, &Y);
                    } else {
                        linear_algebra::direct_product(alpha, B, X, beta, &Y);
                    }
                },
                [&](T x, T y) { return alpha * x * other + beta * y; });
        }
    }
}

TEMPLATE_LIST_TEST_CASE("RuntimeTensor direct_product — in place on a strided view", "[linear-algebra][runtime][view]",
                        testing::AllScalarTypes) {
    // C = alpha*C*B + beta*C with A the very view C is. The alias guard sat on
    // the contiguous path only; a strided view (here rows 0..2 of a 4x3 parent)
    // takes the loops over the vendor routine, which scaled C by beta and then
    // read A, already scaled, back out of it.
    using T       = TestType;
    T const alpha = overlap_alpha<T>();
    for (T const beta : {T{0}, T{0.5}}) {
        CAPTURE(beta);
        auto            parent = overlap_parent<T>(12);
        auto const      before = parent;
        std::vector<T>  b_parent(12, T{3});
        ViewShape const shape{0, {3, 3}, {1, 4}};
        auto            Cv = view_over(parent, shape);
        auto const      Bv = view_over(b_parent, shape);

        auto expected = before;
        for (auto off : element_offsets(shape)) {
            expected[off] = alpha * before[off] * T{3} + beta * before[off];
        }

        linear_algebra::direct_product(alpha, Cv, Bv, beta, &Cv);
        CHECK_THAT(parent, Catch::Matchers::Equals(expected));
    }
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
