//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

#include <complex>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;

namespace {

template <typename T, size_t Rank>
void fill_sequential(Tensor<T, Rank> &t) {
    auto *ptr = t.data();
    for (size_t idx = 0; idx < t.size(); ++idx)
        ptr[idx] = T(static_cast<double>(idx) * 0.1 + 1.0);
}

template <typename T, size_t Rank>
    requires(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
void fill_sequential(Tensor<T, Rank> &t) {
    using Real = typename T::value_type;
    auto *ptr  = t.data();
    for (size_t idx = 0; idx < t.size(); ++idx)
        ptr[idx] = T(Real(idx) * Real(0.1) + Real(1.0), Real(idx) * Real(0.05) + Real(0.5));
}

} // namespace

// ===========================================================================
// Large matrix tests — exercise macro_kernel tiling
// ===========================================================================

TEMPLATE_TEST_CASE("Large rank-2 transpose", "[hptt][large]", float, double) {
    int const N = 64;

    auto A = create_random_tensor<TestType>("A", N, N);
    auto B = create_tensor<TestType>("B", N, N);
    B.zero();

    // B(j,i) = A(i,j)
    permute(Indices{j, i}, &B, Indices{i, j}, A);

    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            REQUIRE(B(jj, ii) == Catch::Approx(A(ii, jj)));
        }
    }
}

TEMPLATE_TEST_CASE("Large rank-2 transpose with alpha", "[hptt][large]", float, double) {
    int const N = 48;

    auto A = create_random_tensor<TestType>("A", N, N);
    auto B = create_tensor<TestType>("B", N, N);
    B.zero();

    auto alpha = TestType(2.5);

    // B(j,i) = alpha * A(i,j)
    permute(TestType(0.0), Indices{j, i}, &B, alpha, Indices{i, j}, A);

    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            REQUIRE(B(jj, ii) == Catch::Approx(alpha * A(ii, jj)));
        }
    }
}

TEMPLATE_TEST_CASE("Large rank-2 transpose with beta", "[hptt][large]", float, double) {
    int const N = 48;

    auto A = create_random_tensor<TestType>("A", N, N);
    auto B = create_random_tensor<TestType>("B", N, N);

    auto B_orig = create_tensor<TestType>("B_orig", N, N);
    B_orig      = B;

    auto alpha = TestType(1.5);
    auto beta  = TestType(0.5);

    // B(j,i) = beta * B(j,i) + alpha * A(i,j)
    permute(beta, Indices{j, i}, &B, alpha, Indices{i, j}, A);

    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            auto expected = alpha * A(ii, jj) + beta * B_orig(jj, ii);
            REQUIRE(B(jj, ii) == Catch::Approx(expected).epsilon(1e-4));
        }
    }
}

TEMPLATE_TEST_CASE("Large rank-2 complex transpose", "[hptt][large][complex]", std::complex<float>, std::complex<double>) {
    int const N = 32;

    auto A = create_tensor<TestType>("A", N, N);
    fill_sequential(A);

    auto B = create_tensor<TestType>("B", N, N);
    B.zero();

    permute(Indices{j, i}, &B, Indices{i, j}, A);

    using Real = typename TestType::value_type;
    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            REQUIRE(std::abs(B(jj, ii) - A(ii, jj)) < Real(1e-4));
        }
    }
}

TEMPLATE_TEST_CASE("Large rectangular transpose", "[hptt][large]", float, double) {
    int const M = 100, N = 50;

    auto A = create_random_tensor<TestType>("A", M, N);
    auto B = create_tensor<TestType>("B", N, M);
    B.zero();

    permute(Indices{j, i}, &B, Indices{i, j}, A);

    for (int ii = 0; ii < M; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            REQUIRE(B(jj, ii) == Catch::Approx(A(ii, jj)));
        }
    }
}

TEMPLATE_TEST_CASE("Large rank-3 transpose", "[hptt][large]", float, double) {
    int const D0 = 20, D1 = 25, D2 = 30;

    auto A = create_random_tensor<TestType>("A", D0, D1, D2);
    auto B = create_tensor<TestType>("B", D2, D0, D1);
    B.zero();

    permute(Indices{k, i, j}, &B, Indices{i, j, k}, A);

    for (int ii = 0; ii < D0; ++ii) {
        for (int jj = 0; jj < D1; ++jj) {
            for (int kk = 0; kk < D2; ++kk) {
                REQUIRE(B(kk, ii, jj) == Catch::Approx(A(ii, jj, kk)));
            }
        }
    }
}

// ===========================================================================
// Multi-threaded tests (use sort which calls HPTT with omp_get_max_threads)
// ===========================================================================

TEMPLATE_TEST_CASE("Large rank-2 transpose correctness", "[hptt][threaded]", float, double) {
    int const N = 128;

    auto A = create_random_tensor<TestType>("A", N, N);
    auto B = create_tensor<TestType>("B", N, N);
    B.zero();

    permute(Indices{j, i}, &B, Indices{i, j}, A);

    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            REQUIRE(B(jj, ii) == Catch::Approx(A(ii, jj)));
        }
    }
}

TEMPLATE_TEST_CASE("Large rank-3 transpose with beta, threaded", "[hptt][threaded]", float, double) {
    int const D0 = 30, D1 = 40, D2 = 50;

    auto A = create_random_tensor<TestType>("A", D0, D1, D2);
    auto B = create_random_tensor<TestType>("B", D1, D0, D2);

    auto B_orig = create_tensor<TestType>("B_orig", D1, D0, D2);
    B_orig      = B;

    auto alpha = TestType(2.0);
    auto beta  = TestType(0.3);

    permute(beta, Indices{j, i, k}, &B, alpha, Indices{i, j, k}, A);

    for (int ii = 0; ii < D0; ++ii) {
        for (int jj = 0; jj < D1; ++jj) {
            for (int kk = 0; kk < D2; ++kk) {
                auto expected = alpha * A(ii, jj, kk) + beta * B_orig(jj, ii, kk);
                REQUIRE(B(jj, ii, kk) == Catch::Approx(expected).margin(1e-5));
            }
        }
    }
}

// NOTE: MEASURE selection method is not tested here because the direct HPTT API
// requires careful column-major vs row-major handling that is abstracted by the
// permute() function. The MEASURE path is functionally identical to ESTIMATE
// except for plan selection timing — it uses the same micro_kernel code paths.
