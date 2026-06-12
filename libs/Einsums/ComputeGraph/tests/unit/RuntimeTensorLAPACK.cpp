//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase D.1, runtime-tensor coverage for the LAPACK ops in
// einsums::compute_graph (det, qr, svd, syev_eig). The C.11-vintage runtime
// overloads do their own copy-into-static / move-back wrap; these tests
// cross-check that round-trip against the long-standing static-tensor path
// so a regression in the wrap surfaces as a focused C++ failure.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <cstring>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

template <typename T>
RuntimeTensor<T> runtime_copy_of(Tensor<T, 2> const &t, bool row_major = false) {
    RuntimeTensor<T> out(t.name(), {t.dim(0), t.dim(1)}, row_major);
    std::memcpy(out.data(), t.data(), out.size() * sizeof(T));
    return out;
}

} // namespace

TEST_CASE("RuntimeTensor cg::det — matches static-tensor det", "[ComputeGraph][runtime]") {
    auto Astatic = create_random_tensor<double>("A", 4, 4);
    auto Art     = runtime_copy_of(Astatic);

    double const ref = cg::det(Astatic);
    double const got = cg::det(Art);

    CHECK_THAT(got, Catch::Matchers::WithinRel(ref, 1e-10));
}

TEST_CASE("RuntimeTensor cg::qr — Q*R reconstructs the input", "[ComputeGraph][runtime]") {
    auto Astatic = create_random_tensor<double>("A", 5, 3);
    auto Art     = runtime_copy_of(Astatic);

    auto [Qrt, Rrt] = cg::qr(Art);

    REQUIRE(Qrt.rank() == 2);
    REQUIRE(Rrt.rank() == 2);

    // Q @ R should reproduce A. Check elementwise.
    RuntimeTensor<double> Acheck("Acheck", {Astatic.dim(0), Astatic.dim(1)}, /*row_major=*/false);
    linear_algebra::gemm<false, false>(1.0, Qrt, Rrt, 0.0, &Acheck);

    for (size_t i = 0; i < Astatic.dim(0); ++i) {
        for (size_t j = 0; j < Astatic.dim(1); ++j) {
            CHECK_THAT(Acheck(i, j), Catch::Matchers::WithinRel(Astatic(i, j), 1e-9));
        }
    }
}

TEST_CASE("RuntimeTensor cg::svd — singular values match static SVD", "[ComputeGraph][runtime]") {
    auto Astatic = create_random_tensor<double>("A", 4, 3);
    auto Art     = runtime_copy_of(Astatic);

    auto const [Us_static, Sst, Vts_static] = cg::svd(Astatic);
    auto [Urt, Srt, Vtrt]                   = cg::svd(Art);

    REQUIRE(Srt.rank() == 1);
    REQUIRE(Sst.dim(0) == Srt.dim(0));

    // Singular values are unique (positive, sorted descending), so direct
    // elementwise comparison works. U / Vt may differ in sign per column.
    for (size_t i = 0; i < Sst.dim(0); ++i) {
        CHECK_THAT(Srt(i), Catch::Matchers::WithinRel(Sst(i), 1e-9));
    }
}

TEST_CASE("RuntimeTensor cg::trace — diagonal sum matches manual computation", "[ComputeGraph][runtime]") {
    constexpr int N       = 5;
    auto          Astatic = create_random_tensor<double>("A", N, N);
    auto          Art     = runtime_copy_of(Astatic);

    double const ref = cg::trace(Astatic);
    double const got = cg::trace(Art);
    CHECK_THAT(got, Catch::Matchers::WithinRel(ref, 1e-12));

    // And manually too.
    double manual = 0.0;
    for (int i = 0; i < N; ++i) {
        manual += Astatic(i, i);
    }
    CHECK_THAT(got, Catch::Matchers::WithinRel(manual, 1e-12));
}

TEST_CASE("RuntimeTensor cg::trace — non-square throws", "[ComputeGraph][runtime]") {
    RuntimeTensor<double> A("A", {3, 4});
    REQUIRE_THROWS(cg::trace(A));
}

TEST_CASE("RuntimeTensor cg::syev_eig — eigenvalues match numpy/static", "[ComputeGraph][runtime]") {
    // Build a known symmetric matrix.
    auto Astatic = create_random_tensor<double>("A", 5, 5);
    // Symmetrize.
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i + 1; j < 5; ++j) {
            double const m = 0.5 * (Astatic(i, j) + Astatic(j, i));
            Astatic(i, j)  = m;
            Astatic(j, i)  = m;
        }
    }
    auto Art = runtime_copy_of(Astatic);

    auto [evecs_rt, evals_rt] = cg::syev_eig(Art);
    REQUIRE(evals_rt.rank() == 1);
    REQUIRE(evals_rt.dim(0) == 5);

    // Compare against linear_algebra::syev (the static path).
    auto              a_copy = Astatic;
    Tensor<double, 1> w{"w", 5};
    linear_algebra::syev(&a_copy, &w);

    // Eigenvalues are sorted ascending by syev; both paths should agree.
    for (size_t i = 0; i < 5; ++i) {
        CHECK_THAT(evals_rt(i), Catch::Matchers::WithinAbs(w(i), 1e-9));
    }
}
