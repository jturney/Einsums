//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file StridedBatchedGemm.cpp
/// @brief Tests for the strided-batch GEMM route of an einsum node: a batch of
///        equal matrix products runs as one `blas::gemm_batch` call with
///        base-plus-stride pointers on CPU, the layout `cublasDgemmStridedBatched`
///        takes on GPU. The node stays an einsum, so it saves and loads like any
///        other; the route is chosen when its executor is built.
///
/// The pattern accepted depends on the tensor layout:
///   - Column-major (Einsums default): batch at the LAST axis of each operand,
///     e.g. ``cg::einsum("ijb;jkb->ikb", &C, A, B)`` with shapes
///     ``(I, J, B) × (J, K, B) → (I, K, B)``.
///   - Row-major: batch at the FIRST axis, e.g. ``cg::einsum("bij;bjk->bik", ...)``
///     with shapes ``(B, I, J) × (B, J, K) → (B, I, K)``.
///
/// Either produces contiguous 2D slices in memory and can therefore be
/// strided-batched. Other arrangements (col-major batch-first, row-major
/// batch-last) have interleaved batches and fall through to generic einsum.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>
#include <Einsums/Testing/TensorCompare.hpp>

#include <cstring>
#include <limits>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

std::string route() {
    return std::string{cg::dispatch::last_dispatch_route()};
}

constexpr char const *kStrided = "strided_batched_gemm";

/// A few ulps of @p T: every element here is a short sum of order-one products.
template <typename T>
double tol() {
    return 100.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon();
}

template <typename T, size_t R>
void require_close(Tensor<T, R> const &got, Tensor<T, R> const &ref) {
    einsums::testing::require_tensors_close(got, ref, {.rtol = tol<T>(), .atol = tol<T>()});
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Fast-path capture: column-major, batch at last axis (Einsums default)
// ═══════════════════════════════════════════════════════════════════════════

// The route used to be taken at capture by recording a BatchedGemm node, which a
// graph refuses to save, so capturing "ijb;jkb->ikb" in the default layout made
// the whole graph unsaveable. The node is now an einsum that picks the route
// when its executor is built, so the loader reaches the same route.
TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: col-major 3D ijb;jkb->ikb stays an einsum, runs batched and saves",
                        "[ComputeGraph][StridedBatchedGemm]", testing::AllScalarTypes) {
    using T            = TestType;
    constexpr size_t I = 3, J = 5, K = 2, B = 4;
    auto             A  = create_random_tensor<T>("A", I, J, B);
    auto             Bt = create_random_tensor<T>("B", J, K, B);
    auto             C  = create_zero_tensor<T>("C", I, K, B);

    cg::Graph graph("strided3d_colmajor");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijb;jkb->ikb", &C, A, Bt);
    }
    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(graph.nodes()[0].kind == cg::OpKind::Einsum);

    graph.execute();
    CHECK(route() == kStrided);
    auto const expected = Tensor<T, 3>(C);

    auto const text = cg::save_graph_string(graph);
    REQUIRE(text.has_value());
    auto loaded = cg::load_graph_string(*text);
    REQUIRE(loaded.has_value());

    auto C2 = create_zero_tensor<T>("C2", I, K, B);
    loaded->bind("A", A, "B", Bt, "C", C2);
    loaded->execute();
    CHECK(route() == kStrided);
    require_close(C2, expected);
}

TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: col-major result matches slice-by-slice reference", "[ComputeGraph][StridedBatchedGemm]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    constexpr size_t I = 4, J = 6, K = 3, B = 5;
    auto             A  = create_random_tensor<T>("A", I, J, B);
    auto             Bt = create_random_tensor<T>("B", J, K, B);

    // Reference: slice-by-slice 2D einsum via the normal capture path.
    // Each batch slice occupies a contiguous I×J (or J×K, I×K) block of
    // the underlying column-major storage.
    auto C_ref = create_zero_tensor<T>("C_ref", I, K, B);
    {
        for (size_t b = 0; b < B; b++) {
            // For col-major with batch at axis 2: slice starts at
            // data() + b * (I * J) [for A], etc.
            T const *a_slice = A.data() + b * I * J;
            T const *b_slice = Bt.data() + b * J * K;
            T       *c_slice = C_ref.data() + b * I * K;
            // Compute C[i,k] = sum_j A[i,j] * B[j,k] for col-major storage:
            //   A[i,j] at a_slice[i + j*I]
            //   B[j,k] at b_slice[j + k*J]
            //   C[i,k] at c_slice[i + k*I]
            for (size_t k = 0; k < K; k++)
                for (size_t i = 0; i < I; i++) {
                    T s{0};
                    for (size_t j = 0; j < J; j++)
                        s += a_slice[i + j * I] * b_slice[j + k * J];
                    c_slice[i + k * I] = s;
                }
        }
    }

    auto      C_cg = create_zero_tensor<T>("C_cg", I, K, B);
    cg::Graph graph("result");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijb;jkb->ikb", &C_cg, A, Bt);
    }
    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(graph.nodes()[0].kind == cg::OpKind::Einsum);

    graph.execute();
    CHECK(route() == kStrided);
    require_close(C_cg, C_ref);
}

TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: nonzero c_prefactor accumulates correctly", "[ComputeGraph][StridedBatchedGemm]",
                        testing::AllScalarTypes) {
    using T                = TestType;
    T const          c_pf  = testing::prefactor<T>(0.5, 0.3);
    T const          ab_pf = testing::prefactor<T>(2.1, -0.7);
    constexpr size_t I = 4, J = 5, K = 3, B = 3;
    auto             A  = create_random_tensor<T>("A", I, J, B);
    auto             Bt = create_random_tensor<T>("B", J, K, B);
    auto             C  = create_random_tensor<T>("C", I, K, B);

    // Reference: C = c_pf*C + ab_pf*A@B, slice by slice (col-major layout).
    auto C_ref = Tensor<T, 3>(C);
    for (size_t b = 0; b < B; b++) {
        T const *a_slice = A.data() + b * I * J;
        T const *b_slice = Bt.data() + b * J * K;
        T       *c_slice = C_ref.data() + b * I * K;
        for (size_t k = 0; k < K; k++)
            for (size_t i = 0; i < I; i++) {
                T s{0};
                for (size_t j = 0; j < J; j++)
                    s += a_slice[i + j * I] * b_slice[j + k * J];
                c_slice[i + k * I] = c_pf * c_slice[i + k * I] + ab_pf * s;
            }
    }

    cg::Graph graph("beta");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijb;jkb->ikb", c_pf, &C, ab_pf, A, Bt);
    }
    graph.execute();
    CHECK(route() == kStrided);
    require_close(C, C_ref);
}

// The capture-time route copied the prefactors into its descriptor, so a later
// update to the node's live scalars was ignored on replay. The route now reads
// the live values on every call.
TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: an updated prefactor is honored on replay", "[ComputeGraph][StridedBatchedGemm]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    constexpr size_t I = 3, J = 4, K = 2, B = 3;
    auto             A  = create_random_tensor<T>("A", I, J, B);
    auto             Bt = create_random_tensor<T>("B", J, K, B);
    auto             C  = create_zero_tensor<T>("C", I, K, B);

    cg::Graph graph("live_prefactor");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijb;jkb->ikb", &C, A, Bt);
    }
    graph.execute();
    auto const once = Tensor<T, 3>(C);

    T const scale = testing::prefactor<T>(3.0, -1.5);
    graph.update_prefactors(graph.nodes()[0].id, T{0}, scale);
    graph.execute();
    CHECK(route() == kStrided);
    for (size_t n = 0; n < C.size(); ++n) {
        REQUIRE(std::abs(C.data()[n] - scale * once.data()[n]) <= tol<T>() * (1.0 + std::abs(scale * once.data()[n])));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fall-through: patterns NOT matching the fast path
// ═══════════════════════════════════════════════════════════════════════════

TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: 3D einsum without batch index falls through to generic", "[ComputeGraph][StridedBatchedGemm]",
                        testing::AllScalarTypes) {
    using T = TestType;
    // pqr;rs->pqs has no batch index (p, q are targets only in A/C; r is link).
    // Not a batched-GEMM pattern, generic nested-loop executor handles it.
    auto X = create_random_tensor<T>("X", 2, 3, 4);
    auto M = create_random_tensor<T>("M", 4, 5);
    auto C = create_zero_tensor<T>("C", 2, 3, 5);

    cg::Graph graph("nonbatch");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("pqr;rs->pqs", &C, X, M);
    }
    graph.execute();
    CHECK(route() != kStrided);
}

TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: col-major with batch at front falls through (interleaved)",
                        "[ComputeGraph][StridedBatchedGemm]", testing::AllScalarTypes) {
    using T = TestType;
    // In Einsums's default column-major layout, batch at position 0
    // interleaves slices in memory, cannot be strided-batched. Must
    // fall through to the generic path; correctness preserved.
    constexpr size_t B = 4, I = 3, J = 4, K = 5;
    auto             A  = create_random_tensor<T>("A", B, I, J);
    auto             Bt = create_random_tensor<T>("B", B, J, K);
    auto             C  = create_zero_tensor<T>("C", B, I, K);

    cg::Graph graph("batch_front_colmajor");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("bij;bjk->bik", &C, A, Bt);
    }
    graph.execute();
    CHECK(route() != kStrided);
}

// A strided batch of plain GEMMs cannot conjugate, so a conjugated operand
// leaves the route to the general dispatch, which conjugates while packing.
TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: a conjugated operand falls through", "[ComputeGraph][StridedBatchedGemm]",
                        testing::ComplexScalarTypes) {
    using T            = TestType;
    constexpr size_t I = 4, J = 5, K = 3, B = 3;
    auto             A  = create_random_tensor<T>("A", I, J, B);
    auto             Bt = create_random_tensor<T>("B", J, K, B);
    auto             C  = create_zero_tensor<T>("C", I, K, B);

    auto C_ref = create_zero_tensor<T>("C_ref", I, K, B);
    testing::reference_einsum("ikb <- ijb ; jkb", T{0}, &C_ref, T{1}, A, Bt, /*conj_a=*/true);

    cg::Graph graph("conj");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb <- conj(ijb) ; jkb", &C, A, Bt);
    }
    graph.execute();
    CHECK(route() != kStrided);
    require_close(C, C_ref);
}

// Conjugating a real operand is the identity, so the flag is dropped at capture
// and the node keeps the strided route. It used to fall through with the
// complex case above, to a slower route that computed the same values.
TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: a conjugated real operand keeps the route", "[ComputeGraph][StridedBatchedGemm]",
                        testing::RealScalarTypes) {
    using T            = TestType;
    constexpr size_t I = 4, J = 5, K = 3, B = 3;
    auto             A  = create_random_tensor<T>("A", I, J, B);
    auto             Bt = create_random_tensor<T>("B", J, K, B);
    auto             C  = create_zero_tensor<T>("C", I, K, B);

    auto C_ref = create_zero_tensor<T>("C_ref", I, K, B);
    testing::reference_einsum("ikb <- ijb ; jkb", T{0}, &C_ref, T{1}, A, Bt);

    cg::Graph graph("conj_real");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb <- conj(ijb) ; conj(jkb)", &C, A, Bt);
    }
    graph.execute();
    CHECK(route() == kStrided);
    require_close(C, C_ref);
}

// ═══════════════════════════════════════════════════════════════════════════
// Replay correctness
// ═══════════════════════════════════════════════════════════════════════════

TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: forcing Target::GPU routes through gpu::blas dispatcher",
                        "[ComputeGraph][StridedBatchedGemm][GPU]", testing::AllScalarTypes) {
    using T = TestType;
    // Exercises the GPU dispatch path end-to-end. On CUDA/HIP builds this
    // hits cublas/hipblas ?gemmStridedBatched directly; on mock / MPS
    // builds it transparently falls back to the CPU pointer-array
    // gemm_batch inside gpu::blas::?gemm_strided_batched. Either way the
    // math is identical, so correctness is the assertion.
    constexpr size_t I = 4, J = 5, K = 3, B = 4;
    auto             A     = create_random_tensor<T>("A", I, J, B);
    auto             Bt    = create_random_tensor<T>("B", J, K, B);
    auto             C_ref = create_zero_tensor<T>("C_ref", I, K, B);

    for (size_t b = 0; b < B; b++) {
        T const *a_slice = A.data() + b * I * J;
        T const *b_slice = Bt.data() + b * J * K;
        T       *c_slice = C_ref.data() + b * I * K;
        for (size_t k = 0; k < K; k++)
            for (size_t i = 0; i < I; i++) {
                T s{};
                for (size_t j = 0; j < J; j++)
                    s += a_slice[i + j * I] * b_slice[j + k * J];
                c_slice[i + k * I] = s;
            }
    }

    auto      C = create_zero_tensor<T>("C", I, K, B);
    cg::Graph graph("gpu_path");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijb;jkb->ikb", &C, A, Bt);
    }
    REQUIRE(graph.num_nodes() == 1);

    // Force GPU placement by hand (bypassing the GPUPlacement pass's
    // cost-model decision). The graph's execute() sees target == GPU
    // and routes through try_gpu_blas_dispatch, which plans the einsum
    // as a strided batch and issues gpu::blas::gemm_strided_batched<T>.
    graph.nodes()[0].target = cg::Target::GPU;

    graph.execute();
    require_close(C, C_ref);
}

// ═══════════════════════════════════════════════════════════════════════════
// Higher-rank batches: multiple batch indices flattened into one GEMM batch
// ═══════════════════════════════════════════════════════════════════════════

TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: rank-4 with two batch indices (col-major, batches last)", "[ComputeGraph][StridedBatchedGemm]",
                        testing::AllScalarTypes) {
    using T = TestType;
    // "ijab;jkab->ikab", two batch indices (a, b) at positions [2, 3].
    // Col-major contiguous → batches are the outermost axes in memory.
    // Effective batch_count = A * B = 4 * 3 = 12.
    constexpr size_t I = 3, J = 4, K = 2, A_ = 4, B_ = 3;
    auto             A  = create_random_tensor<T>("A", I, J, A_, B_);
    auto             Bt = create_random_tensor<T>("B", J, K, A_, B_);

    auto C_ref = create_zero_tensor<T>("C_ref", I, K, A_, B_);
    // Reference: iterate (a, b) jointly and slice-by-slice 2D GEMM.
    // Col-major storage: T(i,j,a,b) at offset i + j*I + a*I*J + b*I*J*A_.
    for (size_t b = 0; b < B_; b++)
        for (size_t a = 0; a < A_; a++) {
            std::size_t const ab_offset_a = (a + b * A_);
            T const          *a_slice     = A.data() + ab_offset_a * I * J;
            T const          *b_slice     = Bt.data() + ab_offset_a * J * K;
            T                *c_slice     = C_ref.data() + ab_offset_a * I * K;
            for (size_t k = 0; k < K; k++)
                for (size_t i = 0; i < I; i++) {
                    T s{};
                    for (size_t j = 0; j < J; j++)
                        s += a_slice[i + j * I] * b_slice[j + k * J];
                    c_slice[i + k * I] = s;
                }
        }

    auto      C = create_zero_tensor<T>("C", I, K, A_, B_);
    cg::Graph graph("rank4_two_batch");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijab;jkab->ikab", &C, A, Bt);
    }
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();
    CHECK(route() == kStrided);
    require_close(C, C_ref);
}

TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: rank-5 with three batch indices", "[ComputeGraph][StridedBatchedGemm]",
                        testing::AllScalarTypes) {
    using T = TestType;
    // "ijabc;jkabc->ikabc", three batch indices.
    // Flat batch = 2 * 3 * 2 = 12.
    constexpr size_t I = 3, J = 4, K = 2, A_ = 2, B_ = 3, C_ = 2;
    auto             A  = create_random_tensor<T>("A", I, J, A_, B_, C_);
    auto             Bt = create_random_tensor<T>("B", J, K, A_, B_, C_);
    auto             C  = create_zero_tensor<T>("C", I, K, A_, B_, C_);

    cg::Graph graph("rank5_three_batch");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijabc;jkabc->ikabc", &C, A, Bt);
    }
    REQUIRE(graph.num_nodes() == 1);

    // Correctness: compute the same result slice-by-slice.
    graph.execute();
    CHECK(route() == kStrided);

    auto C_ref = create_zero_tensor<T>("C_ref", I, K, A_, B_, C_);
    for (size_t flat = 0; flat < A_ * B_ * C_; flat++) {
        T const *a_slice = A.data() + flat * I * J;
        T const *b_slice = Bt.data() + flat * J * K;
        T       *c_slice = C_ref.data() + flat * I * K;
        for (size_t k = 0; k < K; k++)
            for (size_t i = 0; i < I; i++) {
                T s{};
                for (size_t j = 0; j < J; j++)
                    s += a_slice[i + j * I] * b_slice[j + k * J];
                c_slice[i + k * I] = s;
            }
    }
    require_close(C, C_ref);
}

TEST_CASE("StridedBatchedGemm: batch indices at non-matching positions fall through", "[ComputeGraph][StridedBatchedGemm]") {
    // "ijab;jkba->ikab", batch (a,b) order differs between A and B.
    // The stride-flattening assumption breaks, we can't index both with
    // a single flat batch counter. Must fall through to generic einsum.
    constexpr size_t I = 3, J = 3, K = 3, A_ = 3, B_ = 3;
    auto             A  = create_random_tensor<double>("A", I, J, A_, B_);
    auto             Bt = create_random_tensor<double>("B", J, K, B_, A_); // (b, a) swapped
    auto             C  = create_zero_tensor<double>("C", I, K, A_, B_);

    cg::Graph graph("batch_order_mismatch");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijab;jkba->ikab", &C, A, Bt);
    }
    graph.execute();
    CHECK(route() != kStrided);
}

TEMPLATE_LIST_TEST_CASE("StridedBatchedGemm: replay across multiple execute() calls", "[ComputeGraph][StridedBatchedGemm]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    constexpr size_t I = 3, J = 3, K = 3, B = 4;
    auto             A  = create_random_tensor<T>("A", I, J, B);
    auto             Bt = create_random_tensor<T>("B", J, K, B);
    auto             C  = create_zero_tensor<T>("C", I, K, B);

    cg::Graph graph("replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijb;jkb->ikb", &C, A, Bt);
    }
    graph.execute();
    CHECK(route() == kStrided);
    auto snap = Tensor<T, 3>(C);

    C.zero();
    graph.execute();
    require_close(C, snap);
}

TEST_CASE("StridedBatchedGemm: permuted view with transposed slice falls through", "[ComputeGraph][StridedBatchedGemm]") {
    // Regression guard: a permute_view keeps its parent's storage-order flag but
    // presents reordered strides. A view of (L, J, I) presented as (J, L, I)
    // keeps the batch axis outermost - so is_contiguous(), is_column_major(),
    // and the batch-suffix check all passed - but each 2D slice is TRANSPOSED
    // in memory, and the strided-batch stride math silently computed garbage.
    // The layout gate must reject it (fall through to generic) and the result
    // must match. Found by the Python large-rank differential fuzzer.
    constexpr size_t J = 2, L = 2, I = 2, K = 3;
    auto             base = create_random_tensor<double>("base", L, J, I); // (l, j, i) storage
    auto             B    = create_random_tensor<double>("B", L, K, I);
    auto             C    = create_zero_tensor<double>("C", J, K, I);

    RuntimeTensor<double>     base_rt(base);
    RuntimeTensorView<double> At = base_rt.permute_view({1, 0, 2}); // presents (j, l, i)

    RuntimeTensor<double> B_rt(B), C_rt(C);

    cg::Graph graph("view_transposed_slice");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("jki <- jli ; lki", 0.0, &C_rt, 1.0, At, B_rt);
    }
    graph.execute();
    CHECK(route() != kStrided);

    for (size_t j = 0; j < J; j++) {
        for (size_t k = 0; k < K; k++) {
            for (size_t i2 = 0; i2 < I; i2++) {
                double ref = 0.0;
                for (size_t l = 0; l < L; l++) {
                    ref += base(l, j, i2) * B(l, k, i2);
                }
                REQUIRE(std::abs(
                            C_rt(std::vector<ptrdiff_t>{static_cast<ptrdiff_t>(j), static_cast<ptrdiff_t>(k), static_cast<ptrdiff_t>(i2)}) -
                            ref) < 1e-12);
            }
        }
    }
}
