//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file OperandOwnership.cpp
/// @brief Capture takes shared ownership of its operands' storage.
///
/// A captured node used to hold its operands by raw pointer, which made three
/// separate hazards out of one missing ownership model:
///
/// * an operand that dies between capture and ``execute()`` leaves the node
///   reading freed memory (callers worked around it by keeping every
///   intermediate alive by hand for the whole capture),
/// * a parent written through per-slice views and then read whole in the same
///   graph was not ordered against those writes, so the reader saw a
///   half-written parent,
/// * a deferred tensor relocates when it materializes, so a view taken during
///   capture pointed at the pre-materialization buffer once the graph ran.
///
/// The fix is one ownership model: operand storage is refcounted, capture takes
/// a strong reference to it, a view holds one on its parent, and operand
/// handles resolve through a storage descriptor rather than a pointer snapshot.
/// These tests are that model's acceptance criteria.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <memory>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr double kTol = 1e-12;

/// A runtime-rank matrix filled with a deterministic, non-degenerate pattern.
/// Deterministic on purpose: these tests compare a graph result against an
/// eager one computed from the same data, and a seeded RNG would only add a
/// way for the two to differ for reasons that are not the thing under test.
RuntimeTensor<double> filled(std::string name, size_t rows, size_t cols, double seed) {
    RuntimeTensor<double> t(std::move(name), std::vector<size_t>{rows, cols});
    for (size_t i = 0; i < t.size(); i++) {
        t.data()[i] = std::sin(seed + static_cast<double>(i) * 0.37) + 0.1 * static_cast<double>(i % 7);
    }
    return t;
}

RuntimeTensor<double> zeros(std::string name, size_t rows, size_t cols) {
    RuntimeTensor<double> t(std::move(name), std::vector<size_t>{rows, cols});
    t.zero();
    return t;
}

void require_close(RuntimeTensor<double> const &got, RuntimeTensor<double> const &ref) {
    REQUIRE(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); i++) {
        REQUIRE_THAT(got.data()[i], Catch::Matchers::WithinAbs(ref.data()[i], kTol));
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// 1. An operand destroyed between capture and execute
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Operand ownership: a captured input destroyed before execute()", "[ComputeGraph][Ownership]") {
    constexpr size_t M = 6, K = 5, N = 4;

    auto C     = zeros("C", M, N);
    auto C_ref = zeros("C_ref", M, N);

    cg::Graph graph("dead_input");
    {
        // A and B die at the end of this scope, well before execute() below.
        // Nothing outside the graph refers to them afterwards.
        auto A = filled("A", M, K, 0.3);
        auto B = filled("B", K, N, 1.7);

        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C_ref);

        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, A, B, 0.0, &C);
    }

    graph.execute();
    require_close(C, C_ref);
}

TEST_CASE("Operand ownership: a captured intermediate destroyed before execute()", "[ComputeGraph][Ownership]") {
    constexpr size_t M = 6, K = 5, N = 4;

    auto A     = filled("A", M, K, 0.3);
    auto B     = filled("B", K, N, 1.7);
    auto D     = filled("D", N, N, 2.9);
    auto E     = zeros("E", M, N);
    auto E_ref = zeros("E_ref", M, N);

    {
        auto tmp = zeros("tmp_ref", M, N);
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &tmp);
        linear_algebra::gemm<false, false>(1.0, tmp, D, 0.0, &E_ref);
    }

    cg::Graph graph("dead_intermediate");
    {
        cg::CaptureGuard const guard(graph);
        // The classic hazard: an intermediate that exists only inside the
        // capture block, produced by one node and consumed by the next.
        auto tmp = zeros("tmp", M, N);
        cg::gemm<false, false>(1.0, A, B, 0.0, &tmp);
        cg::gemm<false, false>(1.0, tmp, D, 0.0, &E);
    }

    graph.execute();
    require_close(E, E_ref);
}

TEST_CASE("Operand ownership: a recycled address is not mistaken for its predecessor", "[ComputeGraph][Ownership]") {
    // Capture dedupes operands by address, which only identifies a tensor while
    // no captured tensor is ever freed. Once operands may die during a capture,
    // a destroyed wrapper's address is immediately reusable, and a tensor
    // allocated on top of a dead one would inherit its TensorId: every node
    // referring to it would then read the wrong operand, with no diagnostic.
    //
    // Found by deleting the DLPNO port's `tensors.arena()`, which existed to
    // keep every captured temporary alive and so kept addresses unique as a
    // side effect. The failure there was a gemm handed a rank-3 destination
    // because a freed (Q|ja) had been replaced by a rank-2 product.
    //
    // Placement new into one buffer makes the reuse exact rather than a race
    // with the allocator.
    constexpr size_t M = 6, K = 5, N = 4;

    alignas(RuntimeTensor<double>) std::byte slot_storage[sizeof(RuntimeTensor<double>)];
    auto                                    *slot_ptr = reinterpret_cast<RuntimeTensor<double> *>(slot_storage);

    auto A   = filled("A", M, K, 0.3);
    auto B   = filled("B", K, N, 1.7);
    auto out = zeros("out", M, N);
    auto ref = zeros("ref", M, N);
    linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &ref);

    cg::Graph graph("recycled_address");
    {
        cg::CaptureGuard const guard(graph);

        // A rank-3 tensor at the address, captured, then destroyed.
        std::construct_at(slot_ptr, "first (rank 3)", std::vector<size_t>{2, 3, 4});
        slot_ptr->zero();
        cg::scale(2.0, slot_ptr);
        std::destroy_at(slot_ptr);

        // A rank-2 tensor at the SAME address, captured as a gemm destination.
        // Resolving it to the rank-3 tensor's id fails in the executor with
        // "the inputs to gemm need to be matrices".
        std::construct_at(slot_ptr, "second (rank 2)", std::vector<size_t>{M, N});
        slot_ptr->zero();
        cg::gemm<false, false>(1.0, A, B, 0.0, slot_ptr);
    }

    graph.execute();
    require_close(*slot_ptr, ref);
    std::destroy_at(slot_ptr);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Per-slice writes ordered against a whole-parent read
// ═══════════════════════════════════════════════════════════════════════════

namespace {

/// Shared fixture for the slice-vs-whole hazard: NBLK independent GEMMs whose
/// destinations are column blocks of one store, plus a contraction that reads
/// the store whole.
struct SliceFixture {
    static constexpr size_t M = 8, K = 5, BW = 3, NBLK = 4;
    static constexpr size_t N = BW * NBLK;

    std::vector<RuntimeTensor<double>> As, Bs;
    RuntimeTensor<double>              X;
    RuntimeTensor<double>              store_ref, out_ref;

    SliceFixture() : X(filled("X", N, N, 5.1)), store_ref(zeros("store_ref", M, N)), out_ref(zeros("out_ref", M, N)) {
        As.reserve(NBLK);
        Bs.reserve(NBLK);
        for (size_t b = 0; b < NBLK; b++) {
            As.push_back(filled(fmt::format("A{}", b), M, K, 0.3 + static_cast<double>(b)));
            Bs.push_back(filled(fmt::format("B{}", b), K, BW, 1.7 + static_cast<double>(b)));
        }
        for (size_t b = 0; b < NBLK; b++) {
            auto dst = store_ref(AllT{}, Range{b * BW, (b + 1) * BW});
            linear_algebra::gemm<false, false>(1.0, As[b], Bs[b], 0.0, &dst);
        }
        linear_algebra::gemm<false, false>(1.0, store_ref, X, 0.0, &out_ref);
    }
};

} // namespace

TEST_CASE("Operand ownership: per-slice writes are ordered against a whole-parent read", "[ComputeGraph][Ownership]") {
    // The DLPNO residual's shape: one store per shape class, written by a
    // batched GEMM through per-pair column slices, then read as a whole tensor
    // by the next contraction. Without alias ordering the reader can run
    // before the writes and sees a half-written parent; the parent itself ends
    // up correct, so the only visible symptom is a wrong downstream result.
    SliceFixture fx;

    auto store = zeros("store", SliceFixture::M, SliceFixture::N);
    auto out   = zeros("out", SliceFixture::M, SliceFixture::N);

    // The slices are built outside the capture, exactly as a caller assembling
    // batched-GEMM operands up front does.
    std::vector<RuntimeTensorView<double>> dsts;
    dsts.reserve(SliceFixture::NBLK);
    for (size_t b = 0; b < SliceFixture::NBLK; b++) {
        dsts.push_back(store(AllT{}, Range{b * SliceFixture::BW, (b + 1) * SliceFixture::BW}));
    }

    cg::Graph graph("slice_then_whole");
    {
        cg::CaptureGuard const guard(graph);

        std::vector<RuntimeTensor<double> const *> a_ptrs, b_ptrs;
        std::vector<RuntimeTensorView<double> *>   c_ptrs;
        for (size_t b = 0; b < SliceFixture::NBLK; b++) {
            a_ptrs.push_back(&fx.As[b]);
            b_ptrs.push_back(&fx.Bs[b]);
            c_ptrs.push_back(&dsts[b]);
        }
        cg::batched_gemm(1.0, a_ptrs, b_ptrs, 0.0, c_ptrs);

        // Reads the WHOLE parent the batch just wrote through slices. This is
        // the edge the scheduler has to see. No graph boundary here on purpose.
        cg::gemm<false, false>(1.0, store, fx.X, 0.0, &out);
    }

    graph.execute();
    require_close(store, fx.store_ref);
    require_close(out, fx.out_ref);
}

TEST_CASE("Operand ownership: per-slice writes ordered under the default pass manager", "[ComputeGraph][Ownership]") {
    // Same hazard, but with the optimizer free to reorder and batch. A pass
    // that does not know the slices alias the parent has more chances to get
    // the order wrong than the capture-order executor does.
    SliceFixture fx;

    auto store = zeros("store", SliceFixture::M, SliceFixture::N);
    auto out   = zeros("out", SliceFixture::M, SliceFixture::N);

    std::vector<RuntimeTensorView<double>> dsts;
    dsts.reserve(SliceFixture::NBLK);
    for (size_t b = 0; b < SliceFixture::NBLK; b++) {
        dsts.push_back(store(AllT{}, Range{b * SliceFixture::BW, (b + 1) * SliceFixture::BW}));
    }

    cg::Graph graph("slice_then_whole_opt");
    {
        cg::CaptureGuard const guard(graph);
        for (size_t b = 0; b < SliceFixture::NBLK; b++) {
            cg::gemm<false, false>(1.0, fx.As[b], fx.Bs[b], 0.0, &dsts[b]);
        }
        cg::gemm<false, false>(1.0, store, fx.X, 0.0, &out);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();
    require_close(store, fx.store_ref);
    require_close(out, fx.out_ref);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. A view of a deferred tensor survives materialization
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Operand ownership: a view of a declared tensor reads correctly after execute()", "[ComputeGraph][Ownership]") {
    // declare_tensor hands back a shell with no storage. Materialization
    // allocates it during execute(), which relocates the buffer; a view taken
    // before that used to keep pointing at the old (null) one.
    constexpr size_t M = 6, K = 5, N = 4;

    auto A = create_random_tensor<double>("A", M, K);
    auto B = create_random_tensor<double>("B", K, N);

    auto ref = create_zero_tensor<double>("ref", M, N);
    linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &ref);

    cg::Graph graph("declared_view");
    auto     &D = graph.declare_tensor<double, 2>("D", M, N);
    REQUIRE_FALSE(D.is_materialized());

    TensorView<double, 2> *slice = nullptr;
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, A, B, 0.0, &D);
        // Taken DURING capture, before D has any storage.
        slice = &cg::view(D, cg::ViewAxis::full(), cg::ViewAxis::range(1, 3));
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    REQUIRE(D.is_materialized());
    REQUIRE(slice != nullptr);
    REQUIRE(slice->dim(0) == M);
    REQUIRE(slice->dim(1) == 2);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < 2; j++) {
            REQUIRE_THAT((*slice)(i, j), Catch::Matchers::WithinAbs(ref(i, j + 1), kTol));
        }
    }
}

TEST_CASE("Operand ownership: a view outlives the parent it was taken from", "[ComputeGraph][Ownership]") {
    // A view captured as an operand holds a strong reference to its parent's
    // storage, so the parent wrapper going out of scope is not a use-after-free.
    constexpr size_t M = 6, K = 5, N = 4;

    auto A   = filled("A", M, K, 0.3);
    auto B   = filled("B", K, N, 1.7);
    auto ref = zeros("ref", M, N);
    linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &ref);

    cg::Graph                                  graph("orphaned_view");
    std::unique_ptr<RuntimeTensorView<double>> slice;
    {
        auto parent = zeros("parent", M, N);
        slice       = std::make_unique<RuntimeTensorView<double>>(parent(AllT{}, AllT{}));

        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, A, B, 0.0, slice.get());
    }

    graph.execute();
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            REQUIRE_THAT((*slice)(i, j), Catch::Matchers::WithinAbs(ref.data()[j * M + i], kTol));
        }
    }
}
