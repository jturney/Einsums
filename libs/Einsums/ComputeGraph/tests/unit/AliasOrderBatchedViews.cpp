//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file AliasOrderBatchedViews.cpp
/// @brief Ordering of accesses through views against their parent, including a
/// view that covers the parent EXACTLY.
///
/// Found by bisecting DLPNO. `lmp2_iterations` runs its iteration as seven
/// graphs; merging the couplings, repack and residual into one changed exactly
/// ONE pair's residual out of 25, and that pair is the sole member of the only
/// two shape classes with a single member. Every class with 4, 12 or 16 members
/// is bit-identical.
///
/// The root cause was two compounding defects, both now fixed:
///
///  1. `Graph::link_alias_storage()` required an owner to have STRICTLY more
///     elements than the alias, so a view spanning its whole parent (equal
///     byte span, equal element count) linked to nothing. A single-member
///     shape class is exactly that: its `_W_pair` slice covers all of `_W`.
///     With no link, the hazard scan saw two unrelated tensors on one buffer
///     and emitted no edge between the repack's parent write and the
///     residual's view read.
///  2. `Graph::topological_sort()` used a FIFO ready queue, which does not
///     preserve program order: a zero-in-degree node LATE in capture order
///     pops ahead of an EARLIER node that waits on any edge. The missing edge
///     from (1) therefore became a reorder, not a harmless redundancy - the
///     view-reading batch ran before the parent write it consumes.
///
/// That is why the failure needed three or more merged phases: the reorder
/// requires a producer that is DELAYED behind an in-edge while the unlinked
/// consumer is initially ready. Two-node producer/consumer reductions (both
/// initially ready, FIFO keeps index order) can never reproduce it, which is
/// how seven synthetic hypotheses in a row came back green.
///
/// The damage is easy to under-read from the top. One wrong pair of 25 shows up
/// as 4e-03 in the residual, 9e-05 in the amplitudes after the Jacobi step, and
/// 5e-08 in the converged correlation energy, because the solver re-converges
/// around the bad block. The energy-level number looks like rounding and is not.
///
/// The batch-of-16 case is the control: it passed before this file existed,
/// and if a change ever makes it fail the diagnosis moves from "the ordering"
/// to "the batched path".

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Parent written whole, then `count` views of it accumulated into by ONE
/// batched node, with other nodes present in the same graph.
///
/// @param count      batch length; 1 is the degenerate case DLPNO hits.
/// @param companions extra unrelated nodes in the graph. DLPNO's failure needs
///                   the residual to share a graph with the couplings and
///                   repack; alone in its own graph it is correct.
/// @return max |element - 1.0| over the store; 0 when the ordering holds.
double run_case(size_t count, size_t companions) {
    constexpr size_t m = 4, n = 4;

    RuntimeTensor<double> store("store", std::vector<size_t>{m, n * count});

    std::vector<RuntimeTensor<double>> a_store, b_store;
    a_store.reserve(count);
    b_store.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        RuntimeTensor<double> a("a", std::vector<size_t>{m, m});
        RuntimeTensor<double> b("b", std::vector<size_t>{m, n});
        a.zero();
        b.zero();
        for (size_t d = 0; d < m; ++d) {
            a(std::vector<ptrdiff_t>{static_cast<ptrdiff_t>(d), static_cast<ptrdiff_t>(d)}) = 1.0; // identity
        }
        for (size_t r = 0; r < m; ++r) {
            for (size_t c = 0; c < n; ++c) {
                b(std::vector<ptrdiff_t>{static_cast<ptrdiff_t>(r), static_cast<ptrdiff_t>(c)}) = 1.0; // ones
            }
        }
        a_store.push_back(std::move(a));
        b_store.push_back(std::move(b));
    }

    std::vector<RuntimeTensor<double> const *> a_list, b_list;
    for (size_t i = 0; i < count; ++i) {
        a_list.push_back(&a_store[i]);
        b_list.push_back(&b_store[i]);
    }

    // Views made once and kept alive for the graph's lifetime, as a captured
    // view operand must be (examples/dlpno/dlpno/mp2.py:384).
    std::vector<RuntimeTensorView<double>> views;
    views.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        views.push_back(store(AllT{}, Range{i * n, (i + 1) * n}));
    }
    std::vector<RuntimeTensorView<double> *> c_list;
    c_list.reserve(count);
    for (auto &v : views) {
        c_list.push_back(&v);
    }

    // Unrelated work sharing the graph, standing in for the couplings/repack.
    std::vector<RuntimeTensor<double>> filler;
    filler.reserve(companions);
    for (size_t i = 0; i < companions; ++i) {
        RuntimeTensor<double> f("filler", std::vector<size_t>{m, n});
        f.zero();
        filler.push_back(std::move(f));
    }

    cg::Graph graph("batched view write");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &store); // the parent, written whole (DLPNO's prologue)
        for (auto &f : filler) {
            cg::scale(2.0, &f); // touches nothing the batch touches
        }
        cg::batched_gemm(1.0, a_list, b_list, 1.0, c_list); // accumulate into the views
    }
    graph.execute();

    double worst = 0.0;
    for (size_t r = 0; r < m; ++r) {
        for (size_t c = 0; c < n * count; ++c) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(r), static_cast<ptrdiff_t>(c)};
            worst = std::max(worst, std::abs(store(idx) - 1.0));
        }
    }
    return worst;
}

/// The three-node shape that reproduces the DLPNO defect: a feeder, a parent
/// write DELAYED behind the feeder, and a batch reading views of the parent.
/// At `count == 1` the single view covers the parent exactly; before the fix
/// the batch was hoisted above the parent write and accumulated into a store
/// the write then clobbered.
///
/// @param count batch length; 1 makes the view full-cover (the failing case),
///              larger counts make every view a strict subview (control).
/// @return max |element - expected| over the store; 0 when the ordering holds.
double run_delayed_producer_case(size_t count) {
    constexpr size_t m = 4, n = 4;

    RuntimeTensor<double> feeder("feeder", std::vector<size_t>{m, n * count});
    RuntimeTensor<double> store("store", std::vector<size_t>{m, n * count});
    feeder.zero();
    store.zero();
    for (size_t r = 0; r < m; ++r) {
        for (size_t c = 0; c < n * count; ++c) {
            feeder(std::vector<ptrdiff_t>{static_cast<ptrdiff_t>(r), static_cast<ptrdiff_t>(c)}) = 3.0;
        }
    }

    std::vector<RuntimeTensor<double>> a_store, b_store;
    a_store.reserve(count);
    b_store.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        RuntimeTensor<double> a("a", std::vector<size_t>{m, m});
        RuntimeTensor<double> b("b", std::vector<size_t>{m, n});
        a.zero();
        b.zero();
        for (size_t d = 0; d < m; ++d) {
            a(std::vector<ptrdiff_t>{static_cast<ptrdiff_t>(d), static_cast<ptrdiff_t>(d)}) = 1.0; // identity
        }
        for (size_t r = 0; r < m; ++r) {
            for (size_t c = 0; c < n; ++c) {
                b(std::vector<ptrdiff_t>{static_cast<ptrdiff_t>(r), static_cast<ptrdiff_t>(c)}) = 1.0; // ones
            }
        }
        a_store.push_back(std::move(a));
        b_store.push_back(std::move(b));
    }
    std::vector<RuntimeTensor<double> const *> a_list, b_list;
    for (size_t i = 0; i < count; ++i) {
        a_list.push_back(&a_store[i]);
        b_list.push_back(&b_store[i]);
    }

    std::vector<RuntimeTensorView<double>> views;
    views.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        views.push_back(store(AllT{}, Range{i * n, (i + 1) * n}));
    }
    std::vector<RuntimeTensorView<double> *> c_list;
    c_list.reserve(count);
    for (auto &v : views) {
        c_list.push_back(&v);
    }

    cg::Graph graph("delayed parent write, view batch");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &feeder);                            // feeder = 6; the producer below must wait on this
        cg::axpby(1.0, feeder, 0.0, &store);                // store = feeder, written whole THROUGH THE PARENT
        cg::batched_gemm(1.0, a_list, b_list, 1.0, c_list); // += 1 through the views; must run last
    }
    graph.execute();

    double worst = 0.0;
    for (size_t r = 0; r < m; ++r) {
        for (size_t c = 0; c < n * count; ++c) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(r), static_cast<ptrdiff_t>(c)};
            worst = std::max(worst, std::abs(store(idx) - 7.0));
        }
    }
    return worst;
}

} // namespace

TEST_CASE("batched_gemm through views, batch of 16 (control)", "[ComputeGraph][Alias]") {
    CHECK(run_case(/*count=*/16, /*companions=*/0) == 0.0);
    CHECK(run_case(/*count=*/16, /*companions=*/8) == 0.0);
}

TEST_CASE("batched_gemm through views, batch of ONE", "[ComputeGraph][Alias]") {
    CHECK(run_case(/*count=*/1, /*companions=*/0) == 0.0);
    CHECK(run_case(/*count=*/1, /*companions=*/8) == 0.0);
}

TEST_CASE("full-cover view read after a delayed parent write", "[ComputeGraph][Alias]") {
    CHECK(run_delayed_producer_case(/*count=*/1) == 0.0);  // the DLPNO singleton-class shape
    CHECK(run_delayed_producer_case(/*count=*/16) == 0.0); // strict-subview control
}
