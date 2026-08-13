//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file AliasBoxDerivation.cpp
/// @brief Which view shapes the hazard scan can prove element-disjoint, and
/// which it must refuse to.
///
/// A view sliced outside a capture reaches the graph as an ordinary operand and
/// is related to its parent by storage containment; what keeps two such views
/// from being serialized against each other is the interval box
/// `Graph::link_alias_storage` derives from the pointer offset and the two
/// stride sets. This file is the map of that derivation: one case per view
/// shape, asserting the SCHEDULE rather than any value, for the reason
/// `AliasOrderSharedScratch.cpp` gives at length - a missing hazard edge is a
/// race that reproduces probabilistically, while the level it belongs to is
/// deterministic.
///
/// Both directions are asserted, and the second is the important one. A box
/// that is too WIDE costs edges; a box that is too NARROW drops a real edge and
/// races. So every shape whose region cannot be recovered has a case here
/// pinning it to the conservative answer, including two layouts a plausible
/// derivation gets wrong: a parent whose own strides make two indices name one
/// element, and a child whose strides do the same.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Per chain: write a view, then read it back into a private output. The parent
/// is written whole by a node first, so every view is linked to it and the box
/// is the only thing that can keep the chains apart.
///
/// @return the schedule's widest level: `views.size()` when the views are
///         proved disjoint, 1 when they are not.
template <typename Parent>
size_t widest_level(Parent &parent, std::vector<RuntimeTensorView<double>> &views) {
    size_t const                       chains = views.size();
    std::vector<RuntimeTensor<double>> sources, outputs;
    sources.reserve(chains);
    outputs.reserve(chains);
    for (auto const &v : views) {
        auto const            shape = v.dims();
        std::vector<size_t>   dims(shape.begin(), shape.end());
        RuntimeTensor<double> s("source", dims);
        s.zero();
        sources.push_back(std::move(s));
        RuntimeTensor<double> o("out", dims);
        o.zero();
        outputs.push_back(std::move(o));
    }

    cg::Graph graph("box derivation");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        for (size_t i = 0; i < chains; ++i) {
            cg::axpby(1.0, sources[i], 0.0, &views[i]);
            cg::axpby(1.0, views[i], 0.0, &outputs[i]);
        }
    }
    graph.topological_sort();
    size_t widest = 0;
    for (auto const &level : graph.dependencies().levels) {
        widest = std::max(widest, level.size());
    }
    return widest;
}

constexpr size_t CHAINS = 6;

} // namespace

// => shapes whose region the per-axis match recovers <= //

TEST_CASE("dropped-axis views at different indices are disjoint", "[ComputeGraph][Alias]") {
    // A dropped axis is a single-index interval in the parent, which the axis
    // match derives from the offset without any special case: the axes the
    // child kept take the parent strides they reuse, and everything else is
    // pinned by what is left of the offset. Every axis position is checked
    // because the peel runs largest stride first and a defect in it would show
    // up at one end.
    constexpr size_t n = 8;

    SECTION("last axis") {
        RuntimeTensor<double>                  parent("parent", std::vector<size_t>{n, n, CHAINS});
        std::vector<RuntimeTensorView<double>> views;
        for (size_t s = 0; s < CHAINS; ++s) {
            views.push_back(parent(AllT{}, AllT{}, static_cast<int>(s)));
        }
        CHECK(widest_level(parent, views) == CHAINS);
    }

    SECTION("middle axis") {
        RuntimeTensor<double>                  parent("parent", std::vector<size_t>{n, CHAINS, n});
        std::vector<RuntimeTensorView<double>> views;
        for (size_t p = 0; p < CHAINS; ++p) {
            views.push_back(parent(AllT{}, static_cast<int>(p), AllT{}));
        }
        CHECK(widest_level(parent, views) == CHAINS);
    }

    SECTION("first axis") {
        RuntimeTensor<double>                  parent("parent", std::vector<size_t>{CHAINS, n});
        std::vector<RuntimeTensorView<double>> views;
        for (size_t r = 0; r < CHAINS; ++r) {
            views.push_back(parent(static_cast<int>(r), AllT{}));
        }
        CHECK(widest_level(parent, views) == CHAINS);
    }

    SECTION("two axes at once") {
        RuntimeTensor<double>                  parent("parent", std::vector<size_t>{4, 4, 3, 2});
        std::vector<RuntimeTensorView<double>> views;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                views.push_back(parent(AllT{}, AllT{}, static_cast<int>(i), static_cast<int>(j)));
            }
        }
        CHECK(widest_level(parent, views) == 6);
    }
}

TEST_CASE("dropped-axis views at the same index conflict", "[ComputeGraph][Alias]") {
    // The control the disjointness cases need. Same slot, same elements, and
    // the chains must be ordered however tight the box is.
    constexpr size_t                       n = 8;
    RuntimeTensor<double>                  parent("parent", std::vector<size_t>{n, n, CHAINS});
    std::vector<RuntimeTensorView<double>> views;
    for (size_t s = 0; s < CHAINS; ++s) {
        views.push_back(parent(AllT{}, AllT{}, 2));
    }
    CHECK(widest_level(parent, views) == 1);
}

TEST_CASE("a dropped axis mixed with ranges is disjoint", "[ComputeGraph][Alias]") {
    // The DLPNO bucketed-store shape: a pair's block is the leading corner of
    // one slot, and the corner is a different size for every pair. Ragged
    // extents are where a box test has the most room to answer wrong, since
    // every box starts at the origin of the two range axes and they differ
    // only in the dropped one.
    constexpr size_t                       n = 8;
    RuntimeTensor<double>                  parent("parent", std::vector<size_t>{n, n, CHAINS});
    std::vector<RuntimeTensorView<double>> views;
    for (size_t s = 0; s < CHAINS; ++s) {
        size_t const e = n - (s % 3);
        views.push_back(parent(Range{0, e}, Range{0, e}, static_cast<int>(s)));
    }
    CHECK(widest_level(parent, views) == CHAINS);
}

// => shapes whose region only the offset-span bound recovers <= //

TEST_CASE("reshaped windows of a flat pool are disjoint", "[ComputeGraph][Alias]") {
    // The case the per-axis match cannot answer and the span bound can. A
    // reshaped window carries axes that are PRODUCTS of the parent's, so there
    // is no stride to match; what is recoverable is the contiguous offset range
    // it occupies, and in a rank-1 parent that range IS the box.
    //
    // The shape is `examples/dlpno/dlpno/lccsd.py::_shared`, which hands out
    // rank-3 windows of a flat scratch buffer, and before the span bound every
    // one of them widened to the whole pool.
    constexpr size_t                       a = 4, b = 3, c = 2;
    constexpr size_t                       flat = a * b * c;
    RuntimeTensor<double>                  pool("pool", std::vector<size_t>{flat * CHAINS});
    std::vector<RuntimeTensorView<double>> views;
    for (size_t s = 0; s < CHAINS; ++s) {
        views.push_back(pool(Range{s * flat, (s + 1) * flat}).reshape_view(std::vector<size_t>{a, b, c}));
    }
    CHECK(widest_level(pool, views) == CHAINS);
}

TEST_CASE("reshaped windows that overlap still conflict", "[ComputeGraph][Alias]") {
    // The control, and the pool's real usage: every hand-out is a PREFIX, so
    // two of them share their first elements whatever their shapes are.
    constexpr size_t                       a = 4, b = 3, c = 2;
    constexpr size_t                       flat = a * b * c;
    RuntimeTensor<double>                  pool("pool", std::vector<size_t>{flat * CHAINS});
    std::vector<RuntimeTensorView<double>> views;
    for (size_t s = 0; s < CHAINS; ++s) {
        views.push_back(pool(Range{0, flat}).reshape_view(std::vector<size_t>{a, b, c}));
    }
    CHECK(widest_level(pool, views) == 1);
}

TEST_CASE("a reshaped window of a matrix is bounded by its columns", "[ComputeGraph][Alias]") {
    // A column block of a column-major matrix is contiguous, so it reshapes,
    // and the offset range it spans is exactly its columns. The bound is exact
    // here; it is only lossy when the range wraps an axis.
    constexpr size_t                       rows = 8, block = 4;
    RuntimeTensor<double>                  parent("parent", std::vector<size_t>{rows, block * CHAINS});
    std::vector<RuntimeTensorView<double>> views;
    for (size_t s = 0; s < CHAINS; ++s) {
        views.push_back(parent(AllT{}, Range{s * block, (s + 1) * block}).reshape_view(std::vector<size_t>{rows * block}));
    }
    CHECK(widest_level(parent, views) == CHAINS);
}

// => shapes that must NOT be boxed <= //

TEST_CASE("an overlapping parent layout is never boxed", "[ComputeGraph][Alias]") {
    /// The adversarial case for the whole mechanism. A parent whose strides
    /// grow more slowly than its extents reaches one element through more than
    /// one index - dims (4, 4) with strides (1, 2) is at offset 4 as both
    /// (0, 2) and (2, 1) - so two boxes that do not intersect in ITS axis space
    /// can still name the same memory.
    ///
    /// The two views below are exactly that: column 1 and column 2 of the
    /// parent, whose boxes [0,4) x [1,2) and [0,4) x [2,3) share no index and
    /// whose elements are offsets 2..5 and 4..7. A derivation that trusts the
    /// axis space here removes a real hazard edge and races.
    ///
    /// So the layout is classified before anything is decoded from it, and a
    /// parent that is not provably injective declines. That check is what this
    /// case asserts, and it fails without it.
    RuntimeTensor<double>     store("store", std::vector<size_t>{16});
    RuntimeTensorView<double> parent(store, std::vector<size_t>{4, 4}, std::vector<size_t>{1, 2}, std::vector<size_t>{0});

    std::vector<RuntimeTensorView<double>> views;
    for (size_t col = 1; col <= 2; ++col) {
        views.push_back(
            RuntimeTensorView<double>(store, std::vector<size_t>{4, 1}, std::vector<size_t>{1, 2}, std::vector<size_t>{2 * col}));
    }
    CHECK(widest_level(parent, views) == 1);
}

TEST_CASE("a gapped parent keeps its matched boxes and refuses spans", "[ComputeGraph][Alias]") {
    // A parent that is itself a strided sub-block is injective but not packed:
    // there are offsets between its elements. The per-axis match still holds
    // there - it only ever lands on the lattice, and an offset that does not is
    // rejected by the remainder - so its slices stay disjoint. The span bound
    // does NOT hold, because it walks a range of offsets rather than the
    // lattice, and it declines.
    constexpr size_t      rows = 4, gap = 8;
    RuntimeTensor<double> store("store", std::vector<size_t>{gap * CHAINS});
    // Four of every eight elements, so the columns are `gap` apart.
    RuntimeTensorView<double> parent(store, std::vector<size_t>{rows, CHAINS}, std::vector<size_t>{1, gap}, std::vector<size_t>{0});

    SECTION("matched slices stay disjoint") {
        std::vector<RuntimeTensorView<double>> views;
        for (size_t s = 0; s < CHAINS; ++s) {
            views.push_back(
                RuntimeTensorView<double>(store, std::vector<size_t>{rows, 1}, std::vector<size_t>{1, gap}, std::vector<size_t>{s * gap}));
        }
        CHECK(widest_level(parent, views) == CHAINS);
    }

    SECTION("a window with no matching stride is conservative") {
        // Disjoint in fact - one pair of elements each, none shared - and the
        // derivation still has to decline, because the offsets between the
        // parent's elements are not its to decode.
        std::vector<RuntimeTensorView<double>> views;
        for (size_t s = 0; s < CHAINS; ++s) {
            views.push_back(
                RuntimeTensorView<double>(store, std::vector<size_t>{2, 1}, std::vector<size_t>{2, 1}, std::vector<size_t>{s * gap}));
        }
        CHECK(widest_level(parent, views) == 1);
    }
}

TEST_CASE("a child whose own elements coincide is bounded by its reach", "[ComputeGraph][Alias]") {
    // The other adversarial direction: a child whose strides repeat, so several
    // of its indices name one element. The span bound reads the OFFSETS it can
    // reach rather than counting elements, which is what makes it right about
    // such a child in both directions - a (4, 4) window with strides (1, 1)
    // reaches offsets 0 through 6, so it is neither the sixteen elements its
    // extents multiply out to nor the four its innermost axis suggests.
    constexpr size_t      side = 4, reach = 2 * (side - 1) + 1;
    RuntimeTensor<double> store("store", std::vector<size_t>{reach * CHAINS});

    SECTION("windows that do not reach each other") {
        std::vector<RuntimeTensorView<double>> views;
        for (size_t s = 0; s < CHAINS; ++s) {
            views.push_back(RuntimeTensorView<double>(store, std::vector<size_t>{side, side}, std::vector<size_t>{1, 1},
                                                      std::vector<size_t>{s * reach}));
        }
        CHECK(widest_level(store, views) == CHAINS);
    }

    SECTION("windows that do") {
        // One element apart, so consecutive windows share all but two.
        std::vector<RuntimeTensorView<double>> views;
        for (size_t s = 0; s < CHAINS; ++s) {
            views.push_back(
                RuntimeTensorView<double>(store, std::vector<size_t>{side, side}, std::vector<size_t>{1, 1}, std::vector<size_t>{s}));
        }
        CHECK(widest_level(store, views) == 1);
    }
}

// => what the boxes cost the scan <= //

TEST_CASE("repeated writes of one region do not accumulate edges", "[ComputeGraph][Alias]") {
    /// A writer that a later write COVERS is retired from the scan the way a
    /// covered reader already was, and this is the case that needs it: `n`
    /// writes of one slice used to keep all `n` writers live and emit an edge
    /// from every one of them to every later access, which is quadratic in a
    /// graph that repeatedly refreshes a scratch buffer.
    ///
    /// The edges dropped are transitively implied - anything reaching the
    /// covered writer reaches the covering one, which already carries an edge
    /// from it - so the SCHEDULE cannot move, and that is asserted alongside
    /// the count. On the DLPNO merged iteration this is 531,197 hazard edges
    /// against 132,063 for the same 179 levels.
    constexpr size_t      n = 24, rows = 4;
    RuntimeTensor<double> parent("parent", std::vector<size_t>{rows, 2 * rows});
    RuntimeTensor<double> source("source", std::vector<size_t>{rows, rows});
    source.zero();
    RuntimeTensorView<double> slice = parent(AllT{}, Range{0, rows});

    cg::Graph graph("repeated writes");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent); // registers the parent, so the slice is boxed
        for (size_t i = 0; i < n; ++i) {
            cg::axpby(1.0, source, 0.0, &slice);
        }
    }
    // The schedule is a chain either way. What changes is its edge count: each
    // write keeps an edge from the write before it and one from the parent's
    // whole-store write, which does NOT retire because a slice does not cover
    // it. That is 2n - 1. Without the retirement every write also takes an edge
    // from every earlier write, which is n (n + 1) / 2 - 1, six times as many
    // at this length and worse at every longer one.
    CHECK(graph.schedule_level_sizes().size() == n + 1);
    CHECK(graph.schedule_edge_count() == 2 * n - 1);
}
