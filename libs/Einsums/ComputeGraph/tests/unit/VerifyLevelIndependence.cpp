//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file VerifyLevelIndependence.cpp
/// @brief What `Graph::verify_level_independence` must and must not report.
///
/// The checker exists to catch a missing hazard edge, so its cost of being
/// wrong is asymmetric: a missed conflict is a race nothing else reports, and a
/// reported non-conflict makes the checker unusable as a gate. Both directions
/// are pinned here.
///
/// A `View` node is the case that decides usability. It writes the slice
/// handle's dims/strides and nothing else - the parent's elements are never
/// touched - so two of them on one level are independent no matter how their
/// slices overlap. Attributing the parent's DATA to them made every graph that
/// records more than one view of a buffer report a conflict, which is every
/// captured graph in `examples/dlpno`.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The widest level in @p graph's schedule.
size_t widest_level(cg::Graph const &graph) {
    size_t widest = 0;
    for (auto const &level : graph.dependencies().levels) {
        widest = std::max(widest, level.size());
    }
    return widest;
}

} // namespace

TEST_CASE("views of one parent recorded on one level verify clean", "[compute-graph][verify-levels]") {
    // The reproduction, at the shape `examples/dlpno` records: one buffer, a
    // view per pair, all recorded before anything consumes them.
    constexpr size_t rows = 16, chains = 8;

    RuntimeTensor<double> parent("parent", std::vector<size_t>{rows, rows * chains});
    parent.zero();

    cg::Graph graph("views only");
    {
        cg::CaptureGuard const guard(graph);
        for (size_t i = 0; i < chains; ++i) {
            cg::view_runtime(parent, {cg::ViewAxis::full(),
                                      cg::ViewAxis::range(static_cast<std::int64_t>(i * rows), static_cast<std::int64_t>((i + 1) * rows))});
        }
    }
    graph.topological_sort();

    // They are independent, so the schedule puts them together - which is the
    // only way the checker sees them at all.
    REQUIRE(widest_level(graph) == chains);
    REQUIRE_NOTHROW(graph.verify_level_independence());
}

TEST_CASE("rank-reducing views of one parent verify clean", "[compute-graph][verify-levels]") {
    // The reported shape, and the one no box test can help with: dropping an
    // axis leaves a slice of a DIFFERENT rank from its parent, so the region
    // cannot be expressed in the parent's axis space and every access to it
    // reads as the whole buffer. `pno_xform.py::pair_quantities` records one
    // such view per pair off one shared factor - `fits[k][:, p, :]` - which is
    // what aborted first.
    constexpr size_t planes = 6;

    RuntimeTensor<double> parent("parent", std::vector<size_t>{4, planes, 4});
    parent.zero();

    cg::Graph graph("dropped-axis views");
    {
        cg::CaptureGuard const guard(graph);
        for (size_t p = 0; p < planes; ++p) {
            cg::view_runtime(parent, {cg::ViewAxis::full(), cg::ViewAxis::drop(static_cast<std::int64_t>(p)), cg::ViewAxis::full()});
        }
    }
    graph.topological_sort();

    REQUIRE(widest_level(graph) == planes);
    REQUIRE_NOTHROW(graph.verify_level_independence());
}

TEST_CASE("views of one parent whose slices overlap verify clean", "[compute-graph][verify-levels]") {
    // Overlap is irrelevant to a handle bind: two nodes describing the same
    // region still write two different handles. The scan orders these anyway -
    // its box test is on the slice handles, which do overlap - so what this
    // pins is that nothing reports a conflict either way.
    constexpr size_t rows = 16;

    RuntimeTensor<double> parent("parent", std::vector<size_t>{rows, rows});
    parent.zero();

    cg::Graph graph("overlapping views");
    {
        cg::CaptureGuard const guard(graph);
        cg::view_runtime(parent, {cg::ViewAxis::full(), cg::ViewAxis::range(0, 8)});
        cg::view_runtime(parent, {cg::ViewAxis::full(), cg::ViewAxis::range(0, 8)});  // the same region
        cg::view_runtime(parent, {cg::ViewAxis::full(), cg::ViewAxis::range(4, 12)}); // partially overlapping
        cg::view_runtime(parent, {cg::ViewAxis::full(), cg::ViewAxis::full()});       // the whole parent
    }
    graph.topological_sort();

    REQUIRE_NOTHROW(graph.verify_level_independence());
}

TEST_CASE("a view and a consumer of it may not share a level", "[compute-graph][verify-levels]") {
    // The hazard the metadata write still carries. The consumer reads the
    // slice's dims/strides, so it cannot run alongside the node that binds
    // them. The scan orders the two; the checker's job is to notice if it ever
    // stops.
    constexpr size_t rows = 8;

    RuntimeTensor<double> parent("parent", std::vector<size_t>{rows, rows});
    RuntimeTensor<double> out("out", std::vector<size_t>{rows, rows});
    parent.zero();
    out.zero();

    cg::Graph graph("view then read");
    {
        cg::CaptureGuard const guard(graph);
        auto                  &slice = cg::view_runtime(parent, {cg::ViewAxis::full(), cg::ViewAxis::full()});
        cg::axpby(1.0, slice, 0.0, &out);
    }
    graph.topological_sort();

    REQUIRE(widest_level(graph) == 1); // ordered, so no conflict to report
    REQUIRE_NOTHROW(graph.verify_level_independence());
}

namespace {

/// A graph whose hazard scan has been defeated on purpose: two nodes touch the
/// SAME half of one buffer, but the second one's handle carries a box claiming
/// the other half, so the scan proves them disjoint, emits no edge, and the two
/// land in one level. This is the shape of every defect the checker was written
/// for - the scan believing two accesses to one buffer cannot collide - without
/// depending on a defect actually being present.
///
/// @param as_view record both nodes as `View` nodes rather than data writers,
///                which is the case that must NOT be reported.
cg::Graph make_unordered_pair(bool as_view) {
    constexpr size_t rows = 8, half = rows / 2;

    // Leaked into the graph, which outlives this function in the caller.
    auto *parent = new RuntimeTensor<double>("parent", std::vector<size_t>{rows, rows});
    parent->zero();

    cg::Graph graph("unordered pair");
    graph.adopt([parent]() { delete parent; });

    auto *views = new std::vector<RuntimeTensorView<double>>;
    graph.adopt([views]() { delete views; });
    views->push_back((*parent)(AllT{}, Range{0, half}));
    views->push_back((*parent)(AllT{}, Range{0, half}));

    auto               parent_handle = cg::make_handle(*parent, 0);
    cg::TensorId const parent_id     = graph.register_tensor(std::move(parent_handle));

    for (size_t i = 0; i < 2; ++i) {
        auto handle    = cg::make_handle((*views)[i], 0);
        handle.name    = "half";
        handle.aliases = parent_id;
        // The lie, and the only one: the second handle's box names the other
        // half. Both boxes are in the parent's axis space, which is where the
        // scan reads them from (Graph::for_each_hazard_edge).
        std::int64_t const lo  = i == 0 ? 0 : static_cast<std::int64_t>(half);
        handle.alias_box       = {{0, static_cast<std::int64_t>(rows)}, {lo, lo + static_cast<std::int64_t>(half)}};
        cg::TensorId const tid = graph.register_tensor(std::move(handle));

        cg::Node node;
        node.id      = graph.reserve_node_id();
        node.kind    = as_view ? cg::OpKind::View : cg::OpKind::Scale;
        node.label   = as_view ? "view (unordered)" : "scale (unordered)";
        node.outputs = {tid};
        node.execute = []() {};
        if (as_view) {
            cg::ViewDescriptor desc;
            desc.parent_id   = parent_id;
            desc.result_rank = 2;
            node.op_data     = std::move(desc);
        }
        graph.add_node(std::move(node));
    }

    graph.topological_sort();
    return graph;
}

} // namespace

TEST_CASE("an unordered write-write on one level is still reported", "[compute-graph][verify-levels]") {
    cg::Graph graph = make_unordered_pair(false);

    // The scan found no edge between them, which is the defect the checker is
    // for: the storage says they collide whatever the boxes claim.
    REQUIRE(widest_level(graph) == 2);
    REQUIRE_THROWS_AS(graph.verify_level_independence(), std::runtime_error);
}

TEST_CASE("the same unordered nodes as views are not reported", "[compute-graph][verify-levels]") {
    // Identical storage, identical schedule, `View` instead of `Scale`: no
    // conflict, because neither writes an element of the buffer, and the
    // handles they bind are two different objects.
    cg::Graph graph = make_unordered_pair(true);

    REQUIRE(widest_level(graph) == 2);
    REQUIRE_NOTHROW(graph.verify_level_independence());
}
