//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file AliasStructural.cpp
/// @brief Alias discovery with no addresses, and its equivalence to the
/// address-derived one.
///
/// `Graph::link_alias_storage` recovers which tensors share storage from
/// registration-time data pointers and strides. A graph read from a file has
/// none: nothing is allocated when the structure is read, and the tensors bound
/// afterwards are not the tensors that were captured. So the relation has to be
/// derivable from `View` nodes, their `ViewDescriptor` axes and the manifest
/// alone, and it has to produce the SAME relation the pointer path produces
/// wherever both are defined.
///
/// That equivalence is the whole point of the file. The history is the argument:
/// the full-cover alias bug (equal-span views never linked) and the 32-hop
/// `resolve_alias` cap (views past the cap silently losing hazard edges) were
/// both subtly incomplete alias relations, and both surfaced as a race or a
/// wrong number rather than as an error. A second derivation of the same
/// relation is exactly the shape that produced them, so it is asserted equal
/// rather than assumed equal.
///
/// Three tiers here, and the Python shard `test_fuzz_alias_equiv_python.py`
/// carries the fourth (schedule equality over the differential fuzz corpus):
///
///  1. **Exact relation.** For every view shape, the map
///     `{tensor -> (alias root, box)}` derived from addresses and the one
///     derived structurally are compared element for element, both taken from
///     the same cleared starting state so neither is handed an answer the other
///     had to compute. This includes the shapes the hazard scan's own View-node
///     path refused before this task - a permuted view and a view of a view.
///  2. **Schedule.** The two derivations produce the same `schedule_edge_count`
///     and `schedule_level_sizes`, which is the property that actually has to
///     hold; asserting the schedule rather than any value is deliberate, for the
///     reason `AliasOrderSharedScratch.cpp` gives at length - a missing hazard
///     edge is a race that reproduces probabilistically, while the level it
///     belongs to is deterministic.
///  3. **Both ways.** Running the pointer derivation on top of a structurally
///     linked graph changes nothing.
///
/// Where structural is legitimately MORE conservative - a manifest declaration
/// carries no region, a runtime-bounded view has no constant box - it is fenced
/// with its own directional case rather than smuggled into the equality tier.
/// An extra hazard edge costs parallelism; a missing one is the race.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// One tensor's place in the alias relation: the ROOT it resolves to and the
/// region it covers there. Both halves matter and they fail differently - a
/// wrong root drops every edge between two accesses to one buffer, a wrong box
/// drops the edges between two regions of it.
using Relation = std::map<cg::TensorId, std::pair<cg::TensorId, std::vector<std::pair<std::int64_t, std::int64_t>>>>;

Relation relation_of(cg::Graph &graph) {
    Relation out;
    for (auto const &[id, handle] : graph.tensors_map()) {
        out.emplace(id, std::pair{graph.resolve_alias(id), handle.alias_box});
    }
    return out;
}

/// Human-readable form, so a mismatch reports which tensor and which box rather
/// than "two maps differ".
std::string describe(Relation const &rel) {
    std::string out;
    for (auto const &[id, entry] : rel) {
        out += fmt::format("  {} -> root {}", id, entry.first);
        if (!entry.second.empty()) {
            out += " box";
            for (auto const &[lo, hi] : entry.second) {
                out += fmt::format(" [{},{})", lo, hi);
            }
        }
        out += '\n';
    }
    return out;
}

/// The relation the ADDRESSES give, from a cleared start.
Relation pointer_relation(cg::Graph &graph) {
    graph.clear_alias_links();
    graph.link_alias_storage();
    return relation_of(graph);
}

/// The relation the STRUCTURE gives, from the same cleared start.
Relation structural_relation(cg::Graph &graph) {
    graph.clear_alias_links();
    graph.link_alias_structural();
    return relation_of(graph);
}

/// Tier 1 and tier 3 together, which is what every shape case below asserts.
///
/// Both derivations start from a cleared relation on purpose. Capture sets
/// ``aliases`` on a ``cg::view`` result itself, so comparing a normally linked
/// graph against a structurally linked one would be comparing the structural
/// answer to a copy of itself; clearing first is what makes the pointer path
/// actually re-derive the relation from the offsets and strides, and so what
/// makes the equality a statement about two independent computations.
void require_derivations_agree(cg::Graph &graph) {
    Relation const pointer    = pointer_relation(graph);
    Relation const structural = structural_relation(graph);
    INFO("pointer-derived:\n" << describe(pointer) << "structural:\n" << describe(structural));
    CHECK(pointer == structural);

    // Both ways: the pointer pass run over a structurally linked graph has
    // nothing left to say. link_alias_structural deliberately leaves the
    // pointer derivation marked stale, so this actually runs.
    graph.link_alias_storage();
    INFO("after the pointer pass ran on top:\n" << describe(relation_of(graph)));
    CHECK(relation_of(graph) == structural);
}

/// Tier 2: the two derivations schedule identically.
void require_schedules_agree(cg::Graph &graph) {
    graph.clear_alias_links();
    graph.link_alias_storage();
    size_t const              pointer_edges  = graph.schedule_edge_count();
    std::vector<size_t> const pointer_levels = graph.schedule_level_sizes();

    graph.clear_alias_links();
    graph.link_alias_structural();
    CHECK(graph.schedule_edge_count() == pointer_edges);
    CHECK(graph.schedule_level_sizes() == pointer_levels);
}

/// Per chain: write a view, then read it back into a private output, with the
/// parent written whole first so every view is linked to it. The schedule's
/// widest level is `views.size()` when the views are proved disjoint and 1 when
/// they are not - the same instrument `AliasBoxDerivation.cpp` uses.
size_t widest_level(cg::Graph &graph) {
    graph.topological_sort();
    size_t widest = 0;
    for (auto const &level : graph.dependencies().levels) {
        widest = std::max(widest, level.size());
    }
    return widest;
}

} // namespace

// => tier 1: every shape the pointer derivation can describe <= //
//
// Most cases here record their views with ``cg::view_runtime``, which is
// historical rather than required: the runtime-rank recorder was for a while the
// only one that offset its capture-time placeholder by the slice's constant
// bounds, so it was the only one whose registration-time geometry really was the
// slice's geometry. The typed ``cg::view`` recorder now does the same, so both
// recorders present the pointer derivation with a region rather than with the
// slice's extents at the parent's address, and the last case in this file asserts
// the two agree on one shape recorded both ways.

TEST_CASE("structural and pointer derivations agree on sliced views", "[ComputeGraph][Alias]") {
    constexpr size_t      n = 8, chains = 4;
    RuntimeTensor<double> parent("parent", std::vector<size_t>{n, n});

    cg::Graph graph("sliced");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        for (size_t s = 0; s < chains; ++s) {
            auto &slice =
                cg::view_runtime(parent, {cg::ViewAxis::range(static_cast<std::int64_t>(2 * s), static_cast<std::int64_t>(2 * s + 2)),
                                          cg::ViewAxis::full()});
            cg::scale(2.0, &slice);
        }
    }
    require_derivations_agree(graph);
    require_schedules_agree(graph);

    graph.clear_alias_links();
    graph.link_alias_structural();
    CHECK(widest_level(graph) == chains);
}

TEST_CASE("structural and pointer derivations agree on dropped axes", "[ComputeGraph][Alias]") {
    // A dropped axis is a single-index interval in the parent. The pointer path
    // recovers it from what is left of the offset after the kept axes take their
    // strides; the structural path reads it off the descriptor. Every axis
    // position is exercised because the pointer peel runs largest stride first
    // and a defect in it shows up at one end.
    constexpr size_t n = 6, chains = 3;

    auto run = [](std::vector<size_t> const &dims, size_t drop_axis) {
        RuntimeTensor<double> parent("parent", dims);
        cg::Graph             graph("dropped");
        {
            cg::CaptureGuard const guard(graph);
            cg::scale(0.0, &parent);
            for (size_t s = 0; s < chains; ++s) {
                std::vector<cg::ViewAxis> axes(dims.size(), cg::ViewAxis::full());
                axes[drop_axis] = cg::ViewAxis::drop(static_cast<std::int64_t>(s));
                auto &slice     = cg::view_runtime(parent, axes);
                cg::scale(2.0, &slice);
            }
        }
        require_derivations_agree(graph);
        require_schedules_agree(graph);
    };

    SECTION("last axis") {
        run({n, n, chains}, 2);
    }
    SECTION("middle axis") {
        run({n, chains, n}, 1);
    }
    SECTION("first axis") {
        run({chains, n}, 0);
    }
}

TEST_CASE("structural and pointer derivations agree on ranges mixed with drops", "[ComputeGraph][Alias]") {
    // The DLPNO bucketed-store shape: a pair's block is the leading corner of
    // one slot and the corner is a different size for every pair. Ragged extents
    // are where a box test has the most room to answer wrong, since every box
    // starts at the origin of the range axes and they differ only in the drop.
    constexpr size_t      n = 8, chains = 4;
    RuntimeTensor<double> parent("parent", std::vector<size_t>{n, n, chains});

    cg::Graph graph("ranges and drops");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        for (size_t s = 0; s < chains; ++s) {
            auto const e     = static_cast<std::int64_t>(n - (s % 3));
            auto      &slice = cg::view_runtime(
                parent, {cg::ViewAxis::range(0, e), cg::ViewAxis::range(0, e), cg::ViewAxis::drop(static_cast<std::int64_t>(s))});
            cg::scale(2.0, &slice);
        }
    }
    require_derivations_agree(graph);
    require_schedules_agree(graph);
}

TEST_CASE("structural and pointer derivations agree on a gapped parent", "[ComputeGraph][Alias]") {
    // A parent that is itself a strided sub-block is injective but not packed:
    // there are offsets between its elements. The pointer path's per-axis match
    // still holds there and its span bound declines; the structural path never
    // looks at an offset at all, so the two have to be checked to still meet.
    constexpr size_t            rows = 4, cols = 6, gap = 8;
    RuntimeTensor<double> const store("store", std::vector<size_t>{gap * cols});
    RuntimeTensorView<double>   parent(store, std::vector<size_t>{rows, cols}, std::vector<size_t>{1, gap}, std::vector<size_t>{0});

    cg::Graph graph("gapped");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        for (size_t s = 0; s < cols; ++s) {
            auto &slice = cg::view_runtime(parent, {cg::ViewAxis::full(), cg::ViewAxis::drop(static_cast<std::int64_t>(s))});
            cg::scale(2.0, &slice);
        }
    }
    require_derivations_agree(graph);
    require_schedules_agree(graph);
}

TEST_CASE("structural and pointer derivations agree on a whole-parent view", "[ComputeGraph][Alias]") {
    // The full-cover case the alias bug was about. A view spanning its whole
    // parent has the SAME byte span, so the pointer path links it by the id
    // tie-break and normalizes its box away; the structural path has to reach
    // the same "root, no box" or the two disagree on the shape that raced.
    constexpr size_t      n = 5;
    RuntimeTensor<double> parent("parent", std::vector<size_t>{n, n});

    cg::Graph graph("whole cover");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        auto &whole = cg::view_runtime(parent, {cg::ViewAxis::full(), cg::ViewAxis::full()});
        cg::scale(2.0, &whole);
    }
    require_derivations_agree(graph);
    require_schedules_agree(graph);

    // Spelled out, because "no box" is the assertion and an empty vector is easy
    // to read as "not computed".
    graph.clear_alias_links();
    graph.link_alias_structural();
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.aliases != 0) {
            CHECK(handle.alias_box.empty());
        }
    }
}

TEST_CASE("structural and pointer derivations agree on a runtime-bounded view", "[ComputeGraph][Alias]") {
    // A Param bound is not known until execute, so the structural path has no
    // constant box and the view conflicts as its whole parent. The pointer path
    // reaches the same answer by a different route (the capture-time placeholder
    // spans the full parent, so the box normalizes away), and that coincidence
    // is worth pinning: it is what keeps a runtime-bounded view out of the
    // conservative-only tier below.
    constexpr size_t      n = 8;
    RuntimeTensor<double> parent("parent", std::vector<size_t>{n, n});

    cg::Graph graph("param bound");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        auto &head = cg::view_runtime(parent, {cg::ViewAxis::range(0, "hi"), cg::ViewAxis::full()});
        cg::scale(2.0, &head);
    }
    require_derivations_agree(graph);
    require_schedules_agree(graph);

    graph.clear_alias_links();
    graph.link_alias_structural();
    for (auto const &[id, handle] : graph.tensors_map()) {
        INFO("tensor " << id);
        CHECK(handle.alias_box.empty()); // no constant bound, so no region
    }
}

// => tier 1b: the shapes the hazard scan's View-node path refused <= //

TEST_CASE("structural derivation composes a permuted view", "[ComputeGraph][Alias]") {
    // The inline View-node scan this replaces bailed on any non-empty
    // permutation, so a transposed slice widened to its whole parent and every
    // access through it serialized against every other. The descriptor says
    // exactly which parent axis each result axis reads, so the box is derivable;
    // what makes it easy to get wrong is that ``axes[i]`` slices parent axis
    // ``permutation[i]``, not parent axis ``i``.
    constexpr size_t      rows = 8, cols = 4, chains = 4;
    RuntimeTensor<double> parent("parent", std::vector<size_t>{rows, cols});

    cg::Graph graph("permuted");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        for (size_t s = 0; s < chains; ++s) {
            // Result axis 0 reads parent axis 1 (all of it); result axis 1 reads
            // parent axis 0, restricted to this chain's two rows.
            auto &slice = cg::view_runtime(
                parent, {cg::ViewAxis::full(), cg::ViewAxis::range(static_cast<std::int64_t>(2 * s), static_cast<std::int64_t>(2 * s + 2))},
                std::vector<size_t>{1, 0});
            cg::scale(2.0, &slice);
        }
    }
    require_derivations_agree(graph);

    // And the box is real: the chains touch different rows of the parent, so
    // after the whole-parent write they are free to run together.
    graph.clear_alias_links();
    graph.link_alias_structural();
    CHECK(widest_level(graph) == chains);
}

TEST_CASE("structural derivation composes a view of a view", "[ComputeGraph][Alias]") {
    // The other refusal: the inline scan required the parent to be its own alias
    // root, so a chained view carried no box at all. Composition places the
    // child's intervals inside the parent's own region and links to the ROOT,
    // which is also what keeps the chain short enough for resolve_alias.
    constexpr size_t      n = 12, chains = 3;
    RuntimeTensor<double> parent("parent", std::vector<size_t>{n, n});

    cg::Graph graph("view of view");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        for (size_t s = 0; s < chains; ++s) {
            auto &outer =
                cg::view_runtime(parent, {cg::ViewAxis::range(static_cast<std::int64_t>(4 * s), static_cast<std::int64_t>(4 * s + 4)),
                                          cg::ViewAxis::full()});
            auto &inner = cg::view_runtime(outer, {cg::ViewAxis::range(1, 3), cg::ViewAxis::full()});
            cg::scale(2.0, &inner);
        }
    }
    require_derivations_agree(graph);

    graph.clear_alias_links();
    graph.link_alias_structural();
    // Every chained view resolves to the parent in ONE hop, not to the view
    // above it: path compression, which is what the 32-hop cap bug was about.
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.aliases != 0) {
            CHECK(graph.resolve_alias(handle.aliases) == handle.aliases);
        }
    }
    CHECK(widest_level(graph) == chains);
}

TEST_CASE("structural derivation composes a view of a permuted view", "[ComputeGraph][Alias]") {
    // Both generalizations at once, which is where an axis map that is merely
    // "identity or nothing" stops being enough: the inner view's intervals live
    // in the transposed axis order and have to be mapped back through the outer
    // view's permutation before they mean anything in the parent.
    constexpr size_t      rows = 8, cols = 6, chains = 3;
    RuntimeTensor<double> parent("parent", std::vector<size_t>{rows, cols});

    cg::Graph graph("view of permuted view");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        // ONE transposed view, sliced `chains` ways. A whole-cover view carries
        // no box by construction (it is the parent), so recording one per chain
        // would put `chains` whole-parent metadata writes in the graph and
        // serialize everything however precisely the inner slices are described.
        auto &transposed = cg::view_runtime(parent, {cg::ViewAxis::full(), cg::ViewAxis::full()}, std::vector<size_t>{1, 0});
        for (size_t s = 0; s < chains; ++s) {
            auto &inner =
                cg::view_runtime(transposed, {cg::ViewAxis::range(static_cast<std::int64_t>(2 * s), static_cast<std::int64_t>(2 * s + 2)),
                                              cg::ViewAxis::full()});
            cg::scale(2.0, &inner);
        }
    }
    require_derivations_agree(graph);

    graph.clear_alias_links();
    graph.link_alias_structural();
    CHECK(widest_level(graph) == chains);
}

TEST_CASE("chained views that overlap still conflict", "[ComputeGraph][Alias]") {
    // The control every disjointness case needs. Same region reached through
    // different chains: the boxes coincide and the chains must be ordered
    // however precisely they are described.
    constexpr size_t      n = 12, chains = 3;
    RuntimeTensor<double> parent("parent", std::vector<size_t>{n, n});

    cg::Graph graph("chained overlap");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        for (size_t s = 0; s < chains; ++s) {
            auto &outer = cg::view_runtime(parent, {cg::ViewAxis::range(0, 4), cg::ViewAxis::full()});
            auto &inner = cg::view_runtime(outer, {cg::ViewAxis::range(1, 3), cg::ViewAxis::full()});
            cg::scale(2.0, &inner);
        }
    }
    require_derivations_agree(graph);

    graph.clear_alias_links();
    graph.link_alias_structural();
    CHECK(widest_level(graph) == 1);
}

TEST_CASE("a non-injective parent is never boxed structurally", "[ComputeGraph][Alias]") {
    // The adversarial case, and the one place the structural derivation could be
    // LESS safe than the pointer one rather than more conservative. A parent
    // whose strides grow more slowly than its extents reaches one element
    // through more than one index - dims (4, 4) with strides (1, 2) is at offset
    // 4 as both (0, 2) and (2, 1) - so two boxes that share no index can still
    // name the same memory. Reading the boxes off the descriptor and trusting
    // the axis space there would DROP a real hazard edge and race.
    //
    // So the root's layout is classified before anything is placed in its axis
    // space, and a parent that is not provably injective declines. Columns 1 and
    // 2 below are exactly the trap: boxes [0,4)x[1,2) and [0,4)x[2,3), elements
    // at offsets 2..5 and 4..7.
    RuntimeTensor<double> const store("store", std::vector<size_t>{16});
    RuntimeTensorView<double>   parent(store, std::vector<size_t>{4, 4}, std::vector<size_t>{1, 2}, std::vector<size_t>{0});

    cg::Graph graph("non-injective");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        for (std::int64_t col = 1; col <= 2; ++col) {
            auto &slice = cg::view_runtime(parent, {cg::ViewAxis::full(), cg::ViewAxis::drop(col)});
            cg::scale(2.0, &slice);
        }
    }
    require_derivations_agree(graph);

    graph.clear_alias_links();
    graph.link_alias_structural();
    for (auto const &[id, handle] : graph.tensors_map()) {
        INFO("tensor " << id);
        CHECK(handle.alias_box.empty());
    }
    CHECK(widest_level(graph) == 1);
}

// => where structural is legitimately more conservative <= //

TEST_CASE("a declared alias conflicts as the whole parent", "[ComputeGraph][Alias]") {
    // A manifest declaration names a BUFFER, never a region: the schema carries
    // no box, and the pair it describes has no View node to read one from. So a
    // declared alias is coarser than the same relation derived from addresses,
    // and it is fenced here rather than in the equality tier.
    //
    // The direction of the inequality is the whole point. An extra hazard edge
    // costs parallelism; a missing one is a data race that reproduces
    // probabilistically. Coarser is the only safe way to be wrong.
    // COLUMN blocks, not row blocks: the tensors are column-major, so two row
    // slices interleave in memory and their byte SPANS overlap however disjoint
    // their elements are. That is a real property of the pointer derivation
    // (partial overlap with no common container is linked conservatively) and it
    // would mask the point of this case.
    constexpr size_t n      = 8;
    auto             parent = create_zero_tensor<double>("parent", n, n);
    auto             left   = parent(AllT{}, Range{0, 4});
    auto             right  = parent(AllT{}, Range{4, 8});
    auto             source = create_zero_tensor<double>("source", n, 4);

    cg::Graph graph("declared coarser");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        cg::axpby(1.0, source, 0.0, &left);
        cg::axpby(1.0, source, 0.0, &right);
    }

    // Addresses: two provably disjoint halves, so the two writes share a level
    // behind the whole-parent write.
    graph.clear_alias_links();
    graph.link_alias_storage();
    size_t const pointer_edges = graph.schedule_edge_count();
    CHECK(widest_level(graph) == 2);

    // Both halves carry the parent's name-derived spelling, so identify them by
    // the link the pointer pass just made rather than by name.
    cg::TensorId              parent_id = 0;
    std::vector<cg::TensorId> halves;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.aliases != 0) {
            halves.push_back(id);
            parent_id = handle.aliases;
        }
    }
    REQUIRE(parent_id != 0);
    REQUIRE(halves.size() == 2);

    // Declarations only, which is all a loaded graph would carry for this pair:
    // same buffer, no region, so the two writes serialize.
    graph.clear_alias_links();
    for (cg::TensorId const half : halves) {
        graph.declare_alias(half, parent_id);
    }
    graph.link_alias_structural();
    for (cg::TensorId const half : halves) {
        CHECK(graph.resolve_alias(half) == parent_id);
        CHECK(graph.find_tensor(half)->alias_box.empty());
    }
    // Structural is never allowed to emit FEWER edges than the pointer path.
    CHECK(graph.schedule_edge_count() >= pointer_edges);
    CHECK(widest_level(graph) == 1);
}

// => tier 4: the loaded-graph state <= //

TEST_CASE("a graph with no addresses links structurally", "[ComputeGraph][Alias]") {
    // The loader's state, simulated the only way it can be until the loader
    // exists: capture normally, then strip every address. The point of the case
    // is what link_alias_storage does when it finds none - linking NOTHING is
    // not a safe default, it is the exact shape of the full-cover alias bug,
    // a silently incomplete relation with no error anywhere.
    //
    // The captured graph is its own twin: same node list, same tensor ids, so
    // the relation recovered from the stripped state is comparable element for
    // element against the one recovered from the addresses.
    constexpr size_t      n = 12, chains = 3;
    RuntimeTensor<double> parent("parent", std::vector<size_t>{n, n});

    cg::Graph graph("loaded");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        for (size_t s = 0; s < chains; ++s) {
            auto &outer =
                cg::view_runtime(parent, {cg::ViewAxis::range(static_cast<std::int64_t>(4 * s), static_cast<std::int64_t>(4 * s + 4)),
                                          cg::ViewAxis::full()});
            auto &inner = cg::view_runtime(outer, {cg::ViewAxis::range(1, 3), cg::ViewAxis::full()});
            cg::scale(2.0, &inner);
        }
    }

    Relation const captured = pointer_relation(graph);
    size_t const   edges    = graph.schedule_edge_count();

    // Strip every address, which is what a graph read from a file has.
    for (auto const &[id, handle] : graph.tensors_map()) {
        graph.find_tensor(id)->data_ptr = nullptr;
    }
    graph.clear_alias_links();
    graph.link_alias_storage(); // must DELEGATE, not link nothing

    INFO("captured:\n" << describe(captured) << "loaded:\n" << describe(relation_of(graph)));
    CHECK(relation_of(graph) == captured);
    CHECK(graph.schedule_edge_count() == edges);
}

TEST_CASE("a graph with no addresses and no View nodes still links its declarations", "[ComputeGraph][Alias]") {
    // The complement: no addresses AND no View node, which is the only state in
    // which a declaration is the sole evidence an alias exists at all. Without
    // it the two writes look independent and a threading executor runs them
    // concurrently over one buffer.
    constexpr size_t n      = 6;
    auto             parent = create_zero_tensor<double>("parent", n, n);
    auto             slice  = parent(Range{0, 3}, AllT{});
    auto             source = create_zero_tensor<double>("source", 3, n);

    cg::Graph graph("declared only");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        cg::axpby(1.0, source, 0.0, &slice);
    }

    graph.clear_alias_links();
    graph.link_alias_storage();
    cg::TensorId parent_id = 0, slice_id = 0;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.aliases != 0) {
            slice_id  = id;
            parent_id = handle.aliases;
        }
    }
    REQUIRE(parent_id != 0);
    REQUIRE(slice_id != 0);

    for (auto const &[id, handle] : graph.tensors_map()) {
        graph.find_tensor(id)->data_ptr = nullptr;
    }

    SECTION("undeclared, the relation is genuinely lost") {
        // Stated so the next case is not read as a tautology: without a
        // declaration there is nothing left to recover the pair from, which is
        // precisely why the manifest has to carry one.
        graph.clear_alias_links();
        graph.link_alias_storage();
        CHECK(graph.resolve_alias(slice_id) == slice_id);
    }

    SECTION("declared, it survives with no address consulted") {
        graph.clear_alias_links();
        graph.declare_alias(slice_id, parent_id); // what a loader installs from the file
        graph.link_alias_storage();
        CHECK(graph.resolve_alias(slice_id) == parent_id);
        CHECK(graph.find_tensor(slice_id)->alias_box.empty()); // buffer, not region
        CHECK(widest_level(graph) == 1);
    }
}

TEST_CASE("a declaration survives clearing the derived links", "[ComputeGraph][Alias]") {
    // Declarations are an INPUT to alias discovery, not an output of it, so a
    // relink from a cleared state has to reapply them. Losing one silently is
    // the same failure as never having had it.
    auto A = create_zero_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("declaration persists");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    cg::TensorId a_id = 0, b_id = 0;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.name == "A") {
            a_id = id;
        } else if (handle.name == "B") {
            b_id = id;
        }
    }
    REQUIRE(a_id != 0);
    REQUIRE(b_id != 0);

    graph.declare_alias(b_id, a_id);
    CHECK(graph.resolve_alias(b_id) == a_id);
    graph.clear_alias_links();
    CHECK(graph.resolve_alias(b_id) == b_id); // cleared, as asked
    graph.link_alias_storage();
    CHECK(graph.resolve_alias(b_id) == a_id); // and reapplied by the next pass

    // Containment is not symmetric, and the reverse declaration would make every
    // later resolve_alias throw about a corrupt link rather than about the
    // declaration that corrupted it. Refuse it where both names are still in hand.
    REQUIRE_THROWS_WITH(graph.declare_alias(a_id, b_id), Catch::Matchers::ContainsSubstring("each other's alias parent"));
}

// => bind-time enforcement <= //

TEST_CASE("bind rejects an undeclared span overlap", "[ComputeGraph][Alias][Bind]") {
    // Overlapping slices of one buffer, bound to two slots the manifest relates
    // in no way. Their base addresses DIFFER, so the identical-pointer check
    // this replaces accepted the bind and the hazard edges between the two slots
    // went missing - the full-cover alias bug's failure shape arriving through
    // the interface instead of through capture.
    auto A = create_zero_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("bind overlap");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto buffer = create_zero_tensor<double>("buffer", 4, 6);
    auto left   = buffer(AllT{}, Range{0, 4});
    auto right  = buffer(AllT{}, Range{2, 6});
    REQUIRE(static_cast<void const *>(left.data()) != static_cast<void const *>(right.data()));

    graph.bind("A", left);
    REQUIRE_THROWS_WITH(graph.bind("B", right), Catch::Matchers::ContainsSubstring("overlaps the storage") &&
                                                    Catch::Matchers::ContainsSubstring("declares no alias"));
}

TEST_CASE("bind accepts disjoint slices of one buffer", "[ComputeGraph][Alias][Bind]") {
    // The control. Two windows of one allocation that share no byte are not an
    // aliasing bind and must not be reported as one; a span test says so and a
    // "same allocation" test would not.
    auto A = create_zero_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("bind disjoint");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto buffer = create_zero_tensor<double>("buffer", 4, 8);
    auto left   = buffer(AllT{}, Range{0, 4});
    auto right  = buffer(AllT{}, Range{4, 8});
    REQUIRE_NOTHROW(graph.bind("A", left));
    REQUIRE_NOTHROW(graph.bind("B", right));
}

TEST_CASE("a declared aliasing bind installs the hazard link", "[ComputeGraph][Alias][Bind]") {
    // Accepting a declared aliasing bind and doing nothing else would be the
    // worst of both: the manifest says the two slots share storage and the
    // scheduler still orders nothing between them. The declaration is therefore
    // installed as a link, and the schedule is what proves it.
    constexpr size_t n      = 6;
    auto             parent = create_zero_tensor<double>("parent", n, n);
    auto             slice  = parent(Range{0, 3}, AllT{});
    auto             source = create_zero_tensor<double>("source", 3, n);

    cg::Graph graph("declared bind");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        cg::axpby(1.0, source, 0.0, &slice);
    }

    auto const contract = graph.manifest();
    // Both ends are caller-supplied, so the relation between them is declarable
    // and the manifest carries it.
    cg::ManifestEntry declared;
    for (auto const &entry : contract.entries()) {
        if (entry.aliases_input != 0) {
            declared = entry;
        }
    }
    REQUIRE(declared.aliases_input != 0);
    std::string const slice_name  = declared.name;
    auto const       *parent_slot = contract.find_by_id(declared.aliases_input);
    REQUIRE(parent_slot != nullptr);
    std::string const parent_name = parent_slot->name;

    // Strip the derived relation, then bind fresh overlapping storage. The link
    // has to come back from the declaration alone.
    graph.clear_alias_links();
    auto parent2 = create_zero_tensor<double>("parent2", n, n);
    auto slice2  = parent2(Range{0, 3}, AllT{});
    REQUIRE_NOTHROW(graph.bind(parent_name, parent2));
    REQUIRE_NOTHROW(graph.bind(slice_name, slice2));

    CHECK(graph.resolve_alias(declared.id) == declared.aliases_input);
    // The whole-parent write and the slice write are ordered against each other,
    // so they cannot share a level.
    CHECK(widest_level(graph) == 1);
}

// => the typed recorder registers where its slice actually is <= //

TEST_CASE("a typed view registers at its own address, not its parent's", "[ComputeGraph][Alias]") {
    // ``cg::view`` (the compile-time-rank recorder) used to emplace its
    // capture-time placeholder at the PARENT's base pointer while giving it the
    // SLICE's extents, so the geometry on the handle described no real region:
    // every slice of one parent registered at the same address. Nothing consulted
    // it - the hazard scan reads a ``View`` node's box from the DESCRIPTOR and
    // prefers it over anything on the handle - so the gap was harmless and was
    // carried as a pinned observation rather than a bug.
    //
    // It is fixed rather than pinned now, because a region rewrite that reaches a
    // handle would inherit the lie, and "harmless as long as nobody asks" is not a
    // property a new consumer preserves. The recorder applies the slice's constant
    // offset, so this case asserts the placeholders are DISTINCT and that both
    // derivations agree on a shape the two recorders record identically.
    constexpr size_t n = 8, chains = 4;
    auto             parent = create_zero_tensor<double>("parent", n, n);

    cg::Graph graph("typed views");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        for (size_t s = 0; s < chains; ++s) {
            auto &slice = cg::view<double, 2>(
                parent, cg::ViewAxis::range(static_cast<std::int64_t>(2 * s), static_cast<std::int64_t>(2 * s + 2)), cg::ViewAxis::full());
            cg::scale(2.0, &slice);
        }
    }
    CHECK(widest_level(graph) == chains);

    // Four slices, four addresses. Column-major storage puts row-slice s at the
    // parent's base + 2*s elements, which is the offset the recorder applies.
    std::set<void const *> addresses;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.aliases != 0) {
            addresses.insert(handle.data_ptr);
        }
    }
    CHECK(addresses.size() == chains);
    CHECK(addresses.count(static_cast<void const *>(parent.data())) == 1); // slice 0 begins at the base

    // And the two derivations agree, which they could not while every slice
    // registered on top of every other.
    require_derivations_agree(graph);
    require_schedules_agree(graph);
}
