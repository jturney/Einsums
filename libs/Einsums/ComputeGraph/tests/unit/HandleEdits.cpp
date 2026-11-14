//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The changes a pass may make to a registered tensor's handle, one Graph method each. Each method
// owns rules a pass used to write out for itself; these cases pin those rules where they now live.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cstddef>
#include <optional>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Two spaces, registered in a registry the graph is told to use.
struct Spaces {
    cg::SpaceRegistry registry;
    cg::SpaceId       occ;
    cg::SpaceId       virt;
    Spaces() {
        occ  = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .typical_extent = 4.0});
        virt = registry.register_space(cg::IndexSpace{.name = "virt", .scale_symbol = "v", .typical_extent = 4.0});
    }
};

cg::TensorId id_of(cg::Graph const &graph, void const *tensor) {
    cg::TensorId const id = graph.find_tensor_id_by_ptr(tensor);
    REQUIRE(id != 0);
    return id;
}

} // namespace

TEST_CASE("Handle edits - infer_spaces records an inference on an intermediate", "[ComputeGraph][HandleEdits]") {
    Spaces    spaces;
    cg::Graph graph("infer_spaces");
    graph.set_space_registry(spaces.registry);
    auto      &tmp = graph.scratch<double, 2>("tmp", 3, 4);
    auto const id  = id_of(graph, &tmp);

    CHECK(graph.infer_spaces(id, {spaces.occ, spaces.virt}));
    CHECK(graph.tensor_spaces(id) == std::vector<cg::SpaceId>{spaces.occ, spaces.virt});
    CHECK(graph.tensor(id).spaces_inferred);

    // A re-run that would write the same thing reports nothing new.
    CHECK_FALSE(graph.infer_spaces(id, {spaces.occ, spaces.virt}));
    // A later inference may refine an earlier one.
    CHECK(graph.infer_spaces(id, {spaces.virt, spaces.virt}));
}

TEST_CASE("Handle edits - infer_spaces never overrides what it may not", "[ComputeGraph][HandleEdits]") {
    Spaces    spaces;
    cg::Graph graph("infer_spaces_refusals");
    graph.set_space_registry(spaces.registry);

    SECTION("a tensor the caller owns") {
        auto      &result = graph.declare_tensor<double, 2>("result", 3, 4);
        auto const id     = id_of(graph, &result);
        CHECK_FALSE(graph.infer_spaces(id, {spaces.occ, spaces.virt}));
        CHECK(graph.tensor_spaces(id).empty());
    }
    SECTION("an annotation that is not one space per axis") {
        auto      &tmp = graph.scratch<double, 2>("tmp", 3, 4);
        auto const id  = id_of(graph, &tmp);
        CHECK_FALSE(graph.infer_spaces(id, {spaces.occ}));
        CHECK(graph.tensor_spaces(id).empty());
    }
    SECTION("a declared annotation") {
        auto      &tmp = graph.scratch<double, 2>("tmp", 3, 4);
        auto const id  = id_of(graph, &tmp);
        graph.annotate_spaces(id, {spaces.occ, spaces.virt});
        CHECK_FALSE(graph.infer_spaces(id, {spaces.virt, spaces.occ}));
        CHECK(graph.tensor_spaces(id) == std::vector<cg::SpaceId>{spaces.occ, spaces.virt});
        CHECK_FALSE(graph.tensor(id).spaces_inferred);
    }
}

TEST_CASE("Handle edits - infer_symmetry reaches the tensor only when asked and allowed", "[ComputeGraph][HandleEdits]") {
    cg::Graph                graph("infer_symmetry");
    SymmetryDescriptor const swap{{SymmetryOp::swap(0, 1)}};

    SECTION("graph-only on a tensor the caller owns annotates the handle and leaves the tensor alone") {
        auto      &result = graph.declare_tensor<double, 2>("result", 3, 3);
        auto const id     = id_of(graph, &result);
        CHECK(graph.infer_symmetry(id, swap, cg::SymmetryReach::GraphOnly));
        REQUIRE(graph.tensor(id).symmetry_hint != nullptr);
        CHECK(*graph.tensor(id).symmetry_hint == swap);
        CHECK(result.symmetry() == nullptr);
        // The same descriptor again is no news.
        CHECK_FALSE(graph.infer_symmetry(id, swap, cg::SymmetryReach::GraphOnly));
    }
    SECTION("tensor reach on a tensor the caller owns is refused") {
        auto      &result = graph.declare_tensor<double, 2>("result", 3, 3);
        auto const id     = id_of(graph, &result);
        CHECK_FALSE(graph.infer_symmetry(id, swap, cg::SymmetryReach::Tensor));
        CHECK(graph.tensor(id).symmetry_hint == nullptr);
        CHECK(result.symmetry() == nullptr);
    }
    SECTION("tensor reach on an intermediate gives the tensor the descriptor too") {
        auto      &tmp = graph.scratch<double, 2>("tmp", 3, 3);
        auto const id  = id_of(graph, &tmp);
        CHECK(graph.infer_symmetry(id, swap, cg::SymmetryReach::Tensor));
        REQUIRE(tmp.symmetry() != nullptr);
        CHECK(*tmp.symmetry() == swap);
    }
}

// The tensor, the handle and the slot each hold the geometry, and a re-layout that missed one left
// them disagreeing; LayoutAssignment used to update all four copies by hand.
TEST_CASE("Handle edits - relayout moves every copy of the geometry together", "[ComputeGraph][HandleEdits]") {
    Spaces    spaces;
    cg::Graph graph("relayout");
    graph.set_space_registry(spaces.registry);
    auto      &tmp = graph.scratch<double, 3>("tmp", 2, 3, 5);
    auto const id  = id_of(graph, &tmp);
    graph.annotate_spaces(id, {spaces.occ, spaces.virt, spaces.occ});

    std::vector<std::size_t> const order{2, 0, 1};
    graph.relayout(id, order);

    auto const &handle = graph.tensor(id);
    CHECK(handle.dims == std::vector<std::size_t>{5, 2, 3});
    CHECK(handle.strides == std::vector<std::size_t>{1, 5, 10});
    CHECK(handle.spaces == std::vector<cg::SpaceId>{spaces.occ, spaces.occ, spaces.virt});
    CHECK(tmp.dim(0) == 5);
    CHECK(tmp.dim(1) == 2);
    CHECK(tmp.dim(2) == 3);
    if (cg::TensorSlot const *slot = graph.find_slot(id); slot != nullptr) {
        CHECK(slot->dims == handle.dims);
    }
}

TEST_CASE("Handle edits - relayout refuses what is not a re-declaration", "[ComputeGraph][HandleEdits]") {
    cg::Graph graph("relayout_refusals");

    SECTION("an order that is not a permutation of the axes") {
        auto      &tmp = graph.scratch<double, 3>("tmp", 2, 3, 5);
        auto const id  = id_of(graph, &tmp);
        CHECK_THROWS_AS(graph.relayout(id, std::vector<std::size_t>{0, 0, 1}), std::logic_error);
        CHECK_THROWS_AS(graph.relayout(id, std::vector<std::size_t>{0, 1}), std::logic_error);
        CHECK(graph.tensor(id).dims == std::vector<std::size_t>{2, 3, 5});
    }
    SECTION("a tensor whose storage is live") {
        auto const A  = create_zero_tensor<double>("A", 2, 3, 5);
        auto const id = graph.register_operand(A);
        CHECK_THROWS_AS(graph.relayout(id, std::vector<std::size_t>{2, 0, 1}), std::logic_error);
    }
}

// A replicated tensor carried the last plan's descriptor forward, and a reader that asks only
// whether there is one (SUMMAExpansion does) took it for distributed.
TEST_CASE("Handle edits - set_distribution keeps the three fields in agreement", "[ComputeGraph][HandleEdits]") {
    cg::Graph  graph("distribution");
    auto      &tmp = graph.scratch<double, 2>("tmp", 4, 4);
    auto const id  = id_of(graph, &tmp);

    comm::DistributionDescriptor desc;
    desc.dim_to_axis = {comm::GridAxis::Row, comm::GridAxis::None};
    graph.set_distribution(id, desc);
    CHECK(graph.tensor(id).is_distributed);
    CHECK_FALSE(graph.tensor(id).is_replicated);
    CHECK(graph.tensor(id).distribution_info != nullptr);

    graph.set_distribution(id, std::nullopt);
    CHECK_FALSE(graph.tensor(id).is_distributed);
    CHECK(graph.tensor(id).is_replicated);
    CHECK(graph.tensor(id).distribution_info == nullptr);
}

TEST_CASE("Handle edits - a stage adopts a declaring scope's lifecycle, or finds it done", "[ComputeGraph][HandleEdits]") {
    cg::Graph  graph("lifecycle");
    auto      &tmp = graph.scratch<double, 2>("tmp", 4, 4);
    auto const id  = id_of(graph, &tmp);

    cg::TensorHandle from;
    from.alloc_state = cg::AllocState::Deferred;
    from.init_kind   = cg::InitKind::Custom;
    void *filled     = nullptr;
    from.custom_init = [&filled](void *object) { filled = object; };
    graph.adopt_lifecycle(id, from);
    CHECK(graph.tensor(id).alloc_state == cg::AllocState::Deferred);
    CHECK(graph.tensor(id).init_kind == cg::InitKind::Custom);
    REQUIRE(graph.tensor(id).custom_init);
    graph.tensor(id).custom_init(graph.tensor(id).live_ptr());
    CHECK(filled == graph.tensor(id).live_ptr());

    graph.mark_materialized(id);
    CHECK(graph.tensor(id).alloc_state == cg::AllocState::Materialized);
    CHECK(graph.tensor(id).init_kind == cg::InitKind::None);
}

// The transfer passes read residency back at their next run, so a re-run plans from where the
// last one left the data rather than inserting the same transfers again.
TEST_CASE("Handle edits - set_residency records where the data is left", "[ComputeGraph][HandleEdits]") {
    cg::Graph  graph("residency");
    auto      &tmp = graph.scratch<double, 2>("tmp", 4, 4);
    auto const id  = id_of(graph, &tmp);
    REQUIRE(graph.tensor(id).residency == cg::Residency::Host);
    graph.set_residency(id, cg::Residency::Device);
    CHECK(graph.tensor(id).residency == cg::Residency::Device);
}
