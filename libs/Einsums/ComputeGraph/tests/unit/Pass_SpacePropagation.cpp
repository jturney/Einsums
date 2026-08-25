//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_SpacePropagation.cpp
/// @brief Unit tests for the SpacePropagation pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The three spaces every case below draws from, registered in a registry the case owns so no
/// two tests can see each other's declarations.
struct Spaces {
    cg::SpaceRegistry registry;
    cg::SpaceId       occ;
    cg::SpaceId       virt;
    cg::SpaceId       aux;

    Spaces() {
        occ  = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .typical_extent = 4.0});
        virt = registry.register_space(cg::IndexSpace{.name = "virt", .scale_symbol = "v", .typical_extent = 8.0});
        aux  = registry.register_space(cg::IndexSpace{.name = "aux", .scale_symbol = "x", .typical_extent = 16.0});
    }
};

/// The spaces annotated on a registered tensor, by object.
template <typename TensorType>
std::vector<cg::SpaceId> spaces_of(cg::Graph const &graph, TensorType const &tensor) {
    auto const id = graph.find_tensor_id_by_ptr(&tensor);
    REQUIRE(id != 0);
    return graph.tensor_spaces(id);
}

/// Whether a registered tensor's annotation was inferred rather than declared.
template <typename TensorType>
bool inferred_flag(cg::Graph const &graph, TensorType const &tensor) {
    auto const id = graph.find_tensor_id_by_ptr(&tensor);
    REQUIRE(id != 0);
    return graph.tensor(id).spaces_inferred;
}

} // namespace

// Every case below annotates AFTER capture on purpose. Capture already derives the spaces of an
// intermediate it writes for the first time, so annotating first would leave the pass nothing to
// do; annotating afterwards is both the case the pass exists for (a declaration that arrives late,
// or a graph a pass rebuilt) and the only way to see the pass work in isolation.

TEST_CASE("SpacePropagation - a contraction chain resolves in one sweep", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8);  // occ x virt
    auto B = create_random_tensor<double>("B", 8, 16); // virt x aux
    auto E = create_random_tensor<double>("E", 16, 4); // aux x occ

    cg::Graph graph("chain");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 4, 16);
    auto &D = graph.create_zero_tensor<double, 2>("D", 4, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
        cg::einsum("ij <- ix ; xj", &D, C, E);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});
    graph.annotate_spaces(E, {spaces.aux, spaces.occ});

    auto [_m, pass] = graph.apply<cg::passes::SpacePropagation>();

    // Topological order makes one sweep a fixpoint: C is annotated before D is examined.
    CHECK(pass.num_inferred() == 2);
    CHECK(spaces_of(graph, C) == std::vector<cg::SpaceId>{spaces.occ, spaces.aux});
    CHECK(spaces_of(graph, D) == std::vector<cg::SpaceId>{spaces.occ, spaces.occ});
    CHECK(inferred_flag(graph, C));
    CHECK(inferred_flag(graph, D));

    // The inputs are untouched: propagation flows producer to output, never back onto an operand.
    CHECK(spaces_of(graph, A) == std::vector<cg::SpaceId>{spaces.occ, spaces.virt});
    CHECK_FALSE(inferred_flag(graph, A));
}

TEST_CASE("SpacePropagation - a permute reorders the spaces with the axes", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8); // occ x virt

    cg::Graph graph("permute");
    graph.set_space_registry(spaces.registry);
    auto &T = graph.create_zero_tensor<double, 2>("T", 8, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", 0.0, &T, 1.0, A);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});

    auto [_m, pass] = graph.apply<cg::passes::SpacePropagation>();

    CHECK(pass.num_inferred() == 1);
    CHECK(spaces_of(graph, T) == std::vector<cg::SpaceId>{spaces.virt, spaces.occ});
    CHECK(inferred_flag(graph, T));
}

TEST_CASE("SpacePropagation - a declaration is never overwritten", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 16);

    cg::Graph graph("declared");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 4, 16);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});
    // The user says C's axes are something else entirely. The letters would infer (occ, aux).
    graph.annotate_spaces(C, {spaces.aux, spaces.occ});

    auto [_m, pass] = graph.apply<cg::passes::SpacePropagation>();

    CHECK(pass.num_inferred() == 0);
    CHECK(spaces_of(graph, C) == std::vector<cg::SpaceId>{spaces.aux, spaces.occ});
    CHECK_FALSE(inferred_flag(graph, C));
}

TEST_CASE("SpacePropagation - a multi-writer intermediate is left alone", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 16);
    auto F = create_random_tensor<double>("F", 4, 16);

    cg::Graph graph("multiwriter");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 4, 16);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
        cg::axpy(1.0, F, &C); // second writer: nothing about C's slots is settled by one node
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    auto [_m, pass] = graph.apply<cg::passes::SpacePropagation>();

    CHECK(pass.num_inferred() == 0);
    CHECK(spaces_of(graph, C).empty());
}

TEST_CASE("SpacePropagation - a user-owned output is never annotated", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 16);
    auto C = create_zero_tensor<double>("C", 4, 16); // user-owned, not graph.create_*

    cg::Graph graph("user_owned");
    graph.set_space_registry(spaces.registry);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    auto [_m, pass] = graph.apply<cg::passes::SpacePropagation>();

    CHECK(pass.num_inferred() == 0);
    CHECK(spaces_of(graph, C).empty());
}

TEST_CASE("SpacePropagation - disagreeing linear-combination inputs are counted, not raised", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto X = create_random_tensor<double>("X", 4, 4);

    cg::Graph graph("axpby_conflict");
    graph.set_space_registry(spaces.registry);
    auto &Y = graph.create_zero_tensor<double, 2>("Y", 4, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(1.0, X, &Y);
    }

    // X and Y index different things, which is a cross-space bug in the source program. The pass
    // declines it and says so; diagnosing it belongs to the validation pass.
    graph.annotate_spaces(X, {spaces.occ, spaces.virt});
    graph.annotate_spaces(Y, {spaces.virt, spaces.occ});

    auto [_m, pass] = graph.apply<cg::passes::SpacePropagation>();

    CHECK(pass.num_inferred() == 0);
    CHECK(spaces_of(graph, Y) == std::vector<cg::SpaceId>{spaces.virt, spaces.occ});

    auto const skips = pass.skip_reasons();
    REQUIRE(skips.size() == 1);
    CHECK(skips[0].first == "inputs of a linear combination disagree about their spaces");
    CHECK(skips[0].second == 1);
}

TEST_CASE("SpacePropagation - a second run infers nothing new", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 16);

    cg::Graph graph("idempotent");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 4, 16);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    auto [_m1, first] = graph.apply<cg::passes::SpacePropagation>();
    CHECK(first.num_inferred() == 1);

    auto [_m2, second] = graph.apply<cg::passes::SpacePropagation>();
    CHECK(second.num_inferred() == 0); // already says exactly this
    CHECK(spaces_of(graph, C) == std::vector<cg::SpaceId>{spaces.occ, spaces.aux});
    CHECK(inferred_flag(graph, C));
}

TEST_CASE("SpacePropagation - an unannotated program is unchanged", "[ComputeGraph][Passes][Spaces]") {
    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 4);

    cg::Graph graph("unannotated");
    auto     &C = graph.create_zero_tensor<double, 2>("C", 4, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ia ; aj", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::SpacePropagation>();

    CHECK_FALSE(modified); // an analysis pass never claims a modification
    CHECK(pass.num_inferred() == 0);
    CHECK(pass.skip_reasons().empty());
    CHECK(spaces_of(graph, C).empty());
    CHECK_FALSE(inferred_flag(graph, C));

    graph.execute();

    // The contraction summed by hand, so "no behavior change" is checked against something that
    // shares no code with the path under test.
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            double expected = 0.0;
            for (int link = 0; link < 8; ++link) {
                expected += A(row, link) * B(link, col);
            }
            CHECK_THAT(C(row, col), Catch::Matchers::WithinAbs(expected, 1.0e-12));
        }
    }
}

TEST_CASE("SpacePropagation - a loop body intermediate is annotated too", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 16);

    cg::Graph graph("loop_body");
    graph.set_space_registry(spaces.registry);
    auto &body = graph.add_loop("iter", 1, [](size_t) { return false; });
    body.set_space_registry(spaces.registry);
    auto &C = body.create_zero_tensor<double, 2>("C", 4, 16);

    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    body.annotate_spaces(A, {spaces.occ, spaces.virt});
    body.annotate_spaces(B, {spaces.virt, spaces.aux});

    cg::PassManager pm;
    pm.add<cg::passes::SpacePropagation>();
    pm.run(graph);

    CHECK(spaces_of(body, C) == std::vector<cg::SpaceId>{spaces.occ, spaces.aux});
}
