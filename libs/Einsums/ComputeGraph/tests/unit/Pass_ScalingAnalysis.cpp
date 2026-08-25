//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_ScalingAnalysis.cpp
/// @brief Unit tests for the ScalingAnalysis pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The spaces every case below draws from, registered in a registry the case owns so no two tests
/// can see each other's declarations. The scale order is what lets the rate-limiting verdict rank
/// two polynomials asymptotically instead of falling through to a numeric rung.
struct Spaces {
    cg::SpaceRegistry registry;
    cg::SpaceId       occ;
    cg::SpaceId       virt;
    cg::SpaceId       aux;

    Spaces() {
        occ  = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .typical_extent = 4.0});
        virt = registry.register_space(cg::IndexSpace{.name = "virt", .scale_symbol = "v", .typical_extent = 8.0});
        aux  = registry.register_space(cg::IndexSpace{.name = "aux", .scale_symbol = "x", .typical_extent = 16.0});
        registry.declare_less(occ, virt);
        registry.declare_less(virt, aux);
    }
};

/// The monomial made of one space variable per id, with the given coefficient.
cg::SymbolicPoly monomial(double coefficient, std::vector<cg::SpaceId> const &spaces) {
    cg::SymbolicPoly poly = cg::SymbolicPoly::constant(coefficient);
    for (auto const id : spaces) {
        poly *= cg::SymbolicPoly::variable(cg::SymbolicVar::space(id));
    }
    return poly;
}

/// Whether a pass's skip tally records the given reason at least once.
bool skipped_for(std::vector<std::pair<std::string, std::size_t>> const &skips, std::string_view reason) {
    return std::ranges::any_of(skips, [reason](auto const &entry) { return entry.first == reason; });
}

/// Whether any line of a report mentions the given text.
bool mentions(std::vector<std::string> const &lines, std::string_view text) {
    return std::ranges::any_of(lines, [text](std::string const &line) { return line.find(text) != std::string::npos; });
}

} // namespace

TEST_CASE("ScalingAnalysis - an annotated contraction reports its exact cost polynomial", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3); // occ x virt
    auto B = create_random_tensor<double>("B", 3, 4); // virt x aux

    cg::Graph graph("exact_cost");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    auto [modified, pass] = graph.apply<cg::passes::ScalingAnalysis>();

    CHECK_FALSE(modified); // an analysis pass never claims a modification
    REQUIRE(pass.num_analyzed() == 1);

    // The loop space is i, a, x, one distinct letter each, times two for the multiply and the add.
    CHECK(pass.total_flops() == monomial(2.0, {spaces.occ, spaces.virt, spaces.aux}));
    CHECK(pass.total_flops().to_string(&spaces.registry) == "2*o*v*x");

    // Traffic is C, then A, then B, each counted once.
    cg::SymbolicPoly const expected_traffic =
        monomial(1.0, {spaces.occ, spaces.aux}) + monomial(1.0, {spaces.occ, spaces.virt}) + monomial(1.0, {spaces.virt, spaces.aux});
    CHECK(pass.total_traffic() == expected_traffic);
    CHECK(pass.node_costs().front().cost.resident == expected_traffic);

    CHECK(pass.num_unannotated_nodes() == 0);
}

TEST_CASE("ScalingAnalysis - the rate-limiting node is the one whose flops dominate", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3); // occ x virt
    auto B = create_random_tensor<double>("B", 3, 2); // virt x occ

    cg::Graph graph("rate_limiting");
    graph.set_space_registry(spaces.registry);
    auto &Small = graph.create_zero_tensor<double, 2>("Small", 2, 2); // occ x occ
    auto &Big   = graph.create_zero_tensor<double, 2>("Big", 3, 3);   // virt x virt

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ia ; aj", &Small, A, B); // loop space o, o, v
        cg::einsum("ab <- ai ; ib", &Big, B, A);   // loop space v, v, o
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.occ});

    auto [_m, pass] = graph.apply<cg::passes::ScalingAnalysis>();

    REQUIRE(pass.num_analyzed() == 2);

    // With occ declared below virt, 2 o v^2 dominates 2 o^2 v asymptotically, so the second
    // contraction is the one that limits the program.
    cg::SymbolicPoly const dominant = monomial(2.0, {spaces.occ, spaces.virt, spaces.virt});
    REQUIRE(pass.rate_limiting().size() == 1);
    CHECK(pass.rate_limiting().front().cost.flops == dominant);
    CHECK(pass.rate_limiting().front().node_id == pass.node_costs()[1].node_id);

    CHECK(pass.total_flops() == monomial(2.0, {spaces.occ, spaces.occ, spaces.virt}) + dominant);
}

TEST_CASE("ScalingAnalysis - intermediates are sized and summed into a memory bound", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3); // occ x virt
    auto B = create_random_tensor<double>("B", 3, 4); // virt x aux
    auto E = create_random_tensor<double>("E", 4, 2); // aux x occ

    cg::Graph graph("intermediates");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 4);
    auto &D = graph.create_zero_tensor<double, 2>("D", 2, 2);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
        cg::einsum("ij <- ix ; xj", &D, C, E);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});
    graph.annotate_spaces(E, {spaces.aux, spaces.occ});
    graph.annotate_spaces(C, {spaces.occ, spaces.aux}); // what SpacePropagation would infer

    auto [_m, pass] = graph.apply<cg::passes::ScalingAnalysis>();

    REQUIRE(pass.intermediate_sizes().size() == 2);
    CHECK(pass.intermediate_sizes()[0].name == "C");
    CHECK(pass.intermediate_sizes()[0].size == monomial(1.0, {spaces.occ, spaces.aux}));
    CHECK(pass.intermediate_sizes()[1].name == "D");
    CHECK(pass.intermediate_sizes()[1].size == monomial(1.0, {spaces.occ, spaces.occ}));

    // An upper bound, deliberately: the sum of the sizes, not a liveness-aware high-water mark.
    CHECK(pass.memory_bound() == monomial(1.0, {spaces.occ, spaces.aux}) + monomial(1.0, {spaces.occ, spaces.occ}));
}

TEST_CASE("ScalingAnalysis - a half-annotated graph still reports, in anonymous variables", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3); // occ x virt
    auto B = create_random_tensor<double>("B", 3, 4); // unannotated

    cg::Graph graph("half_annotated");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});

    auto [_m, pass] = graph.apply<cg::passes::ScalingAnalysis>();

    REQUIRE(pass.num_analyzed() == 1);
    CHECK(pass.num_unannotated_nodes() == 1);

    // i and a resolve through A; x met no annotated slot and becomes the anonymous letter variable.
    cg::SymbolicPoly expected = monomial(2.0, {spaces.occ, spaces.virt});
    expected *= cg::SymbolicPoly::variable(cg::SymbolicVar::anonymous("x"));
    CHECK(pass.total_flops() == expected);
    CHECK(pass.total_flops().to_string(&spaces.registry) == "2*o*v*?x");

    CHECK(mentions(pass.explain(), "index-space annotation"));
}

TEST_CASE("ScalingAnalysis - an unannotated graph still yields a report", "[ComputeGraph][Passes][Spaces]") {
    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 3, 2);

    cg::Graph graph("unannotated");
    auto     &C = graph.create_zero_tensor<double, 2>("C", 2, 2);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ia ; aj", &C, A, B);
    }

    auto [_m, pass] = graph.apply<cg::passes::ScalingAnalysis>();

    REQUIRE(pass.num_analyzed() == 1);
    CHECK(pass.num_unannotated_nodes() == 1);
    CHECK(pass.total_flops().to_string(nullptr) == "2*?a*?i*?j");
    CHECK(pass.intermediate_sizes().size() == 1);

    auto const lines = pass.explain();
    CHECK_FALSE(lines.empty());
    CHECK(mentions(lines, "total flops"));
    CHECK(mentions(lines, "index-space annotation"));
}

TEST_CASE("ScalingAnalysis - non-contraction nodes are skipped with a counted reason", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 3, 2);
    auto F = create_random_tensor<double>("F", 2, 2);

    cg::Graph graph("skips");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 2);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ia ; aj", &C, A, B);
        cg::axpy(1.0, F, &C); // not a contraction: no cost formula is invented for it
    }

    auto [_m, pass] = graph.apply<cg::passes::ScalingAnalysis>();

    CHECK(pass.num_analyzed() == 1);

    CHECK(skipped_for(pass.skip_reasons(), "not a contraction node"));
}

TEST_CASE("ScalingAnalysis - explain and print_report carry the headline numbers", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 3, 4);

    cg::Graph graph("reporting");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    auto [_m, pass] = graph.apply<cg::passes::ScalingAnalysis>();

    auto const lines = pass.explain();
    REQUIRE_FALSE(lines.empty());
    CHECK(mentions(lines, "total flops 2*o*v*x"));
    CHECK(mentions(lines, "rate-limiting"));
    CHECK(mentions(lines, "intermediate footprint"));

    std::ostringstream out;
    pass.print_report(out);
    std::string const report = out.str();
    CHECK(report.find("ScalingAnalysis") != std::string::npos);
    CHECK(report.find("2*o*v*x") != std::string::npos);
    CHECK(report.find("memory bound") != std::string::npos);
    CHECK(report.size() > 100);
}

TEST_CASE("ScalingAnalysis - an empty report says nothing rather than saying nothing happened", "[ComputeGraph][Passes][Spaces]") {
    auto A = create_random_tensor<double>("A", 2, 2);

    cg::Graph graph("no_contractions");
    auto     &C = graph.create_zero_tensor<double, 2>("C", 2, 2);

    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(1.0, A, &C);
    }

    auto [_m, pass] = graph.apply<cg::passes::ScalingAnalysis>();

    CHECK(pass.num_analyzed() == 0);
    CHECK(pass.explain().empty());
    CHECK(pass.rate_limiting().empty());
    CHECK(pass.total_flops().is_zero());

    std::ostringstream out;
    pass.print_report(out);
    CHECK(out.str().find("no contraction nodes analysed") != std::string::npos);
}

TEST_CASE("ScalingAnalysis - the graph still executes and its results are unchanged", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 3, 2);

    cg::Graph graph("executable");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 2);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ia ; aj", &C, A, B);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.occ});

    auto [_m, pass] = graph.apply<cg::passes::ScalingAnalysis>();
    CHECK(pass.num_analyzed() == 1);

    graph.execute();

    // Summed by hand, so "no behavior change" is checked against something that shares no code
    // with the path under test.
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            double expected = 0.0;
            for (int link = 0; link < 3; ++link) {
                expected += A(row, link) * B(link, col);
            }
            CHECK_THAT(C(row, col), Catch::Matchers::WithinAbs(expected, 1.0e-12));
        }
    }
}
