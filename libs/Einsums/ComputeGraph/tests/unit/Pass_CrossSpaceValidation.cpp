//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_CrossSpaceValidation.cpp
/// @brief Unit tests for the CrossSpaceValidation pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The spaces every case below draws from, in a registry the case owns so no two tests can see
/// each other's declarations. Occupied and virtual orbitals share no element, and a PNO domain
/// lives inside the virtual space; both facts are declarations, since the registry infers nothing.
struct Spaces {
    cg::SpaceRegistry registry;
    cg::SpaceId       occ;
    cg::SpaceId       virt;
    cg::SpaceId       aux;
    cg::SpaceId       pno;

    Spaces() {
        occ  = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .typical_extent = 2.0});
        virt = registry.register_space(cg::IndexSpace{.name = "virt", .scale_symbol = "v", .typical_extent = 3.0});
        aux  = registry.register_space(cg::IndexSpace{.name = "aux", .scale_symbol = "x", .typical_extent = 4.0});
        pno  = registry.register_space(cg::IndexSpace{.name = "pno", .scale_symbol = "p", .typical_extent = 2.0});
        registry.declare_disjoint(occ, virt);
        registry.declare_contained(pno, virt);
    }
};

/// Whether a pass's skip tally records the given reason at least once.
bool skipped_for(std::vector<std::pair<std::string, std::size_t>> const &skips, std::string_view reason) {
    return std::ranges::any_of(skips, [reason](auto const &entry) { return entry.first == reason; });
}

/// Whether any line of a report mentions the given text.
bool mentions(std::vector<std::string> const &lines, std::string_view text) {
    return std::ranges::any_of(lines, [text](std::string const &line) { return line.find(text) != std::string::npos; });
}

} // namespace

TEST_CASE("CrossSpaceValidation - a consistently annotated graph reports nothing", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3); // occ x virt
    auto B = create_random_tensor<double>("B", 3, 4); // virt x aux

    cg::Graph graph("clean");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    auto [modified, pass] = graph.apply<cg::passes::CrossSpaceValidation>();

    CHECK_FALSE(modified); // a diagnostic pass never claims a modification
    CHECK(pass.findings().empty());
    CHECK(pass.num_errors() == 0);
    CHECK(pass.explain().empty());

    graph.execute();

    // The contraction summed by hand: the pass changed nothing about what the graph computes.
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 4; ++col) {
            double expected = 0.0;
            for (int link = 0; link < 3; ++link) {
                expected += A(row, link) * B(link, col);
            }
            CHECK_THAT(C(row, col), Catch::Matchers::WithinAbs(expected, 1.0e-12));
        }
    }
}

TEST_CASE("CrossSpaceValidation - a letter binding two disjoint spaces is an error", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 3, 4);

    cg::Graph graph("disjoint");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    // Annotated AFTER capture, which is precisely what capture's own conflict check cannot see:
    // 'a' is a virtual slot on A and an occupied slot on B, and the two are declared disjoint.
    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.occ, spaces.aux});

    auto [_m, pass] = graph.apply<cg::passes::CrossSpaceValidation>();

    REQUIRE(pass.findings().size() == 1);
    auto const &finding = pass.findings().front();
    CHECK(finding.severity == cg::passes::CrossSpaceSeverity::Error);
    CHECK(finding.letter == "a");
    CHECK(finding.first_space_name == "virt");
    CHECK(finding.second_space_name == "occ");
    CHECK(finding.first_tensor_name == "A");
    CHECK(finding.second_tensor_name == "B");
    CHECK_FALSE(finding.rests_on_inferred);
    CHECK(finding.message.find("'a'") != std::string::npos);
    CHECK(finding.message.find("virt") != std::string::npos);
    CHECK(finding.message.find("occ") != std::string::npos);
    CHECK(finding.message.find("disjoint") != std::string::npos);

    CHECK(pass.num_errors() == 1);
    CHECK(pass.num_warnings() == 0);
    CHECK(pass.num_notes() == 0);
    CHECK(mentions(pass.explain(), "1 error(s), 0 warning(s), 0 note(s)"));
}

TEST_CASE("CrossSpaceValidation - a contained space is a restriction, not a mistake", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 3, 4);

    cg::Graph graph("restriction");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    // 'a' is a PNO slot on A and the parent virtual slot on B: legitimate per the containment
    // reasoning, and listed because an unintended restriction looks exactly like an intended one.
    graph.annotate_spaces(A, {spaces.occ, spaces.pno});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    auto [_m, pass] = graph.apply<cg::passes::CrossSpaceValidation>();

    REQUIRE(pass.findings().size() == 1);
    auto const &finding = pass.findings().front();
    CHECK(finding.severity == cg::passes::CrossSpaceSeverity::Note);
    CHECK(finding.letter == "a");
    CHECK(finding.message.find("contained") != std::string::npos);
    CHECK(finding.message.find("restricts") != std::string::npos);

    CHECK(pass.num_errors() == 0);
    CHECK(pass.num_notes() == 1);
}

TEST_CASE("CrossSpaceValidation - an undeclared relation is only a warning", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 3, 4);

    cg::Graph graph("unknown_relation");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    // Nothing declared relates virt and aux, and "unknown" has to be treated as carefully as "no".
    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.aux, spaces.aux});

    auto [_m, pass] = graph.apply<cg::passes::CrossSpaceValidation>();

    REQUIRE(pass.findings().size() == 1);
    CHECK(pass.findings().front().severity == cg::passes::CrossSpaceSeverity::Warning);
    CHECK(pass.findings().front().message.find("no declared relation") != std::string::npos);
    CHECK(pass.num_warnings() == 1);
    CHECK(pass.num_errors() == 0);
}

TEST_CASE("CrossSpaceValidation - a verdict resting on an inferred annotation is downgraded", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3); // occ x virt
    auto B = create_random_tensor<double>("B", 3, 4); // virt x aux
    auto G = create_random_tensor<double>("G", 4, 5); // declared occ x virt, conflicting with C

    cg::Graph graph("inferred");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 4);
    auto &D = graph.create_zero_tensor<double, 2>("D", 2, 5);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
        cg::einsum("iy <- ix ; xy", &D, C, G);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});
    graph.annotate_spaces(G, {spaces.occ, spaces.virt});

    // SpacePropagation infers C as (occ, aux). The second contraction then binds 'x' to that
    // inferred aux against G's declared occ; aux and occ are not related, but C's side is a
    // derivation rather than a declaration, so the warning drops to a note and says so.
    auto            validation = std::make_shared<cg::passes::CrossSpaceValidation>();
    cg::PassManager pm;
    pm.add<cg::passes::SpacePropagation>();
    pm.add(validation);
    pm.run(graph);

    REQUIRE(validation->findings().size() == 1);
    auto const &finding = validation->findings().front();
    CHECK(finding.letter == "x");
    CHECK(finding.rests_on_inferred);
    CHECK(finding.severity == cg::passes::CrossSpaceSeverity::Note);
    CHECK(finding.message.find("INFERRED") != std::string::npos);
    CHECK(validation->num_notes() == 1);
    CHECK(validation->num_warnings() == 0);
}

TEST_CASE("CrossSpaceValidation - an unannotated graph costs nothing and finds nothing", "[ComputeGraph][Passes][Spaces]") {
    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 3, 2);

    cg::Graph graph("unannotated");
    auto     &C = graph.create_zero_tensor<double, 2>("C", 2, 2);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ia ; aj", &C, A, B);
    }

    auto [_m, pass] = graph.apply<cg::passes::CrossSpaceValidation>();

    CHECK(pass.findings().empty());
    CHECK(pass.num_errors() == 0);
    CHECK(pass.num_warnings() == 0);
    CHECK(pass.num_notes() == 0);
    CHECK(pass.explain().empty());

    CHECK(skipped_for(pass.skip_reasons(), "no operand carries an index-space annotation"));

    graph.execute();

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

TEST_CASE("CrossSpaceValidation - the severity counts match the findings and the report", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 3, 4);
    auto E = create_random_tensor<double>("E", 2, 3);
    auto F = create_random_tensor<double>("F", 3, 4);

    cg::Graph graph("counts");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 4);
    auto &D = graph.create_zero_tensor<double, 2>("D", 2, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B); // 'a': virt against occ, declared disjoint
        cg::einsum("jy <- jb ; by", &D, E, F); // 'b': pno against virt, a restriction
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.occ, spaces.aux});
    graph.annotate_spaces(E, {spaces.occ, spaces.pno});
    graph.annotate_spaces(F, {spaces.virt, spaces.aux});

    auto [_m, pass] = graph.apply<cg::passes::CrossSpaceValidation>();

    REQUIRE(pass.findings().size() == 2);

    std::size_t errors   = 0;
    std::size_t warnings = 0;
    std::size_t notes    = 0;
    for (auto const &finding : pass.findings()) {
        switch (finding.severity) {
        case cg::passes::CrossSpaceSeverity::Error:
            ++errors;
            break;
        case cg::passes::CrossSpaceSeverity::Warning:
            ++warnings;
            break;
        case cg::passes::CrossSpaceSeverity::Note:
            ++notes;
            break;
        }
    }
    CHECK(errors == pass.num_errors());
    CHECK(warnings == pass.num_warnings());
    CHECK(notes == pass.num_notes());
    CHECK(errors == 1);
    CHECK(notes == 1);

    auto const lines = pass.explain();
    REQUIRE_FALSE(lines.empty());
    CHECK(mentions(lines, "1 error(s), 0 warning(s), 1 note(s)"));

    std::ostringstream out;
    pass.print_report(out);
    std::string const report = out.str();
    CHECK(report.find("CrossSpaceValidation") != std::string::npos);
    CHECK(report.find("disjoint") != std::string::npos);
    CHECK(report.find("restricts") != std::string::npos);

    // Most severe first, so the error precedes the note in the rendered report.
    CHECK(report.find("error:") < report.find("note:"));
}

TEST_CASE("CrossSpaceValidation - non-contraction nodes are skipped with a counted reason", "[ComputeGraph][Passes][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 2, 2);

    cg::Graph graph("skips");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 2, 2);

    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(1.0, A, &C);
    }

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});

    auto [_m, pass] = graph.apply<cg::passes::CrossSpaceValidation>();

    CHECK(pass.findings().empty());
    CHECK(skipped_for(pass.skip_reasons(), "not a contraction node"));
}
