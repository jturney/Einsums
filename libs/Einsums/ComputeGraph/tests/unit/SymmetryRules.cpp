//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file SymmetryRules.cpp
/// @brief Unit tests for output_symmetry, the per-node symmetry rules every pass shares.
///
/// These call the rules directly, without a pass around them, so a rule is checked apart from the
/// guard a caller puts in front of it: SymmetryPropagation would decline to tag most of these
/// outputs (user-owned, or written twice), which is the caller's decision and not the rule's.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/SymmetryRules.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <optional>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The one node of @p graph whose kind is @p kind.
cg::Node const &only_node(cg::Graph const &graph, cg::OpKind kind) {
    cg::Node const *found = nullptr;
    for (auto const &node : graph.nodes()) {
        if (node.kind == kind) {
            REQUIRE(found == nullptr);
            found = &node;
        }
    }
    REQUIRE(found != nullptr);
    return *found;
}

} // namespace

TEST_CASE("output_symmetry - Scale keeps its input's descriptor", "[ComputeGraph][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    A.set_symmetry(SymmetryDescriptor::antisymmetric_pair(0, 1));

    cg::Graph graph("scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }

    auto const desc = cg::output_symmetry(graph, only_node(graph, cg::OpKind::Scale));
    REQUIRE(desc.has_value());
    REQUIRE(*desc == SymmetryDescriptor::antisymmetric_pair(0, 1));
}

TEST_CASE("output_symmetry - Scale of a tensor with no descriptor says nothing", "[ComputeGraph][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);

    cg::Graph graph("scale_general");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }

    REQUIRE_FALSE(cg::output_symmetry(graph, only_node(graph, cg::OpKind::Scale)).has_value());
}

TEST_CASE("output_symmetry - Axpby of two equally symmetric tensors is symmetric", "[ComputeGraph][Symmetry]") {
    auto X = create_random_tensor<double>("X", 4, 4);
    auto Y = create_random_tensor<double>("Y", 4, 4);
    X.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
    Y.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));

    cg::Graph graph("axpby");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(0.5, X, 2.0, &Y);
    }

    auto const desc = cg::output_symmetry(graph, only_node(graph, cg::OpKind::Axpby));
    REQUIRE(desc.has_value());
    REQUIRE(*desc == SymmetryDescriptor::symmetric_pair(0, 1));
}

TEST_CASE("output_symmetry - Axpby of differently structured tensors says nothing", "[ComputeGraph][Symmetry]") {
    // A symmetric plus an antisymmetric matrix is neither, so the descriptors must match exactly.
    auto X = create_random_tensor<double>("X", 4, 4);
    auto Y = create_random_tensor<double>("Y", 4, 4);
    X.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
    Y.set_symmetry(SymmetryDescriptor::antisymmetric_pair(0, 1));

    cg::Graph graph("axpby_mixed");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(1.0, X, 1.0, &Y);
    }

    REQUIRE_FALSE(cg::output_symmetry(graph, only_node(graph, cg::OpKind::Axpby)).has_value());
}

TEST_CASE("output_symmetry - a self-contraction is symmetric, or Hermitian with one side conjugated", "[ComputeGraph][Symmetry]") {
    SECTION("real") {
        auto const A = create_random_tensor<double>("A", 5, 3);
        cg::Graph  graph("ata_real");
        auto      &C = graph.create_zero_tensor<double, 2>("C", 3, 3);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ki;kj->ij", 0.0, &C, 1.0, A, A);
        }
        auto const desc = cg::output_symmetry(graph, only_node(graph, cg::OpKind::Einsum));
        REQUIRE(desc.has_value());
        REQUIRE(*desc == SymmetryDescriptor::symmetric_pair(0, 1));
    }

    SECTION("complex, A^H A") {
        auto const A = create_random_tensor<std::complex<double>>("A", 5, 3);
        cg::Graph  graph("aha");
        auto      &C = graph.create_zero_tensor<std::complex<double>, 2>("C", 3, 3);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ki;kj->ij", 0.0, &C, 1.0, A, A, /*conj_a=*/true, /*conj_b=*/false);
        }
        auto const desc = cg::output_symmetry(graph, only_node(graph, cg::OpKind::Einsum));
        REQUIRE(desc.has_value());
        REQUIRE(*desc == SymmetryDescriptor::hermitian_pair(0, 1));
    }

    SECTION("complex, no conjugation") {
        auto const A = create_random_tensor<std::complex<double>>("A", 5, 3);
        cg::Graph  graph("ata_complex");
        auto      &C = graph.create_zero_tensor<std::complex<double>, 2>("C", 3, 3);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ki;kj->ij", 0.0, &C, 1.0, A, A);
        }
        auto const desc = cg::output_symmetry(graph, only_node(graph, cg::OpKind::Einsum));
        REQUIRE(desc.has_value());
        REQUIRE(*desc == SymmetryDescriptor::symmetric_pair(0, 1));
    }
}

TEST_CASE("output_symmetry - a contraction of two different tensors says nothing", "[ComputeGraph][Symmetry]") {
    auto const A = create_random_tensor<double>("A", 4, 4);
    auto const B = create_random_tensor<double>("B", 4, 4);
    cg::Graph  graph("ab");
    auto      &C = graph.create_zero_tensor<double, 2>("C", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ki;kj->ij", 0.0, &C, 1.0, A, B);
    }

    REQUIRE_FALSE(cg::output_symmetry(graph, only_node(graph, cg::OpKind::Einsum)).has_value());
}

TEST_CASE("output_symmetry - a permute keeps a symmetric input's descriptor only when it overwrites", "[ComputeGraph][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    A.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));

    SECTION("zero destination prefactor") {
        cg::Graph graph("permute_overwrite");
        auto     &T = graph.create_zero_tensor<double, 2>("T", 4, 4);
        {
            cg::CaptureGuard const guard(graph);
            cg::permute("ji <- ij", 0.0, &T, 1.0, A);
        }
        auto const desc = cg::output_symmetry(graph, only_node(graph, cg::OpKind::Permute));
        REQUIRE(desc.has_value());
        REQUIRE(*desc == SymmetryDescriptor::symmetric_pair(0, 1));
    }

    SECTION("accumulating") {
        // T's old contents are unknown, so T + A^T is not known to be symmetric.
        cg::Graph graph("permute_accumulate");
        auto     &T = graph.create_zero_tensor<double, 2>("T", 4, 4);
        {
            cg::CaptureGuard const guard(graph);
            cg::permute("ji <- ij", 1.0, &T, 1.0, A);
        }
        REQUIRE_FALSE(cg::output_symmetry(graph, only_node(graph, cg::OpKind::Permute)).has_value());
    }
}
