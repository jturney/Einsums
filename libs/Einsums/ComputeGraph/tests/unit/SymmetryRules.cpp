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
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <cstddef>
#include <optional>
#include <string_view>

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

namespace {

/// Whether @p tensor's values have the symmetry @p desc describes, checked element by element.
template <typename T, std::size_t Rank>
bool values_have(Tensor<T, Rank> &tensor, SymmetryDescriptor const &desc) {
    tensor.set_symmetry(desc);
    return check_symmetry(tensor, 1e-12);
}

/// The graph tensor named @p name, or an invalid id.
cg::TensorId id_of(cg::Graph const &graph, std::string_view name) {
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.name == name) {
            return id;
        }
    }
    return cg::TensorId{};
}

/// The exchange of the index pair at positions (0, 2) with the pair at (1, 3): T(i,j,a,b) = T(j,i,b,a).
SymmetryDescriptor pair_exchange() {
    return SymmetryDescriptor{{SymmetryOp::group_swap({0, 2}, {1, 3})}};
}

} // namespace

TEST_CASE("self_product_symmetry - an outer product of t1 with itself exchanges (i,a) with (j,b)", "[ComputeGraph][Symmetry]") {
    auto const t1 = create_random_tensor<double>("t1", 3, 4);
    cg::Graph  graph("tau_t1t1");
    auto      &tau = graph.create_zero_tensor<double, 4>("tau", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", 0.0, &tau, 1.0, t1, t1);
    }

    auto const desc = cg::self_product_symmetry(graph, only_node(graph, cg::OpKind::Einsum));
    REQUIRE(desc.has_value());
    REQUIRE(*desc == pair_exchange());

    graph.execute();
    REQUIRE(values_have(tau, *desc));
}

TEST_CASE("self_product_symmetry - a rank-4 self contraction exchanges its two pairs", "[ComputeGraph][Symmetry]") {
    auto const X = create_random_tensor<double>("X", 3, 4, 5);
    cg::Graph  graph("xx");
    auto      &C = graph.create_zero_tensor<double, 4>("C", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a,k ; j,b,k", 0.0, &C, 1.0, X, X);
    }

    auto const desc = cg::self_product_symmetry(graph, only_node(graph, cg::OpKind::Einsum));
    REQUIRE(desc.has_value());
    REQUIRE(*desc == pair_exchange());

    graph.execute();
    REQUIRE(values_have(C, *desc));
}

TEST_CASE("self_product_symmetry - a matrix times itself is not symmetric", "[ComputeGraph][Symmetry]") {
    // ij <- ik ; kj with A in both slots: the relabeling would send k to j from A's side and to i
    // from B's, so there is no consistent swap, and A A is indeed not symmetric for general A.
    auto const A = create_random_tensor<double>("A", 4, 4);
    cg::Graph  graph("aa");
    auto      &C = graph.create_zero_tensor<double, 2>("C", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, A);
    }

    REQUIRE_FALSE(cg::self_product_symmetry(graph, only_node(graph, cg::OpKind::Einsum)).has_value());
}

TEST_CASE("self_product_symmetry - an inconsistent relabeling declines even when C's letters close", "[ComputeGraph][Symmetry]") {
    // ij <- ijk ; jki: swapping the operands would send j to i (second slot onto first) and to k
    // (first onto second). Keeping only the first binding of each letter would find i <-> j, which
    // maps C onto itself, and call C symmetric; it is not, as the executed values show.
    auto const A = create_random_tensor<double>("A", 3, 3, 3);
    cg::Graph  graph("cyclic");
    auto      &C = graph.create_zero_tensor<double, 2>("C", 3, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ijk ; jki", 0.0, &C, 1.0, A, A);
    }

    REQUIRE_FALSE(cg::self_product_symmetry(graph, only_node(graph, cg::OpKind::Einsum)).has_value());

    graph.execute();
    REQUIRE_FALSE(values_have(C, SymmetryDescriptor::symmetric_pair(0, 1)));
}

TEST_CASE("self_product_symmetry - one conjugated operand gives a conjugating generator", "[ComputeGraph][Symmetry]") {
    auto const X = create_random_tensor<std::complex<double>>("X", 3, 4, 5);
    cg::Graph  graph("xhx");
    auto      &C = graph.create_zero_tensor<std::complex<double>, 4>("C", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a,k ; j,b,k", 0.0, &C, 1.0, X, X, /*conj_a=*/true, /*conj_b=*/false);
    }

    auto const desc = cg::self_product_symmetry(graph, only_node(graph, cg::OpKind::Einsum));
    REQUIRE(desc.has_value());
    REQUIRE(desc->ops.size() == 1);
    REQUIRE(desc->ops[0].conjugate);

    graph.execute();
    REQUIRE(values_have(C, *desc));
}

TEST_CASE("output_symmetry - a general self product says nothing when it accumulates", "[ComputeGraph][Symmetry]") {
    // The product is symmetric, but C's earlier contents are not known to be, so neither is the sum.
    auto const t1 = create_random_tensor<double>("t1", 3, 4);
    cg::Graph  graph("tau_accumulate");
    auto      &tau = graph.create_zero_tensor<double, 4>("tau", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", 1.0, &tau, 1.0, t1, t1);
    }

    auto const &node = only_node(graph, cg::OpKind::Einsum);
    REQUIRE(cg::self_product_symmetry(graph, node).has_value());
    REQUIRE_FALSE(cg::output_symmetry(graph, node).has_value());
}

TEST_CASE("symmetry_at - tau, an outer product plus a declared-symmetric t2", "[ComputeGraph][Symmetry]") {
    // The coupled-cluster amplitude intermediate: two writes, so no single-writer guard accepts it.
    auto const t1 = create_random_tensor<double>("t1", 3, 4);
    auto       t2 = create_random_tensor<double>("t2", 3, 3, 4, 4);
    t2.set_symmetry(pair_exchange());
    symmetrize(t2);

    cg::Graph graph("tau");
    auto     &tau  = graph.create_zero_tensor<double, 4>("tau", 3, 3, 4, 4);
    auto     &read = graph.create_zero_tensor<double, 4>("read", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", 0.0, &tau, 1.0, t1, t1);
        cg::axpby(1.0, t2, 1.0, &tau);
        cg::permute("i,j,a,b <- i,j,a,b", 0.0, &read, 1.0, tau);
    }

    auto const escape = cg::EscapeAnalysis::over(graph);
    auto const desc   = cg::symmetry_at(escape, id_of(graph, "tau"), only_node(graph, cg::OpKind::Permute).id);
    REQUIRE(desc.has_value());
    REQUIRE(*desc == pair_exchange());

    graph.execute();
    REQUIRE(values_have(tau, *desc));
}

TEST_CASE("symmetry_at - an undeclared t2 gives nothing", "[ComputeGraph][Symmetry]") {
    auto const t1 = create_random_tensor<double>("t1", 3, 4);
    auto const t2 = create_random_tensor<double>("t2", 3, 3, 4, 4);

    cg::Graph graph("tau_undeclared");
    auto     &tau  = graph.create_zero_tensor<double, 4>("tau", 3, 3, 4, 4);
    auto     &read = graph.create_zero_tensor<double, 4>("read", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", 0.0, &tau, 1.0, t1, t1);
        cg::axpby(1.0, t2, 1.0, &tau);
        cg::permute("i,j,a,b <- i,j,a,b", 0.0, &read, 1.0, tau);
    }

    auto const escape = cg::EscapeAnalysis::over(graph);
    REQUIRE_FALSE(cg::symmetry_at(escape, id_of(graph, "tau"), only_node(graph, cg::OpKind::Permute).id).has_value());
}

TEST_CASE("symmetry_at - a read between the two writes sees the first write's symmetry", "[ComputeGraph][Symmetry]") {
    // The later write adds an undeclared tensor, which would break the symmetry, but the read
    // comes before it.
    auto const t1 = create_random_tensor<double>("t1", 3, 4);
    auto const t2 = create_random_tensor<double>("t2", 3, 3, 4, 4);

    cg::Graph graph("tau_read_between");
    auto     &tau  = graph.create_zero_tensor<double, 4>("tau", 3, 3, 4, 4);
    auto     &read = graph.create_zero_tensor<double, 4>("read", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", 0.0, &tau, 1.0, t1, t1);
        cg::permute("i,j,a,b <- i,j,a,b", 0.0, &read, 1.0, tau);
        cg::axpby(1.0, t2, 1.0, &tau);
    }

    auto const escape = cg::EscapeAnalysis::over(graph);
    auto const desc   = cg::symmetry_at(escape, id_of(graph, "tau"), only_node(graph, cg::OpKind::Permute).id);
    REQUIRE(desc.has_value());
    REQUIRE(*desc == pair_exchange());

    graph.execute();
    REQUIRE(values_have(read, *desc));
}

TEST_CASE("symmetry_at - an accumulation onto contents nothing set gives nothing", "[ComputeGraph][Symmetry]") {
    auto const t1 = create_random_tensor<double>("t1", 3, 4);

    cg::Graph graph("tau_accumulate_first");
    auto     &tau  = graph.create_zero_tensor<double, 4>("tau", 3, 3, 4, 4);
    auto     &read = graph.create_zero_tensor<double, 4>("read", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", 1.0, &tau, 1.0, t1, t1);
        cg::permute("i,j,a,b <- i,j,a,b", 0.0, &read, 1.0, tau);
    }

    auto const escape = cg::EscapeAnalysis::over(graph);
    REQUIRE_FALSE(cg::symmetry_at(escape, id_of(graph, "tau"), only_node(graph, cg::OpKind::Permute).id).has_value());
}

TEST_CASE("symmetry_at - a read with no write before it takes the declared symmetry", "[ComputeGraph][Symmetry]") {
    // A loop-carried amplitude: the graph reads it before anything in the graph writes it.
    auto t2 = create_random_tensor<double>("t2", 3, 3, 4, 4);
    t2.set_symmetry(pair_exchange());

    cg::Graph graph("t2_declared");
    auto     &read = graph.create_zero_tensor<double, 4>("read", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("i,j,a,b <- i,j,a,b", 0.0, &read, 1.0, t2);
    }

    auto const escape = cg::EscapeAnalysis::over(graph);
    auto const desc   = cg::symmetry_at(escape, id_of(graph, "t2"), only_node(graph, cg::OpKind::Permute).id);
    REQUIRE(desc.has_value());
    REQUIRE(*desc == pair_exchange());
}

TEST_CASE("symmetry_at - a tensor a nested loop writes gives nothing", "[ComputeGraph][Symmetry][Loop]") {
    // The inner body's write does not appear in the outer body's node list.
    auto const t1 = create_random_tensor<double>("t1", 3, 4);
    auto const t2 = create_random_tensor<double>("t2", 3, 3, 4, 4);

    cg::Graph g("tau_nested");
    auto     &outer = g.add_loop("outer", 1, [](size_t) { return false; });
    auto     &tau   = outer.create_zero_tensor<double, 4>("tau", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(outer);
        cg::einsum("i,j,a,b <- i,a ; j,b", 0.0, &tau, 1.0, t1, t1);
    }
    auto &inner = outer.add_loop("inner", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(inner);
        cg::axpby(1.0, t2, 1.0, &tau);
    }
    auto &read = outer.create_zero_tensor<double, 4>("read", 3, 3, 4, 4);
    {
        cg::CaptureGuard const guard(outer);
        cg::permute("i,j,a,b <- i,j,a,b", 0.0, &read, 1.0, tau);
    }

    auto const escape = cg::EscapeAnalysis::over(outer);
    REQUIRE_FALSE(cg::symmetry_at(escape, id_of(outer, "tau"), only_node(outer, cg::OpKind::Permute).id).has_value());
}
