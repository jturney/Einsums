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
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

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

// ── contribution_symmetry ────────────────────────────────────────────────────

namespace {

constexpr int nQ = 3, no = 3, nv = 4;

/// A tensor whose values have @p desc, declared so the graph knows it.
Tensor<double, 4> declared(std::string const &name, SymmetryDescriptor const &desc, std::size_t d0, std::size_t d1, std::size_t d2,
                           std::size_t d3) {
    auto t = create_random_tensor<double>(name, d0, d1, d2, d3);
    t.set_symmetry(desc);
    symmetrize(t);
    return t;
}

/// The density-fitted particle ladder, H = B tau then R = H B, captured into @p graph.
void capture_df_ladder(cg::Graph &graph, Tensor<double, 4> &R, Tensor<double, 3> const &B_first, Tensor<double, 4> const &tau,
                       Tensor<double, 3> const &B_second) {
    auto                  &H = graph.create_zero_tensor<double, 5>("H", nQ, nv, nv, no, no);
    cg::CaptureGuard const guard(graph);
    cg::einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", 0.0, &H, 1.0, B_first, tau);
    cg::einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", 0.0, &R, 1.0, H, B_second);
}

/// The node writing @p name.
cg::NodeId writer_of(cg::Graph const &graph, std::string_view name) {
    auto const id = id_of(graph, name);
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Einsum && node.outputs.size() == 1 && node.outputs[0] == id)
            return node.id;
    }
    FAIL("no einsum writes " << name);
    return cg::NodeId{};
}

} // namespace

TEST_CASE("contribution_symmetry - the DF ladder is symmetric under the (i,a)-(j,b) exchange", "[ComputeGraph][Symmetry]") {
    auto const B   = create_random_tensor<double>("B", nQ, nv, nv);
    auto const tau = declared("tau", pair_exchange(), no, no, nv, nv);
    cg::Graph  graph("df_ladder");
    auto      &R = graph.create_zero_tensor<double, 4>("R", no, no, nv, nv);
    capture_df_ladder(graph, R, B, tau, B);

    auto const escape = cg::EscapeAnalysis::over(graph);
    auto const op     = cg::contribution_symmetry(escape, writer_of(graph, "R"), pair_exchange().ops[0]);
    REQUIRE(op.has_value());
    REQUIRE(op->sign == +1);

    graph.execute();
    REQUIRE(values_have(R, SymmetryDescriptor{{*op}}));
}

TEST_CASE("contribution_symmetry - the DF ladder declines without tau's symmetry", "[ComputeGraph][Symmetry]") {
    auto const B   = create_random_tensor<double>("B", nQ, nv, nv);
    auto const tau = create_random_tensor<double>("tau", no, no, nv, nv);
    cg::Graph  graph("df_ladder_undeclared");
    auto      &R = graph.create_zero_tensor<double, 4>("R", no, no, nv, nv);
    capture_df_ladder(graph, R, B, tau, B);

    auto const escape = cg::EscapeAnalysis::over(graph);
    REQUIRE_FALSE(cg::contribution_symmetry(escape, writer_of(graph, "R"), pair_exchange().ops[0]).has_value());

    graph.execute();
    REQUIRE_FALSE(values_have(R, pair_exchange()));
}

TEST_CASE("contribution_symmetry - the DF ladder declines with a different tensor in the second B slot", "[ComputeGraph][Symmetry]") {
    auto const B   = create_random_tensor<double>("B", nQ, nv, nv);
    auto const B2  = create_random_tensor<double>("B2", nQ, nv, nv);
    auto const tau = declared("tau", pair_exchange(), no, no, nv, nv);
    cg::Graph  graph("df_ladder_two_b");
    auto      &R = graph.create_zero_tensor<double, 4>("R", no, no, nv, nv);
    capture_df_ladder(graph, R, B, tau, B2);

    auto const escape = cg::EscapeAnalysis::over(graph);
    REQUIRE_FALSE(cg::contribution_symmetry(escape, writer_of(graph, "R"), pair_exchange().ops[0]).has_value());

    graph.execute();
    REQUIRE_FALSE(values_have(R, pair_exchange()));
}

TEST_CASE("contribution_symmetry - the DF ladder is not symmetric under i<->j alone", "[ComputeGraph][Symmetry]") {
    auto const B   = create_random_tensor<double>("B", nQ, nv, nv);
    auto const tau = declared("tau", pair_exchange(), no, no, nv, nv);
    cg::Graph  graph("df_ladder_ij_only");
    auto      &R = graph.create_zero_tensor<double, 4>("R", no, no, nv, nv);
    capture_df_ladder(graph, R, B, tau, B);

    auto const escape = cg::EscapeAnalysis::over(graph);
    REQUIRE_FALSE(cg::contribution_symmetry(escape, writer_of(graph, "R"), SymmetryOp::swap(0, 1)).has_value());

    graph.execute();
    REQUIRE_FALSE(values_have(R, SymmetryDescriptor::symmetric_pair(0, 1)));
}

TEST_CASE("contribution_symmetry - the DF ladder declines when B changes between its two reads", "[ComputeGraph][Symmetry]") {
    // The two B factors trade places in the proof, which needs them to hold the same values.
    auto       B   = create_random_tensor<double>("B", nQ, nv, nv);
    auto const X   = create_random_tensor<double>("X", nQ, nv, nv);
    auto const tau = declared("tau", pair_exchange(), no, no, nv, nv);
    cg::Graph  graph("df_ladder_b_changes");
    auto      &H = graph.create_zero_tensor<double, 5>("H", nQ, nv, nv, no, no);
    auto      &R = graph.create_zero_tensor<double, 4>("R", no, no, nv, nv);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", 0.0, &H, 1.0, B, tau);
        cg::axpby(1.0, X, 1.0, &B);
        cg::einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", 0.0, &R, 1.0, H, B);
    }

    auto const escape = cg::EscapeAnalysis::over(graph);
    REQUIRE_FALSE(cg::contribution_symmetry(escape, writer_of(graph, "R"), pair_exchange().ops[0]).has_value());

    graph.execute();
    REQUIRE_FALSE(values_have(R, pair_exchange()));
}

TEST_CASE("contribution_symmetry - the conventional ladder takes the integral's own symmetry", "[ComputeGraph][Symmetry]") {
    // R_ijab = sum_ef tau_ijef V_abef with V_abef = V_bafe: the search exchanges e and f and uses
    // one generator of each factor.
    auto const tau = declared("tau", pair_exchange(), no, no, nv, nv);
    auto const V   = declared("V", pair_exchange(), nv, nv, nv, nv);
    cg::Graph  graph("conventional_ladder");
    auto      &R = graph.create_zero_tensor<double, 4>("R", no, no, nv, nv);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,j,e,f ; a,b,e,f", 0.0, &R, 1.0, tau, V);
    }

    auto const escape = cg::EscapeAnalysis::over(graph);
    auto const op     = cg::contribution_symmetry(escape, writer_of(graph, "R"), pair_exchange().ops[0]);
    REQUIRE(op.has_value());
    REQUIRE(op->sign == +1);

    graph.execute();
    REQUIRE(values_have(R, SymmetryDescriptor{{*op}}));
}

TEST_CASE("contribution_symmetry - an antisymmetric amplitude gives an antisymmetric ladder", "[ComputeGraph][Symmetry]") {
    // Spin-orbital form: tau antisymmetric in (i,j), so the ladder is too, with sign -1.
    auto const tau = declared("tau", SymmetryDescriptor{{SymmetryOp::swap(0, 1, -1), SymmetryOp::swap(2, 3, -1)}}, no, no, nv, nv);
    auto const V   = create_random_tensor<double>("V", nv, nv, nv, nv);
    cg::Graph  graph("spin_orbital_ladder");
    auto      &R = graph.create_zero_tensor<double, 4>("R", no, no, nv, nv);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,j,e,f ; a,b,e,f", 0.0, &R, 1.0, tau, V);
    }

    auto const escape = cg::EscapeAnalysis::over(graph);
    auto const op     = cg::contribution_symmetry(escape, writer_of(graph, "R"), SymmetryOp::swap(0, 1));
    REQUIRE(op.has_value());
    REQUIRE(op->sign == -1);

    graph.execute();
    REQUIRE(values_have(R, SymmetryDescriptor{{*op}}));
}

TEST_CASE("contribution_symmetry - an intermediate written twice is not inlined", "[ComputeGraph][Symmetry]") {
    // H's second write adds a term with no symmetry, so H is a factor of its own, and nothing is
    // known about it.
    auto const B   = create_random_tensor<double>("B", nQ, nv, nv);
    auto const Y   = create_random_tensor<double>("Y", nQ, nv, nv, no, no);
    auto const tau = declared("tau", pair_exchange(), no, no, nv, nv);
    cg::Graph  graph("df_ladder_two_writes");
    auto      &H = graph.create_zero_tensor<double, 5>("H", nQ, nv, nv, no, no);
    auto      &R = graph.create_zero_tensor<double, 4>("R", no, no, nv, nv);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", 0.0, &H, 1.0, B, tau);
        cg::axpby(1.0, Y, 1.0, &H);
        cg::einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", 0.0, &R, 1.0, H, B);
    }

    auto const escape = cg::EscapeAnalysis::over(graph);
    REQUIRE_FALSE(cg::contribution_symmetry(escape, writer_of(graph, "R"), pair_exchange().ops[0]).has_value());

    graph.execute();
    REQUIRE_FALSE(values_have(R, pair_exchange()));
}

TEST_CASE("contribution_symmetry - every answer on random contractions holds for the values", "[ComputeGraph][Symmetry]") {
    // Soundness sweep: random two-operand einsums over a pool of declared symmetric, antisymmetric
    // and general operands (one tensor may fill both slots), random output letters and a random
    // permutation of them. Whenever the proof answers, the executed contribution must satisfy it.
    constexpr std::size_t n    = 3;
    auto                  make = [](std::string const &name, std::size_t rank, std::optional<SymmetryDescriptor> desc) {
        auto t = rank == 2 ? RuntimeTensor<double>(create_random_tensor<double>(name, n, n))
                           : RuntimeTensor<double>(create_random_tensor<double>(name, n, n, n, n));
        if (desc) {
            if (rank == 2) {
                auto s = create_random_tensor<double>(name, n, n);
                s.set_symmetry(*desc);
                symmetrize(s);
                t = RuntimeTensor<double>(s);
            } else {
                auto s = create_random_tensor<double>(name, n, n, n, n);
                s.set_symmetry(*desc);
                symmetrize(s);
                t = RuntimeTensor<double>(s);
            }
            t.set_symmetry(*desc);
        }
        return t;
    };
    std::vector<RuntimeTensor<double>> pool;
    pool.push_back(make("S4", 4, pair_exchange()));
    pool.push_back(make("A4", 4, SymmetryDescriptor{{SymmetryOp::swap(0, 1, -1), SymmetryOp::swap(2, 3, -1)}}));
    pool.push_back(make("G4", 4, std::nullopt));
    pool.push_back(make("S2", 2, SymmetryDescriptor::symmetric_pair(0, 1)));
    pool.push_back(make("N2", 2, SymmetryDescriptor::antisymmetric_pair(0, 1)));
    pool.push_back(make("G2", 2, std::nullopt));

    std::mt19937                   rng(20261020);
    std::vector<std::string> const alphabet{"i", "j", "k", "l"};
    auto pick   = [&rng](std::size_t bound) { return std::uniform_int_distribution<std::size_t>(0, bound - 1)(rng); };
    int  proven = 0;
    for (int trial = 0; trial < 400; ++trial) {
        auto const &A = pool[pick(pool.size())];
        auto const &B = pick(3) == 0 ? A : pool[pick(pool.size())];

        std::vector<std::string> a_letters, b_letters;
        for (std::size_t p = 0; p < A.rank(); ++p)
            a_letters.push_back(alphabet[pick(alphabet.size())]);
        for (std::size_t p = 0; p < B.rank(); ++p)
            b_letters.push_back(alphabet[pick(alphabet.size())]);

        // Output: two to four distinct letters the operands use, in a random order.
        std::vector<std::string> used;
        for (auto const &l : a_letters)
            if (std::ranges::find(used, l) == used.end())
                used.push_back(l);
        for (auto const &l : b_letters)
            if (std::ranges::find(used, l) == used.end())
                used.push_back(l);
        if (used.size() < 2)
            continue;
        std::ranges::shuffle(used, rng);
        used.resize(2 + pick(used.size() - 1));

        auto                sigma = SymmetryOp::identity();
        std::vector<int8_t> order(used.size());
        std::iota(order.begin(), order.end(), int8_t{0});
        std::ranges::shuffle(order, rng);
        for (std::size_t p = 0; p < order.size(); ++p)
            sigma.permutation[p] = order[p];

        auto const spec = fmt::format("{} <- {} ; {}", fmt::join(used, ","), fmt::join(a_letters, ","), fmt::join(b_letters, ","));
        RuntimeTensor<double> R("R", std::vector<std::size_t>(used.size(), n));
        cg::Graph             graph("random");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum(std::string_view{spec}, 0.0, &R, 1.0, A, B);
        }

        auto const escape = cg::EscapeAnalysis::over(graph);
        auto const op     = cg::contribution_symmetry(escape, only_node(graph, cg::OpKind::Einsum).id, sigma);
        if (!op)
            continue;
        ++proven;
        graph.execute();
        INFO(spec << " with " << A.name() << ", " << B.name() << ", sign " << int{op->sign});
        REQUIRE(check_symmetry(R, SymmetryDescriptor{{*op}}, 1e-10));
    }
    // The sweep must reach the proof often enough to mean something.
    REQUIRE(proven >= 40);
}
