//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// ScratchPrivatization's skip tally. The pass declined silently: a scratch tensor it left alone
// looked exactly like one it never saw, and a test that built its scratch with an allocation node
// read "0 privatized" as the pass refusing something else. Each case builds the CCSD idiom it is
// for - one scratch reused through write, read episodes - and breaks it one way.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/ScratchPrivatization.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <memory>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

struct Operands {
    std::vector<RuntimeTensor<double>> ops;
    RuntimeTensor<double>              acc{"acc", std::vector<size_t>{6, 6}};

    Operands() {
        for (int k = 0; k < 3; ++k) {
            ops.emplace_back(create_random_tensor<double>("A", 6, 6));
        }
        acc.zero();
    }
};

/// Run the pass alone and return it.
std::shared_ptr<cg::passes::ScratchPrivatization> run(cg::Graph &graph, bool require_executor) {
    auto pass = std::make_shared<cg::passes::ScratchPrivatization>();
    pass->set_require_executor(require_executor);
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);
    return pass;
}

/// The single reason the pass gave, or fail the test.
std::string only_reason(cg::passes::ScratchPrivatization const &pass) {
    auto const reasons = pass.skip_reasons();
    REQUIRE(reasons.size() == 1);
    return reasons[0].first;
}

} // namespace

TEST_CASE("ScratchPrivatization - a reused scratch is split, and nothing else is reported", "[ComputeGraph][ScratchPrivatization]") {
    // The inputs, the accumulator and the scratch are all candidates the pass looks at. Only the
    // scratch is reused, so only it may appear in the tally, and here it is privatized.
    Operands  in;
    cg::Graph graph("privatized");
    auto     &tmp = graph.declare_zero_runtime_tensor<double>("tmp", {6, 6}, true);
    {
        cg::CaptureGuard const guard(graph);
        for (auto const &A : in.ops) {
            cg::einsum("ij <- ik ; kj", 0.0, &tmp, 1.0, A, A);
            cg::axpby(1.0, tmp, 1.0, &in.acc);
        }
    }
    auto const pass = run(graph, false);
    CHECK(pass->num_tensors_privatized() == 1);
    CHECK(pass->skip_reasons().empty());
}

TEST_CASE("ScratchPrivatization - reports why it declined", "[ComputeGraph][ScratchPrivatization]") {
    Operands  in;
    cg::Graph graph("declined");

    SECTION("no executor is installed") {
        auto &tmp = graph.declare_zero_runtime_tensor<double>("tmp", {6, 6}, true);
        {
            cg::CaptureGuard const guard(graph);
            for (auto const &A : in.ops) {
                cg::einsum("ij <- ik ; kj", 0.0, &tmp, 1.0, A, A);
                cg::axpby(1.0, tmp, 1.0, &in.acc);
            }
        }
        auto const pass = run(graph, true);
        CHECK(pass->num_tensors_privatized() == 0);
        CHECK(only_reason(*pass).find("executor") != std::string::npos);
    }

    SECTION("an allocation node touches the scratch") {
        // Created rather than declared, so the graph holds an allocation node for it.
        auto &tmp = graph.create_zero_runtime_tensor<double>("tmp", {6, 6}, true);
        {
            cg::CaptureGuard const guard(graph);
            for (auto const &A : in.ops) {
                cg::einsum("ij <- ik ; kj", 0.0, &tmp, 1.0, A, A);
                cg::axpby(1.0, tmp, 1.0, &in.acc);
            }
        }
        auto const pass = run(graph, false);
        CHECK(pass->num_tensors_privatized() == 0);
        CHECK(only_reason(*pass).find("allocation or free node") != std::string::npos);
    }

    SECTION("the scratch carries a value in") {
        // Read before it is first overwritten: the value it holds when the graph starts is used.
        auto &tmp = graph.declare_zero_runtime_tensor<double>("tmp", {6, 6}, true);
        {
            cg::CaptureGuard const guard(graph);
            cg::axpby(1.0, tmp, 1.0, &in.acc);
            for (auto const &A : in.ops) {
                cg::einsum("ij <- ik ; kj", 0.0, &tmp, 1.0, A, A);
                cg::axpby(1.0, tmp, 1.0, &in.acc);
            }
        }
        auto const pass = run(graph, false);
        CHECK(pass->num_tensors_privatized() == 0);
        CHECK(only_reason(*pass).find("carries a value in") != std::string::npos);
    }

    SECTION("an interior generation holds a node that cannot be rebuilt") {
        // A scale is not among the kinds the pass rebuilds onto a clone.
        auto &tmp = graph.declare_zero_runtime_tensor<double>("tmp", {6, 6}, true);
        {
            cg::CaptureGuard const guard(graph);
            for (auto const &A : in.ops) {
                cg::einsum("ij <- ik ; kj", 0.0, &tmp, 1.0, A, A);
                cg::scale(0.5, &tmp);
                cg::axpby(1.0, tmp, 1.0, &in.acc);
            }
        }
        auto const pass = run(graph, false);
        CHECK(pass->num_tensors_privatized() == 0);
        CHECK(only_reason(*pass).find("cannot be rebuilt") != std::string::npos);
    }
}

// Privatizing a scratch rebuilds the nodes that read it. The permute rebuild used to build its
// executor from the index lists alone, so a permute carrying P(ij) lost the antisymmetrizer and
// computed a plain transpose, and it read the capture-time scalars instead of the live ones.
TEST_CASE("ScratchPrivatization - a rebuilt permute keeps its permutation operator", "[ComputeGraph][ScratchPrivatization]") {
    Operands  in;
    cg::Graph graph("privatized_antisymmetrizer");
    auto     &tmp  = graph.declare_zero_runtime_tensor<double>("tmp", {6, 6}, true);
    auto     &anti = graph.declare_zero_runtime_tensor<double>("anti", {6, 6}, true);
    {
        cg::CaptureGuard const guard(graph);
        for (auto const &A : in.ops) {
            cg::einsum("ij <- ik ; kj", 0.0, &tmp, 1.0, A, A);
            cg::permute("i,j <- P(ij) j,i", 0.0, &anti, 1.0, tmp);
            cg::axpby(1.0, anti, 1.0, &in.acc);
        }
    }
    auto const pass = run(graph, false);
    REQUIRE(pass->num_tensors_privatized() >= 1);

    graph.apply<cg::passes::Materialization>();
    graph.execute();

    // acc = sum_A (P(ij) (A A)^T) = sum_A ((A A)^T - (A A)) elementwise, i.e. (AA)(j,i) - (AA)(i,j).
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 6; ++j) {
            double expected = 0.0;
            for (auto const &A : in.ops) {
                double ij = 0.0;
                double ji = 0.0;
                for (size_t k = 0; k < 6; ++k) {
                    ij += A(std::vector<size_t>{i, k}) * A(std::vector<size_t>{k, j});
                    ji += A(std::vector<size_t>{j, k}) * A(std::vector<size_t>{k, i});
                }
                expected += ji - ij;
            }
            REQUIRE(std::abs(in.acc(std::vector<size_t>{i, j}) - expected) < 1e-10);
        }
    }
}
