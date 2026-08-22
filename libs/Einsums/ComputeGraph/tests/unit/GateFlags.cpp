//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

TEST_CASE("GateFlags - construction, writes and bulk writes", "[ComputeGraph][GateFlags]") {
    cg::GateFlags flags(4);
    REQUIRE(flags.size() == 4);
    for (size_t i = 0; i < 4; i++) {
        CHECK_FALSE(flags.get(i));
    }

    cg::GateFlags const all(3, true);
    REQUIRE(all.size() == 3);
    CHECK(all.get(0));
    CHECK(all.get(2));

    flags.set(1, true);
    CHECK_FALSE(flags.get(0));
    CHECK(flags.get(1));

    flags.fill(true);
    CHECK(flags.get(0));
    CHECK(flags.get(3));

    // assign normalizes anything nonzero to one flag's worth of truth.
    flags.assign(std::vector<std::uint8_t>{0, 7, 0, 1});
    CHECK_FALSE(flags.get(0));
    CHECK(flags.get(1));
    CHECK_FALSE(flags.get(2));
    CHECK(flags.get(3));

    flags.resize(6, true);
    REQUIRE(flags.size() == 6);
    CHECK(flags.get(5));

    CHECK_THROWS_AS(flags.get(6), std::out_of_range);
    CHECK_THROWS_AS(flags.set(6, true), std::out_of_range);
}

TEST_CASE("GateFlags - a copy shares the array", "[ComputeGraph][GateFlags]") {
    cg::GateFlags       flags(2, false);
    cg::GateFlags const alias = flags; // NOLINT(performance-unnecessary-copy-initialization)
    flags.set(0, true);
    CHECK(alias.get(0));
}

TEST_CASE("GateFlags - a flag-gated conditional selects its branch", "[ComputeGraph][GateFlags]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("flag_cond");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }

    cg::GateFlags flags(1, true);
    auto [then_g, else_g] = graph.add_conditional_flag("branch", flags, 0);
    {
        cg::CaptureGuard const guard(then_g);
        cg::scale(2.0, &C);
    }
    {
        cg::CaptureGuard const guard(else_g);
        cg::scale(10.0, &C);
    }

    graph.execute();
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(std::abs(C(i, j) - 2.0 * A(i, j)) < 1e-12);
        }
    }

    // The same graph, replayed after the flag is flipped: no re-capture, no
    // re-plan, the node reads the array it was given.
    flags.set(0, false);
    graph.execute();
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(std::abs(C(i, j) - 10.0 * A(i, j)) < 1e-12);
        }
    }

    flags.set(0, true);
    graph.execute();
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            REQUIRE(std::abs(C(i, j) - 2.0 * A(i, j)) < 1e-12);
        }
    }
}

TEST_CASE("GateFlags - an index past the end reads false", "[ComputeGraph][GateFlags]") {
    auto C  = create_zero_tensor<double>("C", 2, 2);
    C(0, 0) = 1.0;

    cg::Graph           graph("flag_oob");
    cg::GateFlags const flags(1, true);
    auto [then_g, else_g] = graph.add_conditional_flag("branch", flags, 5);
    {
        cg::CaptureGuard const guard(then_g);
        cg::scale(2.0, &C);
    }

    graph.execute();
    REQUIRE(std::abs(C(0, 0) - 1.0) < 1e-12);
}

TEST_CASE("GateFlags - the gate outlives the handle it was built from", "[ComputeGraph][GateFlags]") {
    auto C  = create_zero_tensor<double>("C", 2, 2);
    C(0, 0) = 1.0;

    cg::Graph graph("flag_outlives");
    {
        cg::GateFlags const temporary(1, true);
        auto [then_g, else_g] = graph.add_conditional_flag("branch", temporary, 0);
        cg::CaptureGuard const guard(then_g);
        cg::scale(3.0, &C);
    }

    graph.execute();
    REQUIRE(std::abs(C(0, 0) - 3.0) < 1e-12);
}

TEST_CASE("GateFlags - a flag gate and a lambda gate compute the same bits", "[ComputeGraph][GateFlags]") {
    auto A = create_random_tensor<double>("A", 6, 6);

    auto with_flags = create_zero_tensor<double>("with_flags", 6, 6);
    auto with_lamda = create_zero_tensor<double>("with_lambda", 6, 6);

    // Four blocks, two of them open. Whatever the gate is spelled as, the
    // branches that run are the same branches on the same operands.
    std::vector<bool> const live{true, false, true, false};

    auto build = [&](cg::Graph &graph, auto &&gate_for, Tensor<double, 2> &out) {
        for (size_t block = 0; block < live.size(); block++) {
            auto [then_g, else_g] = gate_for(graph, block);
            cg::CaptureGuard const guard(then_g);
            cg::permute("ij <- ij", 1.0, &out, 1.0 + static_cast<double>(block), A);
        }
    };

    cg::Graph     flag_graph("by flags");
    cg::GateFlags flags(live.size(), false);
    for (size_t block = 0; block < live.size(); block++) {
        flags.set(block, live[block]);
    }
    build(
        flag_graph, [&](cg::Graph &g, size_t block) { return g.add_conditional_flag(fmt::format("block [{}]", block), flags, block); },
        with_flags);

    cg::Graph lambda_graph("by lambda");
    build(
        lambda_graph,
        [&](cg::Graph &g, size_t block) {
            return g.add_conditional(fmt::format("block [{}]", block), [&live, block]() { return live[block]; });
        },
        with_lamda);

    flag_graph.execute();
    lambda_graph.execute();

    for (size_t i = 0; i < 6; i++) {
        for (size_t j = 0; j < 6; j++) {
            REQUIRE(with_flags(i, j) == with_lamda(i, j));
        }
    }
}
