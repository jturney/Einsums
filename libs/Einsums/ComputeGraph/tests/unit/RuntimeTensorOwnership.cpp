//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase B Workspace + Graph runtime-rank ownership tests.
//
// Validates the parallel "create_runtime_tensor" / "declare_runtime_tensor"
// APIs that allow callers to register graph-owned or workspace-owned
// RuntimeTensor instances. The typed create_/declare_ functions stay
// templated on (T, Rank); these new ones drop the rank from the type so
// callers can build runtime-shaped intermediates without committing the
// rank at compile time.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

TEST_CASE("Graph::create_runtime_tensor — eager allocation, arbitrary rank", "[ComputeGraph][RuntimeTensor]") {
    cg::Graph graph("rt_owner");

    // Rank that the typed create_tensor_dynamic dispatch couldn't reach (>4).
    auto &big = graph.create_runtime_tensor<double>("big", std::vector<size_t>{2, 2, 2, 2, 2});
    REQUIRE(big.rank() == 5);
    REQUIRE(big.dim(0) == 2);
    REQUIRE(big.dim(4) == 2);
    REQUIRE(big.size() == 32);

    // Rank-2: same eager-allocation semantics as create_tensor<T, 2>.
    auto &small = graph.create_zero_runtime_tensor<double>("small", std::vector<size_t>{3, 4});
    REQUIRE(small.rank() == 2);
    REQUIRE(small.dim(0) == 3);
    REQUIRE(small.dim(1) == 4);
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 4; j++)
            REQUIRE(small(i, j) == 0.0);

    // Both tensors registered an Alloc node with the graph.
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("Graph::create_runtime_tensor — usable in cg::einsum capture", "[ComputeGraph][RuntimeTensor]") {
    cg::Graph graph("rt_einsum_owned");
    auto      A = create_random_tensor<double>("A", 4, 3);
    auto      B = create_random_tensor<double>("B", 3, 5);

    {
        cg::CaptureGuard const guard(graph);
        auto                  &C = graph.create_zero_runtime_tensor<double>("C", std::vector<size_t>{4, 5});
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }

    // 1 alloc node + 1 einsum node.
    REQUIRE(graph.num_nodes() == 2);
    graph.execute();
}

TEST_CASE("Workspace::declare_runtime_tensor — deferred allocation lifecycle", "[ComputeGraph][RuntimeTensor]") {
    cg::Workspace ws("rt_ws");

    auto &t = ws.declare_runtime_tensor<double>("t", std::vector<size_t>{3, 4});

    // Shell tensor: dims/strides set, no backing data yet.
    REQUIRE(t.rank() == 2);
    REQUIRE(t.dim(0) == 3);
    REQUIRE(t.dim(1) == 4);
    REQUIRE_FALSE(t.is_materialized());

    // Workspace recorded the handle in deferred state.
    REQUIRE(ws.tensor_handles().size() == 1);
    REQUIRE(ws.tensor_handles()[0].alloc_state == cg::AllocState::Deferred);

    // The handle's materialize_fn invokes the tensor's materialize(),
    // afterwards data() returns a valid pointer.
    ws.tensor_handles()[0].materialize_fn();
    REQUIRE(t.is_materialized());
    REQUIRE(t.data() != nullptr);
}

TEST_CASE("Workspace::declare_zero_runtime_tensor — zero init via handle", "[ComputeGraph][RuntimeTensor]") {
    cg::Workspace ws("rt_ws_zero");

    auto &t = ws.declare_zero_runtime_tensor<float>("zero", std::vector<size_t>{2, 3});
    REQUIRE_FALSE(t.is_materialized());

    auto const &h = ws.tensor_handles().back();
    REQUIRE(h.init_kind == cg::InitKind::Zero);

    h.zero_fn(); // materialize + zero
    REQUIRE(t.is_materialized());
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 3; j++)
            REQUIRE(t(i, j) == 0.0f);
}
