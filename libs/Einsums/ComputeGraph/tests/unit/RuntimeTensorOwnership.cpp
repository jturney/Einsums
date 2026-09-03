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

#include <fmt/format.h>

#include <cstddef>
#include <new>

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

TEST_CASE("Graph::create_zero_runtime_tensor_dynamic — allocated over a dead operand's address, it gets its own id",
          "[ComputeGraph][RuntimeTensor]") {
    // Capture adopts an operand's storage into a stand-in, so the caller's wrapper may die
    // while its handle, whose tensor_ptr is the wrapper's address, stays registered. A
    // graph-owned tensor of the same type allocated afterwards can land on that freed
    // address, and the id lookup behind the dynamic creators then has two handles at one
    // address to choose between. It used to take the first the tensor table iterated, which
    // is the dead operand's under the MSVC STL and the new tensor's under libc++ and
    // libstdc++: DistributiveFactoring's sum then took a consumed operand's id on Windows,
    // its zero-and-accumulate chain ran over that operand's storage, and one term of the
    // contraction went missing. The lookup now goes through the pointer index, which
    // registration keeps pointing at the tensor that lives at the address NOW.
    cg::Graph graph("recycled_address");

    auto              *wrapper      = new RuntimeTensor<double>("operand", std::vector<size_t>{4, 4});
    void const *const  dead_address = wrapper;
    cg::TensorId const dead_id      = graph.register_operand(*wrapper);
    REQUIRE(graph.find_tensor(dead_id)->owner != nullptr); // adopted, so the wrapper is free to die
    delete wrapper;

    // Allocate until one lands on the freed address. A free-list allocator hands the block
    // straight back on the first try; mimalloc serves a page's never-used blocks first and
    // recycles freed ones only once those run out, a few hundred allocations of this size,
    // so the bound is sized to exhaust a page rather than to be patient.
    bool reused = false;
    for (int i = 0; i < 4096 && !reused; ++i) {
        auto const t_name = fmt::format("sum_{}", i);
        auto       made   = graph.create_zero_runtime_tensor_dynamic(t_name, packed_gemm::ScalarType::Float64, {4, 4});
        REQUIRE(made);
        auto const [id, ptr] = made.value();
        reused               = ptr == dead_address;

        CAPTURE(i, reused);
        REQUIRE(id != dead_id);
        auto const *handle = graph.find_tensor(id);
        REQUIRE(handle != nullptr);
        CHECK(handle->tensor_ptr == ptr);
        CHECK(handle->is_intermediate);
        CHECK(handle->name == t_name);
    }
    if (!reused) {
        SKIP("the allocator never handed the freed address back, so the ambiguity this test pins did not arise");
    }
}

TEST_CASE("Graph::find_tensor_by_ptr — names the tensor that lives at an address now, not the one that died there",
          "[ComputeGraph][RuntimeTensor]") {
    // The allocator-independent form of the case above: two tensors built in one storage
    // slot, the first registered, destroyed, and replaced. Its handle keeps the address as
    // identity, so the table holds two handles at one address, and the lookup the creators
    // and declarers use has to return the live one.
    cg::Graph graph("same_slot");

    alignas(RuntimeTensor<double>) std::byte slot[sizeof(RuntimeTensor<double>)];
    auto                                    *first    = new (slot) RuntimeTensor<double>("first", std::vector<size_t>{4, 4});
    cg::TensorId const                       first_id = graph.register_operand(*first);
    first->~GeneralRuntimeTensor();

    auto *second = new (slot) RuntimeTensor<double>("second", std::vector<size_t>{4, 4});
    REQUIRE(static_cast<void *>(second) == static_cast<void *>(slot));
    cg::TensorId const second_id = graph.register_operand(*second);
    REQUIRE(second_id != first_id); // the liveness token told the two apart

    auto const *live = graph.find_tensor_by_ptr(second);
    REQUIRE(live != nullptr);
    CHECK(live->id == second_id);
    CHECK(live->name == "second");
    CHECK(graph.find_tensor(first_id)->name == "first"); // the dead one keeps its handle

    second->~GeneralRuntimeTensor();
}
