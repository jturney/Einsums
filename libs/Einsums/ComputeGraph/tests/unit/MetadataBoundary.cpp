//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file MetadataBoundary.cpp
/// @brief Contract tests for tensor-handle metadata across the parent/body boundary.
///
/// A loop body and each conditional branch are separate ``Graph`` objects with
/// their own ``tensors_map()``. When a parent-registered tensor is first touched
/// inside one, ``CaptureContext::get_or_register`` falls through to
/// ``register_tensor(make_handle(...))`` and the body receives a *fresh,
/// default* handle: ``is_intermediate``, residency, and distribution flags do
/// not cross. Nothing propagates back up either.
///
/// That loss is deliberate and load-bearing. FreeInsertion Part B,
/// InplaceOptimization, MemoryPlanning, and DeadNodeElimination all gate on the
/// body-map ``is_intermediate``, so a parent tensor arriving with
/// ``is_intermediate == false`` is skipped by all four - conservative by loss. A
/// body-local pass cannot see the parent's readers, so treating a parent tensor
/// as body-owned would let it be freed, aliased, or arena-overlapped while the
/// parent still needs it. Propagating metadata additionally unmasks
/// DeadNodeElimination's downward-only subtree walk, which today is saved from
/// deleting cross-scope producers only by this same gate.
///
/// These tests pin the contract so a future "obvious" propagation fix fails
/// here, loudly, instead of silently unmasking those bugs. If you intend to
/// change the contract, change these tests deliberately and audit the four
/// passes named above first. See the note on
/// ``CaptureContext::get_or_register``.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Find the handle registered for a given tensor object, or nullptr.
cg::TensorHandle const *handle_for(cg::Graph const &g, void const *tensor_ptr) {
    for (auto const &[tid, handle] : g.tensors_map()) {
        if (handle.tensor_ptr == tensor_ptr) {
            return &handle;
        }
    }
    return nullptr;
}

size_t count_nodes(cg::Graph const &g, cg::OpKind kind) {
    size_t n = 0;
    for (auto const &node : g.nodes()) {
        if (node.kind == kind) {
            n++;
        }
    }
    return n;
}

} // namespace

TEST_CASE("MetadataBoundary - parent intermediate loses is_intermediate inside a loop body", "[ComputeGraph][MetadataBoundary]") {
    auto A = create_random_tensor<double>("A", 8, 8);

    cg::Graph g("parent-into-body");
    // Parent-declared, graph-owned: is_intermediate == true on the parent side.
    auto &scratch = g.create_zero_tensor<double, 2>("scratch", 8, 8);
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &scratch, 1.0, A, A);
    }

    // First touch of `scratch` inside the body re-registers it there.
    auto &body = g.add_loop("body", /*max_iterations=*/1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::scale(0.5, &scratch);
    }

    auto const *parent_handle = handle_for(g, &scratch);
    REQUIRE(parent_handle != nullptr);
    CHECK(parent_handle->is_intermediate);

    auto const *body_handle = handle_for(body, &scratch);
    REQUIRE(body_handle != nullptr);

    // The contract: a fresh default handle, NOT a copy of the parent's.
    CHECK_FALSE(body_handle->is_intermediate);

    // The two handles are separate registrations in separate maps. Note that
    // TensorIds are per-graph, so the ids are NOT comparable across the
    // boundary - both are typically 1 here. Passes must resolve a body tid
    // through the owning body graph, never through the parent's map.
    CHECK(body_handle != parent_handle);

    // Shape metadata still comes from make_handle(), so it does match. Only
    // the graph-level policy/ownership fields are dropped.
    CHECK(body_handle->rank == parent_handle->rank);
    CHECK(body_handle->dims == parent_handle->dims);
    CHECK(body_handle->data_ptr == parent_handle->data_ptr);

    REQUIRE_NOTHROW(g.execute());
}

TEST_CASE("MetadataBoundary - parent intermediate loses is_intermediate inside a conditional branch", "[ComputeGraph][MetadataBoundary]") {
    auto A = create_random_tensor<double>("A", 4, 4);

    cg::Graph g("parent-into-branch");
    auto     &scratch = g.create_zero_tensor<double, 2>("scratch", 4, 4);
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &scratch, 1.0, A, A);
    }

    auto [then_g, else_g] = g.add_conditional("maybe", []() { return true; });
    {
        cg::CaptureGuard const guard(then_g);
        cg::scale(2.0, &scratch);
    }

    auto const *parent_handle = handle_for(g, &scratch);
    REQUIRE(parent_handle != nullptr);
    CHECK(parent_handle->is_intermediate);

    auto const *branch_handle = handle_for(then_g, &scratch);
    REQUIRE(branch_handle != nullptr);
    CHECK_FALSE(branch_handle->is_intermediate);

    REQUIRE_NOTHROW(g.execute());
}

TEST_CASE("MetadataBoundary - body-created intermediate does not propagate up to the parent", "[ComputeGraph][MetadataBoundary]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph g("body-to-parent");
    auto     &body = g.add_loop("body", /*max_iterations=*/1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        auto                  &tmp = body.create_zero_tensor<double, 2>("tmp", 4, 4);
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, A);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, tmp, A);
    }

    // The body owns `tmp` and knows it is an intermediate...
    cg::TensorHandle const *body_tmp = nullptr;
    for (auto const &[tid, handle] : body.tensors_map()) {
        if (handle.name == "tmp") {
            body_tmp = &handle;
        }
    }
    REQUIRE(body_tmp != nullptr);
    CHECK(body_tmp->is_intermediate);

    // ...and the parent has no handle for it at all. Metadata does not travel
    // upward any more than it travels downward.
    CHECK(handle_for(g, body_tmp->tensor_ptr) == nullptr);

    REQUIRE_NOTHROW(g.execute());
}

TEST_CASE("MetadataBoundary - FreeInsertion will not free a parent tensor from inside the body",
          "[ComputeGraph][MetadataBoundary][FreeInsertion]") {
    auto A = create_random_tensor<double>("A", 8, 8);
    auto C = create_zero_tensor<double>("C", 8, 8);

    cg::Graph g("no-body-free");
    auto     &scratch = g.create_zero_tensor<double, 2>("scratch", 8, 8);
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &scratch, 1.0, A, A);
    }

    auto &body = g.add_loop("body", /*max_iterations=*/1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::scale(0.5, &scratch);
    }

    // Parent still reads scratch after the loop.
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, scratch, A);
    }

    cg::passes::FreeInsertion fi(/*min_bytes=*/0);
    fi.run(g);

    // Part B gates on the body-map is_intermediate, which the parent tensor does
    // not carry. If metadata ever propagates, Part B would see an "owned"
    // intermediate whose body-local last use is the scale, and free it there -
    // before the post-loop einsum reads it. This assertion is what catches that.
    CHECK(count_nodes(body, cg::OpKind::Free) == 0);

    REQUIRE_NOTHROW(g.execute());
}
