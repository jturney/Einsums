//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("Rebind - basic tensor rebind", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    // Capture graph with A1
    cg::Graph graph("rebind_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    // Execute with A1
    graph.execute();

    auto C_ref1 = create_zero_tensor<double>("Cr1", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref1, Indices{i, k}, A1, Indices{k, j}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref1(ii, jj)) < 1e-12);
        }
    }

    // Rebind A1 → A2 (one line!)
    graph.rebind(A1, A2);

    // Execute again, should now use A2
    C.zero();
    graph.execute();

    auto C_ref2 = create_zero_tensor<double>("Cr2", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref2, Indices{i, k}, A2, Indices{k, j}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref2(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Rebind - dimension mismatch throws", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 5, 3); // Different first dim!
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebind_mismatch");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    REQUIRE_THROWS_AS(graph.rebind(A1, A2), std::invalid_argument);
}

TEST_CASE("Rebind - scale operation", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 3, 3);
    auto A2 = create_random_tensor<double>("A2", 3, 3);

    cg::Graph graph("rebind_scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A1);
    }

    graph.execute();

    auto A2_copy = Tensor<double, 2>(A2);
    graph.rebind(A1, A2);
    graph.execute();

    // A2 should now be doubled
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(A2(ii, jj) - 2.0 * A2_copy(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Rebind - string einsum", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebind_string");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A1, B);
    }

    graph.execute();

    graph.rebind(A1, A2);
    C.zero();
    graph.execute();

    auto C_ref = create_zero_tensor<double>("Cr", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A2, Indices{k, j}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("update_prefactors - correct after CSE removes an earlier einsum", "[ComputeGraph][Rebind][CSE]") {
    // Regression: update_prefactors used to locate the EinsumParams by
    // counting einsum nodes before the target in _nodes and indexing
    // _params_store by that ordinal. Any pass that removes or reorders
    // einsum nodes (CSE here) desynced the ordinal, so the update landed on
    // a dead einsum's params and the target silently kept its old scalars.
    // The params handle now lives on the EinsumDescriptor itself.
    auto A   = create_random_tensor<double>("A", 4, 3);
    auto B   = create_random_tensor<double>("B", 3, 4);
    auto C   = create_zero_tensor<double>("C", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);

    cg::Graph graph("update_pf_after_cse");
    auto     &D = graph.create_tensor<double, 2>("D", 4, 4); // graph-owned duplicate
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);   // survivor
        cg::einsum("ik;kj->ij", &D, A, B);   // duplicate, removed by CSE
        cg::einsum("ik;kj->ij", &out, C, D); // target of update_prefactors
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 3); // Alloc(D) + 2 einsums

    // The target einsum is now the last node; scale its contribution by 2.
    cg::NodeId const out_id = graph.nodes()[graph.num_nodes() - 1].id;
    graph.update_prefactors(out_id, 0.0, 2.0);

    graph.execute();

    auto AB = create_zero_tensor<double>("AB", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &AB, Indices{i, k}, A, Indices{k, j}, B);
    auto out_ref = create_zero_tensor<double>("OUTref", 4, 4);
    tensor_algebra::einsum(0.0, Indices{i, j}, &out_ref, 2.0, Indices{i, k}, AB, Indices{k, j}, AB);

    double max_abs = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            max_abs = std::max(max_abs, std::abs(out(ii, jj)));
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-12);
        }
    }
    REQUIRE(max_abs > 1e-10);
}

TEST_CASE("update_prefactors - changes computation", "[ComputeGraph][Rebind]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_random_tensor<double>("C", 3, 3);

    cg::Graph graph("update_pf");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    // Get the einsum node's ID
    cg::NodeId const einsum_id = graph.nodes()[0].id;

    // Execute with c_pf=0, ab_pf=1
    auto C_save = Tensor<double, 2>(C);
    graph.execute();
    auto C_v1 = Tensor<double, 2>(C);

    // Update prefactors to c_pf=1, ab_pf=2
    graph.update_prefactors(einsum_id, 1.0, 2.0);

    // Execute again, should compute C = 1*C + 2*A*B
    graph.execute();

    // Verify: C_new = C_v1 + 2*A*B
    auto C_ref = Tensor<double, 2>(C_v1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &C_ref, 2.0, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Rebind - the pointer index follows the rebound tensor", "[ComputeGraph][Rebind]") {
    // Every by-address lookup on a Graph goes through the pointer index, and a
    // rebind moves what a slot names. The index used to be left behind: it went
    // on naming the tensor the graph was captured over and answered "not
    // registered" for the one it had just been rebound to, so a caller who
    // rebound and then asked the graph about the new tensor was told the graph
    // had never seen it.
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebind_ptr_index");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    cg::TensorId const a_id = graph.find_tensor_id_by_ptr(&A1);
    REQUIRE(a_id != 0);

    graph.rebind(A1, A2);

    // The index answers about the storage the slot names NOW,
    CHECK(graph.find_tensor_id_by_ptr(&A2) == a_id);
    // and stops claiming the address it was moved off is registered.
    CHECK(graph.find_tensor_id_by_ptr(&A1) == cg::TensorId{0});
    // The handle and the index agree, which is the invariant every by-address
    // lookup relies on.
    auto const *handle = graph.find_tensor_by_ptr(&A2);
    REQUIRE(handle != nullptr);
    CHECK(handle->id == a_id);
    CHECK(handle->tensor_ptr == static_cast<void *>(&A2));
}

TEST_CASE("Rebind - a rebound tensor answers the liveness-checked lookup", "[ComputeGraph][Rebind]") {
    // The by-object metadata readers pair the address with the caller's liveness
    // token, so that a tensor allocated on top of a dead one does not inherit its
    // id. A rebind moved the address and left the token, which made the pair name
    // a tensor that never existed: the new object with the previous object's
    // lifetime. The graph then disowned the very tensor it had just been rebound
    // to, and reading an annotation off it threw "is not registered".
    RuntimeTensor<double> A1("A1", std::vector<size_t>{4, 3});
    RuntimeTensor<double> A2("A2", std::vector<size_t>{4, 3});
    RuntimeTensor<double> B("B", std::vector<size_t>{3, 5});
    RuntimeTensor<double> C("C", std::vector<size_t>{4, 5});
    A1.zero();
    A2.zero();
    B.zero();
    C.zero();

    cg::Graph graph("rebind_live_lookup");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    cg::TensorId const a_id = graph.live_tensor_id_by_ptr(&A1, A1.liveness_token());
    REQUIRE(a_id != 0);

    graph.rebind(A1, A2);

    CHECK(graph.live_tensor_id_by_ptr(&A2, A2.liveness_token()) == a_id);
    CHECK(graph.live_tensor_id_by_ptr(&A1, A1.liveness_token()) == cg::TensorId{0});
}
